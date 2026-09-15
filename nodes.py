
import math

import torch
import torch.nn.functional as F
import torchaudio

import nodes
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
import node_helpers
from  comfy_extras.nodes_minimax_h3 import temporal_shape
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import ComfyExtension, io


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24

def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]

def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count

def _extend_av_latent(video, audio, extend_length):
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("Video latent must be a 5D tensor with 24 channels")
    if audio.ndim != 4 or audio.shape[1] != 32:
        raise ValueError("Audio latent must be a 4D tensor with 32 channels")

    new_video = torch.cat([video, video[:, :, -1:, :, :].repeat(1, 1, extend_length, 1, 1)], dim=2)
    new_audio = torch.cat([audio, audio[:, :, :, -1:].repeat(1, 1, 1, extend_length)], dim=3)

    return (new_video, new_audio)


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))

class MiniMaxH3AddLatentGuide(io.ComfyNode):
    """Anchor image and/or audio guides at an arbitrary pixel frame of the target video."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AddLatentGuide",
            display_name="Add Latent Guide for MiniMax H3",
            category="model/conditioning/minimax",
            description="Anchor an image, a short clip, audio, or a clip with its soundtrack at any frame of a MiniMax H3 video. Chain several nodes to anchor several frames.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE, needed when an image is connected."),
                io.Latent.Input("latent"),
                io.Latent.Input("anchor_latent", optional=True, tooltip="Optional latent to anchor the image/audio to."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Soundtrack to anchor starting at the same frame index, cropped to the video's remaining duration."),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Frame index to anchor the image or the clip's first frame at. Negative values are counted from the end of the video."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, frame_idx, audio=None, anchor_latent=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24:
            raise ValueError("MiniMaxH3AddLatentGuide expects a MiniMax H3 AV latent")
        video = samples.tensors[0]
        height = video.shape[3] * 16
        width = video.shape[4] * 16
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))

        guide_frames = 1
        resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved_frame_index < 0 or resolved_frame_index + guide_frames > frame_count:
            if guide_frames == 1:
                raise ValueError("frame_idx {} is outside the video's {} frames".format(frame_idx, frame_count))
            raise ValueError("a {} frame guide clip at frame_idx {} does not fit in the video's {} frames".format(
                guide_frames, frame_idx, frame_count))

        keyframe = {"resolved_frame_index": resolved_frame_index}
        if anchor_latent is not None:
            # check if only one frame
            anchor_samples = anchor_latent["samples"]
            if anchor_samples.is_nested:
                anchor_t = anchor_samples.tensors[0]
            else:
                anchor_t = anchor_samples
            keyframe["latent"] = anchor_t[:, :, :guide_frames]

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        return io.NodeOutput(positive)
    
class ExtendAVLatent(io.ComfyNode):
    """Extend a MiniMax H3 AV latent to a new length by repeating the last frame and audio latent"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExtendAVLatent",
            display_name="Extend AV Latent",
            category="model/conditioning/minimax",
            inputs=[
                io.Latent.Input("av_latent"),#NestedTensor
                io.Int.Input("extend_length", min=1, tooltip="How many frames to extend the latent by."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, av_latent, extend_length) -> io.NodeOutput:

        latents = av_latent["samples"].unbind()
        video_latent = av_latent.copy()
        video_latent["samples"] = latents[0]
        audio_latent = av_latent.copy()
        audio_latent["samples"] = latents[1]

        output = {}
        output.update(video_latent)
        output.update(audio_latent)
        video_noise_mask = video_latent.get("noise_mask", None)
        audio_noise_mask = audio_latent.get("noise_mask", None)

        new_latent = _extend_av_latent(video_latent["samples"], audio_latent["samples"], extend_length)

        if video_noise_mask is not None or audio_noise_mask is not None:
            if video_noise_mask is None:
                video_noise_mask = torch.ones_like(video_latent["samples"])
            if audio_noise_mask is None:
                audio_noise_mask = torch.ones_like(audio_latent["samples"])
            output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))


        output["samples"] = comfy.nested_tensor.NestedTensor(new_latent)

        return io.NodeOutput(output)


class MiniMaxH3LatentReferenceToVideo(io.ComfyNode):
    """ref2va: prompt + reference images / videos / audio -> conditioning + AV latent.

    References enter the presentation in fixed order: images, then videos (each
    soundtrack's <Audio j> label right before its <Video k>), then standalone
    audio. Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i> / <Video k> / <Audio j>.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentReferenceToVideo",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting.",
            display_name="MiniMax H3 Latent Reference to Video",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so 'max' can be several times slower."),
                io.Image.Input("image_one", optional=True),
                io.Image.Input("image_two", optional=True),
                io.Image.Input("image_three", optional=True),
                io.Image.Input("video_one", optional=True),
                io.Image.Input("video_two", optional=True),
                io.Audio.Input("audio_one", optional=True),
                io.Latent.Input("latent_one", optional=True, tooltip="Optional latent to anchor the first reference image to."),
                io.Latent.Input("latent_two", optional=True, tooltip="Optional latent to anchor the second reference image to."),
                io.Latent.Input("latent_three", optional=True, tooltip="Optional latent to anchor the third reference image to."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, ref_image_size="match",
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None, image_one=None, image_two=None, image_three=None, video_one=None, video_two=None, audio_one=None, latent_one=None, latent_two=None, latent_three=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # for the DiT payload, same order
        
        if image_one is not None:
            ref_images["image_one"] = image_one
        if image_two is not None:
            ref_images["image_two"] = image_two
        if image_three is not None:
            ref_images["image_three"] = image_three
        if latent_one is not None:
            ref_video = latent_one["samples"].clone()
            ref_blocks.append({"kind": "image", "latent_t": ref_video.shape[2], "latent_h": ref_video.shape[3], "latent_w": ref_video.shape[4], "latent": ref_video})
        if latent_two is not None:
            ref_video = latent_two["samples"].clone()
            ref_blocks.append({"kind": "image", "latent_t": ref_video.shape[2], "latent_h": ref_video.shape[3], "latent_w": ref_video.shape[4], "latent": ref_video})
        if latent_three is not None:
            ref_video = latent_three["samples"].clone()
            ref_blocks.append({"kind": "image", "latent_t": ref_video.shape[2], "latent_h": ref_video.shape[3], "latent_w": ref_video.shape[4], "latent": ref_video})
        if video_one is not None:
            ref_videos["video_one"] = video_one
        if video_two is not None:
            ref_videos["video_two"] = video_two
        if audio_one is not None:
            ref_video_audios["audio_one"] = audio_one

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                # aspect-preserving scale (down only) to the generation's pixel area
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            if vae is not None:
                z = vae.encode(resized)
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
        

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5 and n is not 1:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n != 1 and n % 17 != 5:
                n -= 1
            frames = frames[:n]
            if soundtrack is not None:
                # the soundtrack gets its own <Audio j> label, emitted before <Video k>
                ref_items.append({"type": "audio"})
            # Qwen sees the video at 2 fps with timestamps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            if vae is None:
                continue
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None and audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            ref_items.append({"type": "audio"})
            if audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3AddLatentGuide": MiniMaxH3AddLatentGuide,
    "ExtendAVLatent": ExtendAVLatent,
    "MiniMaxH3LatentReferenceToVideo": MiniMaxH3LatentReferenceToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3AddLatentGuide": "MiniMax H3 Add Latent Guide",
    "ExtendAVLatent": "Extend AV Latent",
    "MiniMaxH3LatentReferenceToVideo": "MiniMax H3 Latent Reference to Video"
}

