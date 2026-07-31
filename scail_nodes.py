"""Consolidated SCAIL-2 preparation and long-video inference nodes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

try:
    from .scail_core import (
        prepare_scail_window,
        reinhard_color_transfer,
        resize_image_batch,
        sample_scail_window,
        validate_image_batch,
    )
except ImportError:  # Weight-free tests import modules directly from the pack root.
    from scail_core import (
        prepare_scail_window,
        reinhard_color_transfer,
        resize_image_batch,
        sample_scail_window,
        validate_image_batch,
    )


SCAIL_DEFAULT_SHIFT = 5.0
SCAIL_DEFAULT_CHUNK_LENGTH = 81
SCAIL_DEFAULT_OVERLAP = 5

SEED_FIXED = "fixed"
SEED_INCREMENT = "increment per chunk"

DECODE_NORMAL = "normal"
DECODE_TILED = "tiled"


@dataclass(frozen=True)
class SCAILChunkPlan:
    """A temporal plan whose chunk lengths all satisfy SCAIL's ``4n+1`` rule."""

    input_frames: int
    requested_frames: int
    effective_frames: int
    chunk_lengths: tuple[int, ...]

    @property
    def dropped_tail_frames(self) -> int:
        return self.requested_frames - self.effective_frames


def _is_four_n_plus_one(value: int) -> bool:
    return value >= 1 and (value - 1) % 4 == 0


def plan_scail_chunks(
    frame_count: int,
    chunk_length: int = SCAIL_DEFAULT_CHUNK_LENGTH,
    overlap: int = SCAIL_DEFAULT_OVERLAP,
    max_frames: int = 0,
) -> SCAILChunkPlan:
    """Plan fixed-overlap chunks without generating an invalid temporal length."""

    frame_count = int(frame_count)
    chunk_length = int(chunk_length)
    overlap = int(overlap)
    max_frames = int(max_frames)

    if frame_count < 1:
        raise ValueError("SCAIL pose_video must contain at least one frame.")
    if not _is_four_n_plus_one(chunk_length):
        raise ValueError(
            f"SCAIL chunk_length must be 4n+1 frames; received {chunk_length}."
        )
    if not _is_four_n_plus_one(overlap):
        raise ValueError(f"SCAIL overlap must be 4n+1 frames; received {overlap}.")
    if overlap >= chunk_length:
        raise ValueError(
            f"SCAIL overlap ({overlap}) must be smaller than chunk_length "
            f"({chunk_length})."
        )
    if max_frames < 0:
        raise ValueError("SCAIL max_frames cannot be negative.")

    requested = min(frame_count, max_frames) if max_frames > 0 else frame_count
    effective = ((requested - 1) // 4) * 4 + 1

    if effective <= chunk_length:
        lengths = (effective,)
    else:
        step = chunk_length - overlap
        extension_count = math.ceil((effective - chunk_length) / step)
        final_length = effective - step * extension_count
        if final_length <= overlap or not _is_four_n_plus_one(final_length):
            raise ValueError(
                "Unable to construct a valid SCAIL extension plan from the "
                "requested chunk length and overlap."
            )
        lengths = (chunk_length,) * extension_count + (final_length,)

    return SCAILChunkPlan(frame_count, requested, effective, lengths)


def _model_sampling(model: Any) -> Any:
    try:
        sampling = model.get_model_object("model_sampling")
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "SCAIL MODEL does not expose model_sampling. Apply ModelSamplingSD3 "
            "upstream before connecting it."
        ) from exc
    if sampling is None:
        raise ValueError(
            "SCAIL MODEL does not expose model_sampling. Apply ModelSamplingSD3 "
            "upstream before connecting it."
        )
    return sampling


def _validate_expected_shift(model_sampling: Any, expected_shift: float) -> float | None:
    expected_shift = float(expected_shift)
    if not math.isfinite(expected_shift) or expected_shift < 0.0:
        raise ValueError("expected_shift must be finite and non-negative.")
    if expected_shift == 0.0:
        return None

    raw_shift = getattr(model_sampling, "shift", None)
    try:
        actual_shift = float(raw_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SCAIL MODEL has no readable shift. Apply ModelSamplingSD3 upstream "
            f"with shift {expected_shift:g}, or set expected_shift to 0 to skip validation."
        ) from exc
    if not math.isfinite(actual_shift) or not math.isclose(
        actual_shift, expected_shift, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(
            f"SCAIL MODEL shift is {actual_shift:g}; expected ModelSamplingSD3 "
            f"shift {expected_shift:g}."
        )
    return actual_shift


def build_scail_sigmas(
    model_sampling: Any,
    scheduler: str,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    """Mirror ComfyUI's BasicScheduler so the all-in-one node stays reproducible."""

    steps = int(steps)
    denoise = float(denoise)
    if steps < 1:
        raise ValueError("SCAIL steps must be at least 1.")
    if not math.isfinite(denoise) or not 0.0 < denoise <= 1.0:
        raise ValueError("SCAIL denoise must be greater than 0 and at most 1.")

    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(
        model_sampling, scheduler, total_steps
    ).cpu()
    sigmas = sigmas[-(steps + 1) :]

    if sigmas.ndim != 1 or sigmas.numel() != steps + 1:
        raise ValueError(
            f"SCAIL scheduler returned {sigmas.numel()} sigma values for {steps} steps."
        )
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("SCAIL sigma schedule contains NaN or infinite values.")
    if bool(torch.any(sigmas[1:] > sigmas[:-1])):
        raise ValueError("SCAIL sigma schedule must be non-increasing.")
    if not math.isclose(float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("SCAIL sigma schedule must terminate at zero.")
    return sigmas


def _decode_video(vae: Any, latent_samples: torch.Tensor, decode_mode: str) -> torch.Tensor:
    if decode_mode == DECODE_TILED:
        images = vae.decode_tiled(latent_samples)
    elif decode_mode == DECODE_NORMAL:
        images = vae.decode(latent_samples)
    else:
        raise ValueError(f"Unknown SCAIL decode mode: {decode_mode}")

    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    if images.ndim != 4:
        raise ValueError(
            "SCAIL VAE decode must return IMAGE data shaped [frames, height, width, channels]."
        )
    return images


class WanSCAILMediaPrep(io.ComfyNode):
    """Prepare full driving and reference batches at one SCAIL-safe geometry."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILMediaPrep",
            display_name="Wan SCAIL-2 Media Prep",
            search_aliases=[
                "scail resize",
                "scail video prep",
                "scail reference prep",
            ],
            category="image/transform/wan",
            description=(
                "Resizes the complete driving-video batch and reference batch to "
                "one multiple-of-32 SCAIL geometry without slicing the timeline."
            ),
            inputs=[
                io.Image.Input("pose_video"),
                io.Image.Input("reference_image"),
                io.Int.Input(
                    "width",
                    default=640,
                    min=32,
                    max=8192,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=960,
                    min=32,
                    max=8192,
                    step=32,
                ),
                io.Combo.Input(
                    "resize_method",
                    options=["lanczos", "bicubic", "bilinear", "area"],
                    default="lanczos",
                ),
                io.Combo.Input(
                    "pose_fit",
                    options=["center crop", "stretch"],
                    default="center crop",
                    tooltip="Center crop preserves body proportions in the driving video.",
                ),
                io.Combo.Input(
                    "reference_fit",
                    options=["stretch", "center crop"],
                    default="stretch",
                    tooltip="SCAIL reference vision conditioning was trained with stretch resize.",
                ),
            ],
            outputs=[
                io.Image.Output("pose_video"),
                io.Image.Output("reference_image"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("diagnostics"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pose_video,
        reference_image,
        width,
        height,
        resize_method,
        pose_fit,
        reference_fit,
    ):
        validate_image_batch(pose_video, "pose_video")
        validate_image_batch(reference_image, "reference_image")
        width, height = int(width), int(height)
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError(
                f"SCAIL width and height must be multiples of 32; received "
                f"{width}x{height}."
            )
        prepared_pose = resize_image_batch(
            pose_video, width, height, resize_method, pose_fit
        )
        prepared_reference = resize_image_batch(
            reference_image, width, height, resize_method, reference_fit
        )
        diagnostics = (
            f"SCAIL media {width}x{height} | pose "
            f"{pose_video.shape[0]} frame(s) {pose_fit} | reference "
            f"{reference_image.shape[0]} view(s) {reference_fit} | {resize_method}"
        )
        return io.NodeOutput(
            prepared_pose,
            prepared_reference,
            width,
            height,
            diagnostics,
        )

    prepare = execute


class WanSCAILAutoExtendSampler(io.ComfyNode):
    """Generate the complete driving video through bounded SCAIL-2 windows."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILAutoExtendSampler",
            display_name="Wan SCAIL-2 Auto-Extend Sampler",
            search_aliases=[
                "scail sampler",
                "scail auto extend",
                "wan scail",
                "scail long video",
            ],
            category="model/sampling/wan",
            description=(
                "Pack-native SCAIL conditioning, sampling, VAE decode, overlap "
                "anchoring, and color stabilization for the full driving video. "
                "Defaults match the proven 81/5 SCAIL-2 workflow."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Wan 2.1 SCAIL-2 model with ModelSamplingSD3 applied upstream."
                    ),
                ),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input(
                    "pose_video",
                    tooltip=(
                        "Driving video. Output is trimmed to the nearest valid 4n+1 "
                        "frame count."
                    ),
                ),
                io.Image.Input(
                    "pose_video_mask",
                    optional=True,
                    tooltip=(
                        "Colored identity mask from Wan SCAIL-2 Identity Control."
                    ),
                ),
                io.Image.Input(
                    "reference_image",
                    optional=True,
                    tooltip="Reference character image or multi-view image batch.",
                ),
                io.Image.Input(
                    "reference_image_mask",
                    optional=True,
                    tooltip="Colored mask corresponding to reference_image.",
                ),
                io.ClipVisionOutput.Input(
                    "clip_vision_output",
                    optional=True,
                    tooltip="Optional CLIP vision features for the reference image.",
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=32,
                    max=8192,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=896,
                    min=32,
                    max=8192,
                    step=32,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "steps",
                    default=6,
                    min=1,
                    max=10000,
                    tooltip="The working supplied workflow uses 6 steps.",
                ),
                io.Float.Input(
                    "cfg",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.SAMPLER_NAMES,
                    default="euler",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=comfy.samplers.SCHEDULER_NAMES,
                    default="simple",
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "expected_shift",
                    default=SCAIL_DEFAULT_SHIFT,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Validates upstream ModelSamplingSD3. The supplied workflow "
                        "uses shift 5. Set 0 only to disable validation."
                    ),
                ),
                io.Int.Input(
                    "chunk_length",
                    default=SCAIL_DEFAULT_CHUNK_LENGTH,
                    min=9,
                    max=1024,
                    step=4,
                    advanced=True,
                    tooltip="SCAIL-2 is trained for 81-frame chunks.",
                ),
                io.Int.Input(
                    "overlap",
                    default=SCAIL_DEFAULT_OVERLAP,
                    min=1,
                    max=81,
                    step=4,
                    advanced=True,
                    tooltip="SCAIL-2 is trained with a 5-frame extension anchor.",
                ),
                io.Int.Input(
                    "max_frames",
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    advanced=True,
                    tooltip="0 processes the full pose video; otherwise this is a hard cap.",
                ),
                io.Combo.Input(
                    "seed_mode",
                    options=[SEED_FIXED, SEED_INCREMENT],
                    default=SEED_FIXED,
                    advanced=True,
                    tooltip=(
                        "Fixed matches the supplied workflow and usually gives the "
                        "best cross-window continuity."
                    ),
                ),
                io.Combo.Input(
                    "decode_mode",
                    options=[DECODE_NORMAL, DECODE_TILED],
                    default=DECODE_NORMAL,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "color_transfer",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "Reinhard LAB-match extension chunks to the prior frame to "
                        "reduce color drift."
                    ),
                ),
                io.Float.Input(
                    "color_transfer_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "replacement_mode",
                    default=True,
                    tooltip=(
                        "True keeps the driving background and replaces masked identities; "
                        "False animates the reference scene."
                    ),
                ),
                io.Float.Input(
                    "pose_strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "pose_start",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "pose_end",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "add_noise",
                    default=True,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="Complete overlap-free SCAIL video on CPU.",
                ),
                io.Int.Output(display_name="frame_count"),
                io.String.Output(
                    display_name="diagnostics",
                    tooltip="Resolved temporal plan, sampling settings, and chunk timings.",
                ),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        vae,
        pose_video,
        width,
        height,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        expected_shift,
        chunk_length,
        overlap,
        max_frames,
        seed_mode,
        decode_mode,
        color_transfer,
        color_transfer_strength,
        replacement_mode,
        pose_strength,
        pose_start,
        pose_end,
        add_noise,
        pose_video_mask=None,
        reference_image=None,
        reference_image_mask=None,
        clip_vision_output=None,
    ):
        validate_image_batch(pose_video, "pose_video")
        width, height = int(width), int(height)
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError(
                f"SCAIL width and height must be multiples of 32; received "
                f"{width}x{height}."
            )
        plan = plan_scail_chunks(
            pose_video.shape[0],
            chunk_length=chunk_length,
            overlap=overlap,
            max_frames=max_frames,
        )
        if pose_video_mask is not None:
            validate_image_batch(pose_video_mask, "pose_video_mask")
            if pose_video_mask.shape[0] < plan.effective_frames:
                raise ValueError(
                    "SCAIL pose_video_mask has fewer frames than the planned output. "
                    "Track/render the full driving video before sampling."
                )
        if reference_image is not None:
            validate_image_batch(reference_image, "reference_image")
        if reference_image_mask is not None:
            validate_image_batch(reference_image_mask, "reference_image_mask")
            if reference_image is None:
                raise ValueError(
                    "reference_image_mask requires a connected reference_image."
                )
        if seed_mode not in (SEED_FIXED, SEED_INCREMENT):
            raise ValueError(f"Unknown SCAIL seed mode: {seed_mode}")
        if not 0.0 <= float(pose_start) <= float(pose_end) <= 1.0:
            raise ValueError("SCAIL pose range must satisfy 0 <= pose_start <= pose_end <= 1.")
        if not math.isfinite(float(cfg)) or float(cfg) < 0.0:
            raise ValueError("SCAIL cfg must be finite and non-negative.")

        model_sampling = _model_sampling(model)
        actual_shift = _validate_expected_shift(model_sampling, expected_shift)
        sigmas = build_scail_sigmas(model_sampling, scheduler, steps, denoise)
        sampler = comfy.samplers.sampler_object(sampler_name)

        chunks: list[torch.Tensor] = []
        previous_frames = None
        video_frame_offset = 0
        chunk_timings: list[float] = []
        progress = comfy.utils.ProgressBar(len(plan.chunk_lengths))
        total_started = time.perf_counter()

        for chunk_index, length in enumerate(plan.chunk_lengths):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk_started = time.perf_counter()
            chunk_seed = int(seed)
            if seed_mode == SEED_INCREMENT:
                chunk_seed = (chunk_seed + chunk_index) & 0xFFFFFFFFFFFFFFFF

            (
                pos_chunk,
                neg_chunk,
                latent,
                video_frame_offset,
            ) = prepare_scail_window(
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                pose_strength=float(pose_strength),
                pose_start=float(pose_start),
                pose_end=float(pose_end),
                video_frame_offset=video_frame_offset,
                previous_frame_count=int(overlap),
                replacement_mode=bool(replacement_mode),
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                pose_video=pose_video,
                pose_video_mask=pose_video_mask,
                reference_image_mask=reference_image_mask,
                previous_frames=previous_frames,
            )

            denoised = sample_scail_window(
                model=model,
                positive=pos_chunk,
                negative=neg_chunk,
                latent=latent,
                sampler=sampler,
                sigmas=sigmas,
                seed=chunk_seed,
                cfg=float(cfg),
                add_noise=bool(add_noise),
            )
            images = _decode_video(vae, denoised["samples"], decode_mode)
            if images.shape[0] != length:
                raise ValueError(
                    f"SCAIL chunk {chunk_index + 1} decoded {images.shape[0]} "
                    f"frames; expected {length}."
                )

            if chunk_index == 0:
                contribution = images
            else:
                if images.shape[0] <= overlap:
                    raise ValueError(
                        f"SCAIL extension decoded only {images.shape[0]} frames; "
                        f"cannot remove the {overlap}-frame overlap."
                    )
                contribution = images[overlap:]
                if color_transfer and float(color_transfer_strength) > 0.0:
                    contribution = reinhard_color_transfer(
                        contribution,
                        previous_frames[-1:],
                        float(color_transfer_strength),
                    )

            if contribution.shape[0] < 1:
                raise ValueError("SCAIL produced an empty chunk contribution.")
            chunks.append(contribution.detach().to("cpu"))
            previous_frames = contribution[-int(overlap) :].detach()
            chunk_timings.append(time.perf_counter() - chunk_started)
            progress.update_absolute(chunk_index + 1, len(plan.chunk_lengths))

        images = torch.cat(chunks, dim=0)
        if images.shape[0] != plan.effective_frames:
            raise ValueError(
                f"SCAIL chunk stitching produced {images.shape[0]} frames; "
                f"the temporal plan expected {plan.effective_frames}."
            )

        total_seconds = time.perf_counter() - total_started
        shift_text = "unchecked" if actual_shift is None else f"{actual_shift:g}"
        lengths_text = ",".join(str(length) for length in plan.chunk_lengths)
        timings_text = ",".join(f"{seconds:.2f}s" for seconds in chunk_timings)
        trim_text = (
            f" | trimmed tail {plan.dropped_tail_frames}"
            if plan.dropped_tail_frames
            else ""
        )
        diagnostics = (
            f"SCAIL-2 | {plan.effective_frames}/{plan.input_frames} frames"
            f"{trim_text} | chunks [{lengths_text}] overlap {int(overlap)} | "
            f"{int(steps)} steps {sampler_name}/{scheduler} denoise {float(denoise):g} | "
            f"CFG {float(cfg):g} | shift {shift_text} | seed {int(seed)} {seed_mode} | "
            f"decode {decode_mode} | chunk wall [{timings_text}] total {total_seconds:.2f}s"
        )
        return io.NodeOutput(images, int(images.shape[0]), diagnostics)

    generate = execute


__all__ = [
    "DECODE_NORMAL",
    "DECODE_TILED",
    "SCAILChunkPlan",
    "SCAIL_DEFAULT_CHUNK_LENGTH",
    "SCAIL_DEFAULT_OVERLAP",
    "SCAIL_DEFAULT_SHIFT",
    "SEED_FIXED",
    "SEED_INCREMENT",
    "WanSCAILAutoExtendSampler",
    "WanSCAILMediaPrep",
    "build_scail_sigmas",
    "plan_scail_chunks",
]
