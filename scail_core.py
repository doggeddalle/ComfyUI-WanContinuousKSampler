"""Internal SCAIL-2 conditioning, mask, sampling, and color helpers.

These functions are intentionally not registered as ComfyUI nodes.  Public
nodes live in ``scail_nodes.py`` and ``scail_identity.py`` so workflows get a
small, coherent surface while the implementation remains independently
testable.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview
import node_helpers
from comfy.ldm.sam3.tracker import unpack_masks


# SCAIL-2 was trained on this exact identity-color order.
SCAIL_PALETTE = (
    (0.0, 0.0, 1.0),  # blue
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (1.0, 0.0, 1.0),  # magenta
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 1.0, 0.0),  # yellow
)


def validate_image_batch(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be an IMAGE batch shaped [frames, H, W, C].")
    if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] < 1:
        raise ValueError(f"{name} must contain at least one non-empty frame.")
    if value.shape[-1] < 3:
        raise ValueError(f"{name} must contain at least three color channels.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return value


def resize_image_batch(
    images: torch.Tensor,
    width: int,
    height: int,
    method: str,
    fit: str,
) -> torch.Tensor:
    """Resize a complete IMAGE batch without collapsing its temporal dimension."""

    validate_image_batch(images, "images")
    crop = "center" if fit == "center crop" else "disabled"
    if fit not in ("center crop", "stretch"):
        raise ValueError(f"Unknown SCAIL image fit mode: {fit}")
    resized = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        int(width),
        int(height),
        method,
        crop,
    )
    return resized.movedim(1, -1)


def extract_mask_to_28ch(rgb_video: torch.Tensor) -> torch.Tensor:
    """Convert SCAIL's exact RGB palette into its 28-channel temporal mask."""

    validate_image_batch(rgb_video, "SCAIL colored mask")
    frames, height, width, _ = rgb_video.shape
    on_threshold = 225.0 / 255.0
    mask = rgb_video[..., :3].movedim(-1, 1).float()
    red = (mask[:, 0:1] > on_threshold).float()
    green = (mask[:, 1:2] > on_threshold).float()
    blue = (mask[:, 2:3] > on_threshold).float()
    not_red, not_green, not_blue = 1 - red, 1 - green, 1 - blue
    binary = torch.cat(
        [
            red * green * blue,
            red * not_green * not_blue,
            not_red * green * not_blue,
            not_red * not_green * blue,
            red * green * not_blue,
            red * not_green * blue,
            not_red * green * blue,
        ],
        dim=1,
    )

    latent_height, latent_width = height, width
    for _ in range(3):
        latent_height = (latent_height + 1) // 2
        latent_width = (latent_width + 1) // 2
    binary = F.interpolate(
        binary,
        size=(latent_height, latent_width),
        mode="area",
    )
    temporal = (frames - 1) // 4 + 1
    padded = torch.cat([binary[:1].repeat(4, 1, 1, 1), binary[1:]], dim=0)
    if padded.shape[0] != temporal * 4:
        raise ValueError(
            f"SCAIL mask has invalid temporal length {frames}; expected 4n+1."
        )
    return padded.view(temporal, 28, latent_height, latent_width).unsqueeze(0)


def _empty_latent(batch_size: int, length: int, height: int, width: int) -> torch.Tensor:
    return torch.zeros(
        [
            int(batch_size),
            16,
            ((int(length) - 1) // 4) + 1,
            int(height) // 8,
            int(width) // 8,
        ],
        device=comfy.model_management.intermediate_device(),
    )


def prepare_scail_window(
    *,
    positive,
    negative,
    vae,
    width: int,
    height: int,
    length: int,
    pose_strength: float,
    pose_start: float,
    pose_end: float,
    video_frame_offset: int,
    previous_frame_count: int,
    replacement_mode: bool,
    reference_image=None,
    clip_vision_output=None,
    pose_video=None,
    pose_video_mask=None,
    reference_image_mask=None,
    previous_frames=None,
):
    """Build one native SCAIL conditioning window without invoking another node."""

    latent = _empty_latent(1, length, height, width)
    noise_mask = None

    ref_mask_flag = not bool(replacement_mode)
    positive = node_helpers.conditioning_set_values(
        positive, {"ref_mask_flag": ref_mask_flag}
    )
    negative = node_helpers.conditioning_set_values(
        negative, {"ref_mask_flag": ref_mask_flag}
    )

    previous_trimmed = None
    if previous_frames is not None and previous_frames.shape[0] > 0:
        validate_image_batch(previous_frames, "previous_frames")
        previous_trimmed = previous_frames[-int(previous_frame_count) :]
        video_frame_offset = max(
            0, int(video_frame_offset) - int(previous_trimmed.shape[0])
        )

    if reference_image is not None:
        validate_image_batch(reference_image, "reference_image")
        reference_resized = resize_image_batch(
            reference_image, width, height, "bicubic", "center crop"
        )
        reference_count = reference_resized.shape[0]

        if bool(replacement_mode) and reference_image_mask is not None:
            validate_image_batch(reference_image_mask, "reference_image_mask")
            reference_mask_resized = resize_image_batch(
                reference_image_mask, width, height, "nearest-exact", "center crop"
            )
            indices = [
                min(index, reference_mask_resized.shape[0] - 1)
                for index in range(reference_count)
            ]
            reference_mask_resized = reference_mask_resized[indices]
            is_character = (
                reference_mask_resized[..., :3]
                .amax(dim=-1, keepdim=True)
                .gt(0.1)
                .to(reference_resized.dtype)
            )
            reference_resized = reference_resized * is_character

        reference_latents = [
            vae.encode(reference_resized[index : index + 1, ..., :3])
            for index in range(reference_count)
        ]
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": reference_latents}, append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"reference_latents": reference_latents}, append=True
        )

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(
            positive, {"clip_vision_output": clip_vision_output}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"clip_vision_output": clip_vision_output}
        )

    if pose_video is not None:
        validate_image_batch(pose_video, "pose_video")
        pose_video = (
            None
            if pose_video.shape[0] <= video_frame_offset
            else pose_video[video_frame_offset:]
        )
    if pose_video_mask is not None:
        validate_image_batch(pose_video_mask, "pose_video_mask")
        pose_video_mask = (
            None
            if pose_video_mask.shape[0] <= video_frame_offset
            else pose_video_mask[video_frame_offset:]
        )

    temporal_lengths = [
        value.shape[0]
        for value in (pose_video, pose_video_mask)
        if value is not None
    ]
    if temporal_lengths:
        kept = ((min(min(temporal_lengths), int(length)) - 1) // 4) * 4 + 1
        if kept < 1:
            raise ValueError("SCAIL window has no usable pose/mask frames.")
        if pose_video is not None:
            pose_video = pose_video[:kept]
        if pose_video_mask is not None:
            pose_video_mask = pose_video_mask[:kept]

    if pose_video is not None:
        pose_resized = resize_image_batch(
            pose_video[:length],
            width // 2,
            height // 2,
            "area",
            "center crop",
        )
        pose_latent = vae.encode(pose_resized[..., :3]) * float(pose_strength)
        positive = node_helpers.conditioning_set_values_with_timestep_range(
            positive,
            {"pose_video_latent": pose_latent},
            float(pose_start),
            float(pose_end),
        )
        negative = node_helpers.conditioning_set_values_with_timestep_range(
            negative,
            {"pose_video_latent": pose_latent},
            float(pose_start),
            float(pose_end),
        )

    if pose_video_mask is not None:
        mask_resized = resize_image_batch(
            pose_video_mask[:length],
            width // 2,
            height // 2,
            "area",
            "center crop",
        )
        driving_mask = extract_mask_to_28ch(mask_resized)
        positive = node_helpers.conditioning_set_values(
            positive, {"driving_mask_28ch": driving_mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"driving_mask_28ch": driving_mask}
        )

    if reference_image_mask is not None and reference_image is not None:
        reference_mask_resized = resize_image_batch(
            reference_image_mask,
            width,
            height,
            "nearest-exact",
            "center crop",
        )
        mask_count = reference_mask_resized.shape[0]
        reference_count = reference_image.shape[0]
        additional_masks = [
            extract_mask_to_28ch(
                reference_mask_resized[min(index, mask_count - 1)][None]
            )
            for index in range(1, reference_count)
        ]
        primary_mask = extract_mask_to_28ch(reference_mask_resized[:1])
        zeros = torch.zeros(
            (
                1,
                latent.shape[2],
                28,
                primary_mask.shape[-2],
                primary_mask.shape[-1],
            ),
            device=primary_mask.device,
            dtype=primary_mask.dtype,
        )
        reference_mask_28ch = torch.cat(
            additional_masks + [primary_mask, zeros], dim=1
        )
        positive = node_helpers.conditioning_set_values(
            positive, {"ref_mask_28ch": reference_mask_28ch}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"ref_mask_28ch": reference_mask_28ch}
        )

    if previous_trimmed is not None:
        previous_resized = resize_image_batch(
            previous_trimmed, width, height, "bicubic", "center crop"
        )
        previous_latent = vae.encode(previous_resized[..., :3])
        previous_latent_frames = min(previous_latent.shape[2], latent.shape[2])
        latent[:, :, :previous_latent_frames] = previous_latent[
            :, :, :previous_latent_frames
        ].to(latent.dtype)
        noise_mask = torch.ones(
            (1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]),
            device=latent.device,
            dtype=latent.dtype,
        )
        noise_mask[:, :, :previous_latent_frames] = 0.0

    output_latent = {"samples": latent}
    if noise_mask is not None:
        output_latent["noise_mask"] = noise_mask
    return positive, negative, output_latent, int(video_frame_offset) + int(length)


def _zero_noise_like(samples: Any) -> Any:
    if getattr(samples, "is_nested", False):
        tensors = [
            torch.zeros(
                tensor.shape,
                dtype=tensor.dtype,
                layout=tensor.layout,
                device="cpu",
            )
            for tensor in samples.unbind()
        ]
        return comfy.nested_tensor.NestedTensor(tensors)
    return torch.zeros(
        samples.shape,
        dtype=samples.dtype,
        layout=samples.layout,
        device="cpu",
    )


def sample_scail_window(
    *,
    model,
    positive,
    negative,
    latent: dict,
    sampler,
    sigmas: torch.Tensor,
    seed: int,
    cfg: float,
    add_noise: bool,
) -> dict:
    """Mirror ComfyUI's SamplerCustom and return its denoised latent."""

    working = dict(latent)
    samples = comfy.sample.fix_empty_latent_channels(
        model,
        working["samples"],
        working.get("downscale_ratio_spacial"),
        working.get("downscale_ratio_temporal"),
    )
    working["samples"] = samples
    if add_noise:
        noise = comfy.sample.prepare_noise(
            samples,
            int(seed),
            working.get("batch_index"),
        )
    else:
        noise = _zero_noise_like(samples)

    x0_output: dict[str, Any] = {}
    callback = latent_preview.prepare_callback(
        model, int(sigmas.shape[-1] - 1), x0_output
    )
    sampled = comfy.sample.sample_custom(
        model,
        noise,
        float(cfg),
        sampler,
        sigmas,
        positive,
        negative,
        samples,
        noise_mask=working.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(seed),
    )

    working.pop("downscale_ratio_spacial", None)
    working.pop("downscale_ratio_temporal", None)
    if "x0" in x0_output:
        denoised = model.model.process_latent_out(x0_output["x0"].cpu())
        if getattr(sampled, "is_nested", False):
            latent_shapes = [tensor.shape for tensor in sampled.unbind()]
            denoised = comfy.nested_tensor.NestedTensor(
                comfy.utils.unpack_latents(denoised, latent_shapes)
            )
        working["samples"] = denoised
    else:
        working["samples"] = sampled
    return working


def reinhard_color_transfer(
    image_target: torch.Tensor,
    image_reference: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Per-frame Reinhard LAB transfer used by the proven extension workflow."""

    validate_image_batch(image_target, "color-transfer target")
    validate_image_batch(image_reference, "color-transfer reference")
    strength = float(strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("color_transfer_strength must be between 0 and 1.")
    if strength == 0.0:
        return image_target

    import kornia

    compute_device = comfy.model_management.get_torch_device()
    output_device = comfy.model_management.intermediate_device()
    output_dtype = comfy.model_management.intermediate_dtype()
    count, height, width, channels = image_target.shape
    if channels != 3 or image_reference.shape[-1] != 3:
        raise ValueError("Reinhard color transfer requires three-channel RGB images.")

    reference_lab = kornia.color.rgb_to_lab(
        image_reference[-1:].to(compute_device, dtype=torch.float32).permute(0, 3, 1, 2)
    )
    reference_flat = reference_lab.reshape(3, -1)
    reference_mean = reference_flat.mean(dim=-1, keepdim=True)
    reference_std = reference_flat.std(
        dim=-1, keepdim=True, unbiased=False
    ).clamp_min_(1e-6)

    output = torch.empty(
        count,
        height,
        width,
        channels,
        device=output_device,
        dtype=output_dtype,
    )
    progress = comfy.utils.ProgressBar(count)
    for index in range(count):
        source_lab = kornia.color.rgb_to_lab(
            image_target[index : index + 1]
            .to(compute_device, dtype=torch.float32)
            .permute(0, 3, 1, 2)
        )
        source_flat = source_lab.reshape(3, -1)
        source_mean = source_flat.mean(dim=-1, keepdim=True)
        source_std = source_flat.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min_(1e-6)
        corrected = (source_flat - source_mean) * (
            reference_std / source_std
        ) + reference_mean
        corrected = corrected.view(1, 3, height, width)
        if strength < 1.0:
            corrected = torch.lerp(source_lab, corrected, strength)
        rgb = kornia.color.lab_to_rgb(corrected).clamp_(0, 1)
        output[index] = (
            rgb[0].permute(1, 2, 0).to(device=output_device, dtype=output_dtype)
        )
        progress.update(1)
    return output


def track_object_count(track_data: dict) -> int:
    packed = track_data.get("packed_masks")
    return 0 if packed is None else int(packed.shape[1])


def _unpack_track(track_data: dict) -> torch.Tensor | None:
    packed = track_data.get("packed_masks")
    if packed is None or packed.shape[1] == 0:
        return None
    return unpack_masks(packed)


def _first_appearance_stats(
    masks_bool: torch.Tensor,
) -> tuple[list, list, list]:
    masks = masks_bool.float()
    frames, height, width = (
        masks.shape[0],
        masks.shape[-2],
        masks.shape[-1],
    )
    grid_x = torch.arange(
        width, device=masks.device, dtype=masks.dtype
    ).view(1, 1, 1, width)
    area_by_time = masks.sum(dim=(-1, -2))
    center_by_time = (masks * grid_x).sum(dim=(-1, -2)) / area_by_time.clamp(
        min=1
    )
    present = area_by_time > 0
    frame_indices = torch.arange(frames, device=masks.device).unsqueeze(1)
    first = torch.where(present, frame_indices, frames).amin(dim=0)
    selected = first.clamp(max=frames - 1).unsqueeze(0)
    center = center_by_time.gather(0, selected).squeeze(0)
    area = area_by_time.gather(0, selected).squeeze(0)
    return first.tolist(), (center / width).tolist(), (
        area / (height * width)
    ).tolist()


def _subset_track_data(track_data: dict, indices: list[int]) -> dict:
    output = dict(track_data)
    packed = track_data.get("packed_masks")
    if packed is None or not indices:
        output["packed_masks"] = None
        output["scores"] = []
        return output
    output["packed_masks"] = packed[:, indices].contiguous()
    scores = track_data.get("scores")
    if scores is not None:
        output["scores"] = [
            scores[index] for index in indices if index < len(scores)
        ]
    return output


def select_track_data(
    track_data: dict,
    object_indices: str,
    sort_by: str,
) -> dict:
    """Apply one identical identity order/filter to reference and driving tracks."""

    selected = dict(track_data)
    unpacked = _unpack_track(selected)
    if sort_by != "none" and unpacked is not None:
        first, center, area = _first_appearance_stats(unpacked)
        if sort_by == "left_to_right":
            order = sorted(
                range(len(center)),
                key=lambda index: (first[index], center[index]),
            )
        elif sort_by == "area":
            order = sorted(
                range(len(area)),
                key=lambda index: (first[index], -area[index]),
            )
        else:
            raise ValueError(f"Unknown SCAIL identity sort mode: {sort_by}")
        selected = _subset_track_data(selected, order)

    text = str(object_indices or "").strip()
    if text:
        parsed = []
        for token in text.split(","):
            token = token.strip()
            if not token.isdigit():
                raise ValueError(
                    "object_indices must be comma-separated non-negative integers."
                )
            parsed.append(int(token))
        packed = selected.get("packed_masks")
        available = 0 if packed is None else int(packed.shape[1])
        invalid = [index for index in parsed if index >= available]
        if invalid:
            raise ValueError(
                f"SCAIL object_indices {invalid} exceed the {available} tracked "
                "identity/identities."
            )
        selected = _subset_track_data(selected, parsed)
    return selected


def render_colored_track(
    track_data: dict,
    *,
    background: str,
) -> torch.Tensor:
    packed = track_data.get("packed_masks")
    height, width = track_data["orig_size"]
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    background_rgb = (
        (1.0, 1.0, 1.0) if background == "white" else (0.0, 0.0, 0.0)
    )
    if packed is None or packed.shape[1] == 0:
        frame_count = (
            int(track_data.get("n_frames", 1))
            if packed is None
            else int(packed.shape[0])
        )
        result = torch.empty(
            frame_count, height, width, 3, device=device, dtype=dtype
        )
        result[..., 0] = background_rgb[0]
        result[..., 1] = background_rgb[1]
        result[..., 2] = background_rgb[2]
        return result

    frame_count, identity_count = packed.shape[0], packed.shape[1]
    colors = torch.tensor(
        [
            SCAIL_PALETTE[index % len(SCAIL_PALETTE)]
            for index in range(identity_count)
        ],
        device=device,
        dtype=dtype,
    )
    masks = unpack_masks(packed.to(device)).float()
    mask_height, mask_width = masks.shape[-2], masks.shape[-1]
    masks = F.interpolate(
        masks.view(frame_count * identity_count, 1, mask_height, mask_width),
        size=(height, width),
        mode="nearest",
    ).view(frame_count, identity_count, height, width) > 0.5
    any_mask = masks.any(dim=1)
    overlay = colors[masks.to(torch.uint8).argmax(dim=1)]
    background_tensor = torch.tensor(
        background_rgb, device=device, dtype=overlay.dtype
    ).view(1, 1, 1, 3)
    return torch.where(
        any_mask.unsqueeze(-1),
        overlay,
        background_tensor.expand_as(overlay),
    )


def render_mask_identity(mask: torch.Tensor, *, background: str) -> torch.Tensor:
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("Reference MASK must be shaped [batch, H, W].")
    mask = mask.to(device=device, dtype=dtype)
    batch, height, width = mask.shape
    background_rgb = (
        (1.0, 1.0, 1.0) if background == "white" else (0.0, 0.0, 0.0)
    )
    color = torch.tensor(
        SCAIL_PALETTE[0], device=device, dtype=dtype
    ).view(1, 1, 1, 3)
    background_tensor = torch.tensor(
        background_rgb, device=device, dtype=dtype
    ).view(1, 1, 1, 3)
    return torch.where(
        (mask > 0.5).unsqueeze(-1),
        color.expand(batch, height, width, 3),
        background_tensor.expand(batch, height, width, 3),
    )

