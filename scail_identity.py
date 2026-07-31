"""Consolidated SAM3 identity tracking and SCAIL-2 colored-mask rendering."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io

try:
    from .scail_core import (
        render_colored_track,
        select_track_data,
        track_object_count,
        validate_image_batch,
    )
except ImportError:  # Weight-free tests import modules directly from the pack root.
    from scail_core import (
        render_colored_track,
        select_track_data,
        track_object_count,
        validate_image_batch,
    )


SCAIL_TRACK_DATA = io.Custom("SAM3_TRACK_DATA")


def _empty_track(frame_count: int, height: int, width: int) -> dict:
    return {
        "packed_masks": None,
        "orig_size": (int(height), int(width)),
        "n_frames": int(frame_count),
        "scores": [],
    }


def _normalize_points(marker: dict, width: int, height: int) -> list[list[float]]:
    points = marker.get("points")
    if not points:
        points = [[marker.get("x", 0), marker.get("y", 0), 1]]
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = min(float(width - 1), max(0.0, float(point[0])))
        y = min(float(height - 1), max(0.0, float(point[1])))
        label = 1 if len(point) < 3 or int(point[2]) != 0 else 0
        normalized.append([x, y, label])
    return normalized


def _normalize_markers(value: Any, width: int, height: int) -> list[dict]:
    if not isinstance(value, list):
        return []
    normalized = []
    for raw in value[:6]:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") == "box":
            x = min(float(width - 1), max(0.0, float(raw.get("x", 0))))
            y = min(float(height - 1), max(0.0, float(raw.get("y", 0))))
            box_width = max(1.0, float(raw.get("w", 1)))
            box_height = max(1.0, float(raw.get("h", 1)))
            box_width = min(box_width, float(width) - x)
            box_height = min(box_height, float(height) - y)
            normalized.append(
                {
                    "type": "box",
                    "x": x,
                    "y": y,
                    "w": box_width,
                    "h": box_height,
                }
            )
        else:
            points = _normalize_points(raw, width, height)
            if points:
                normalized.append({"type": "point", "points": points})
    return normalized


def _segment_markers(
    sam3,
    image: torch.Tensor,
    markers: list[dict],
    refine_iterations: int,
    device,
    dtype,
) -> torch.Tensor | None:
    if not markers:
        return None
    _, height, width, _ = image.shape
    frame = comfy.utils.common_upscale(
        image[:1, ..., :3].movedim(-1, 1),
        1008,
        1008,
        "bilinear",
        crop="disabled",
    ).to(device=device, dtype=dtype)

    def refine(mask_logit):
        for _ in range(max(0, int(refine_iterations) - 1)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            mask_logit = sam3.forward_segment(frame, mask_inputs=mask_logit)
        mask = F.interpolate(
            mask_logit,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return (mask[0] > 0).float()

    masks = []
    for marker in markers:
        comfy.model_management.throw_exception_if_processing_interrupted()
        if marker["type"] == "box":
            x1 = marker["x"] / width * 1008
            y1 = marker["y"] / height * 1008
            x2 = (marker["x"] + marker["w"]) / width * 1008
            y2 = (marker["y"] + marker["h"]) / height * 1008
            box = torch.tensor(
                [[[x1, y1], [x2, y2]]],
                device=device,
                dtype=dtype,
            )
            mask_logit = sam3.forward_segment(frame, box_inputs=box)
        else:
            points = marker["points"]
            coordinates = torch.tensor(
                [
                    [
                        [point[0] / width * 1008, point[1] / height * 1008]
                        for point in points
                    ]
                ],
                device=device,
                dtype=dtype,
            )
            labels = torch.tensor(
                [[int(point[2]) for point in points]],
                device=device,
                dtype=torch.int32,
            )
            mask_logit = sam3.forward_segment(
                frame,
                point_inputs={
                    "point_coords": coordinates,
                    "point_labels": labels,
                },
            )
        masks.append(refine(mask_logit))
    return torch.cat(masks, dim=0)


def _conditioning_prompts(conditioning, device, dtype):
    if conditioning is None or len(conditioning) == 0:
        return None
    from comfy_extras.nodes_sam3 import _extract_text_prompts

    return [
        (embedding, mask)
        for embedding, mask, _ in _extract_text_prompts(
            conditioning, device, dtype
        )
    ]


def _track_side(
    sam3,
    images: torch.Tensor,
    seed_masks: torch.Tensor | None,
    text_prompts,
    detection_threshold: float,
    max_identities: int,
    detect_interval: int,
    device,
    dtype,
) -> dict:
    frame_count, height, width, _ = images.shape
    if seed_masks is None and text_prompts is None:
        return _empty_track(frame_count, height, width)
    frames = images[..., :3].movedim(-1, 1)
    initial_masks = (
        None
        if seed_masks is None
        else seed_masks.unsqueeze(1).to(device=device, dtype=dtype)
    )
    progress = comfy.utils.ProgressBar(frame_count)
    result = sam3.forward_video(
        images=frames,
        initial_masks=initial_masks,
        pbar=progress,
        text_prompts=text_prompts,
        new_det_thresh=float(detection_threshold),
        max_objects=int(max_identities),
        detect_interval=int(detect_interval),
        target_device=device,
        target_dtype=dtype,
    )
    result["orig_size"] = (height, width)
    result["n_frames"] = int(frame_count)
    return result


def _save_canvas_preview(image: torch.Tensor, prefix: str) -> dict:
    array = (
        image[..., :3].detach().clamp(0, 1).cpu().numpy() * 255.0
    ).astype(np.uint8)
    filename = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
    output_dir = folder_paths.get_temp_directory()
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(array, "RGB").save(
        os.path.join(output_dir, filename),
        compress_level=3,
    )
    return {"filename": filename, "subfolder": "", "type": "temp"}


class WanSCAILIdentityControl(io.ComfyNode):
    """Track ordered identities and render both SCAIL masks in one node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILIdentityControl",
            display_name="Wan SCAIL-2 Identity Control",
            search_aliases=[
                "scail identity",
                "scail mask",
                "sam3 scail",
                "character replacement",
            ],
            category="model/conditioning/wan/scail",
            description=(
                "Canvas-assisted SAM3 tracking and SCAIL-2 palette-mask rendering "
                "for both reference and driving video. Replaces separate identity "
                "tracker and colored-mask nodes."
            ),
            inputs=[
                io.Model.Input(
                    "sam3_model",
                    tooltip="SAM3 checkpoint MODEL used for segmentation and tracking.",
                ),
                io.Image.Input(
                    "reference_image",
                    tooltip="Prepared reference image or multi-view reference batch.",
                ),
                io.Image.Input(
                    "pose_video",
                    tooltip="Full prepared driving-video frame batch.",
                ),
                io.Conditioning.Input(
                    "reference_conditioning",
                    optional=True,
                    tooltip="Optional SAM3 text conditioning, usually 'person'.",
                ),
                io.Conditioning.Input(
                    "driving_conditioning",
                    optional=True,
                    tooltip="Optional SAM3 text conditioning, usually 'person'.",
                ),
                io.Int.Input(
                    "refine_iterations",
                    default=2,
                    min=1,
                    max=5,
                    tooltip="SAM decoder refinement passes for drawn seeds.",
                ),
                io.Boolean.Input(
                    "auto_detect",
                    default=True,
                    tooltip="Allow connected text conditioning to detect identities.",
                ),
                io.Float.Input(
                    "detection_threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input(
                    "max_identities",
                    default=6,
                    min=1,
                    max=6,
                    tooltip="SCAIL-2's trained palette supports up to six identities.",
                ),
                io.Int.Input(
                    "detect_interval",
                    default=1,
                    min=1,
                    max=64,
                    tooltip="Run text detection every N driving frames.",
                ),
                io.String.Input(
                    "object_indices",
                    default="0",
                    tooltip=(
                        "Comma-separated tracked indices to retain. '0' is safest "
                        "for single-character replacement; empty keeps all."
                    ),
                ),
                io.Combo.Input(
                    "sort_by",
                    options=["none", "left_to_right", "area"],
                    default="none",
                    tooltip="Apply the same identity order to both masks.",
                ),
                io.Boolean.Input(
                    "replacement_mode",
                    default=True,
                    tooltip=(
                        "True keeps the driving background; false animates the "
                        "reference scene."
                    ),
                ),
                io.String.Input(
                    "markers",
                    default='{"reference":[],"driving":[]}',
                    multiline=True,
                    advanced=True,
                    tooltip="Managed by the identity canvas.",
                ),
            ],
            outputs=[
                io.Image.Output("pose_video"),
                io.Image.Output("reference_image"),
                io.Image.Output("pose_video_mask"),
                io.Image.Output("reference_image_mask"),
                io.Int.Output("identity_count"),
                io.String.Output("diagnostics"),
            ],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        sam3_model,
        reference_image,
        pose_video,
        refine_iterations,
        auto_detect,
        detection_threshold,
        max_identities,
        detect_interval,
        object_indices,
        sort_by,
        replacement_mode,
        markers,
        reference_conditioning=None,
        driving_conditioning=None,
    ):
        validate_image_batch(reference_image, "reference_image")
        validate_image_batch(pose_video, "pose_video")
        if (
            reference_image.shape[1:3] != pose_video.shape[1:3]
        ):
            raise ValueError(
                "SCAIL reference_image and pose_video must have matching prepared "
                "height/width. Connect both outputs from Wan SCAIL-2 Media Prep."
            )
        if not 0.0 <= float(detection_threshold) <= 1.0:
            raise ValueError("detection_threshold must be between 0 and 1.")

        try:
            marker_data = (
                json.loads(markers)
                if isinstance(markers, str) and markers.strip()
                else {}
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            marker_data = {}
        if not isinstance(marker_data, dict):
            marker_data = {}

        reference_markers = _normalize_markers(
            marker_data.get("reference", []),
            reference_image.shape[2],
            reference_image.shape[1],
        )
        driving_markers = _normalize_markers(
            marker_data.get("driving", []),
            pose_video.shape[2],
            pose_video.shape[1],
        )
        previews = {
            "reference_preview": [
                _save_canvas_preview(reference_image[0], "wan_scail_reference")
            ],
            "driving_preview": [
                _save_canvas_preview(pose_video[0], "wan_scail_driving")
            ],
        }

        has_text = bool(auto_detect) and (
            (
                reference_conditioning is not None
                and len(reference_conditioning) > 0
            )
            or (
                driving_conditioning is not None
                and len(driving_conditioning) > 0
            )
        )
        preview_only = (
            not reference_markers and not driving_markers and not has_text
        )
        if preview_only:
            reference_track = _empty_track(
                reference_image.shape[0],
                reference_image.shape[1],
                reference_image.shape[2],
            )
            driving_track = _empty_track(
                pose_video.shape[0],
                pose_video.shape[1],
                pose_video.shape[2],
            )
        else:
            comfy.model_management.load_model_gpu(sam3_model)
            device = comfy.model_management.get_torch_device()
            dtype = sam3_model.model.get_dtype()
            sam3 = sam3_model.model.diffusion_model

            reference_seed = _segment_markers(
                sam3,
                reference_image,
                reference_markers,
                refine_iterations,
                device,
                dtype,
            )
            driving_seed = _segment_markers(
                sam3,
                pose_video,
                driving_markers,
                refine_iterations,
                device,
                dtype,
            )
            reference_text = (
                _conditioning_prompts(reference_conditioning, device, dtype)
                if auto_detect
                else None
            )
            driving_text = (
                _conditioning_prompts(driving_conditioning, device, dtype)
                if auto_detect
                else None
            )
            reference_track = _track_side(
                sam3,
                reference_image,
                reference_seed,
                reference_text,
                detection_threshold,
                max_identities,
                detect_interval,
                device,
                dtype,
            )
            driving_track = _track_side(
                sam3,
                pose_video,
                driving_seed,
                driving_text,
                detection_threshold,
                max_identities,
                detect_interval,
                device,
                dtype,
            )

        reference_total = track_object_count(reference_track)
        driving_total = track_object_count(driving_track)
        if preview_only:
            reference_selected = reference_track
            driving_selected = driving_track
        else:
            reference_selected = select_track_data(
                reference_track, object_indices, sort_by
            )
            driving_selected = select_track_data(
                driving_track, object_indices, sort_by
            )
        reference_count = track_object_count(reference_selected)
        driving_count = track_object_count(driving_selected)

        if not preview_only and driving_count < 1:
            raise ValueError(
                "SCAIL identity control selected no driving identity. Inspect the "
                "canvas/text prompt and object_indices before generation."
            )
        if not preview_only and reference_count < 1:
            raise ValueError(
                "SCAIL identity control selected no reference identity. Inspect the "
                "canvas/text prompt and object_indices before generation."
            )
        if not preview_only and reference_count != driving_count:
            raise ValueError(
                f"SCAIL selected {reference_count} reference identity/identities "
                f"but {driving_count} driving identity/identities. Use matching "
                "markers and object_indices."
            )

        pose_background = "white" if replacement_mode else "black"
        reference_background = "black" if replacement_mode else "white"
        pose_mask = render_colored_track(
            driving_selected, background=pose_background
        )
        reference_mask = render_colored_track(
            reference_selected, background=reference_background
        )
        diagnostics = (
            "SCAIL canvas preview only; add markers or connect SAM3 text conditioning"
            if preview_only
            else (
                f"SCAIL identities {reference_count} selected | "
                f"tracked ref {reference_total} / driving {driving_total} | "
                f"indices {object_indices or 'all'} | sort {sort_by} | "
                f"replacement {bool(replacement_mode)} | "
                f"auto-detect {bool(auto_detect)} threshold "
                f"{float(detection_threshold):g} interval {int(detect_interval)}"
            )
        )
        return io.NodeOutput(
            pose_video,
            reference_image,
            pose_mask,
            reference_mask,
            driving_count,
            diagnostics,
            ui=previews,
        )

    track = execute


__all__ = ["WanSCAILIdentityControl"]
