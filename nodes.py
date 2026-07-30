"""Native two-expert sampling for Wan 2.2 high/low-noise models.

The node intentionally accepts models already patched by ComfyUI's
ModelSamplingSD3 node.  V3 nodes execute as classmethods, so keeping patched
MODEL clones in node-global state would retain heavyweight model patchers and
work against ComfyUI's own model lifecycle.  Upstream patching is both safer
and easier to inspect in a workflow.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import logging
import math
import threading
import time
from typing import Any
import uuid

import torch
from PIL import Image

import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

try:
    from server import PromptServer
except ImportError:  # Allows the weight-free unit suite to run outside ComfyUI.
    PromptServer = None


PROFILE_LIGHTX2V = "LightX2V 4-step distilled"
PROFILE_COMFYUI = "ComfyUI 20-step"
PROFILE_CUSTOM = "Custom"

TASK_I2V = "I2V"
TASK_T2V = "T2V"

TASK_BOUNDARIES = {
    TASK_I2V: 0.900,
    TASK_T2V: 0.875,
}

LIVE_PREVIEW_TYPE = "WAN22_LIVE_PREVIEW"


@dataclass(frozen=True)
class SamplingProfile:
    steps: int
    cfg_high: float
    cfg_low: float
    shift: float
    sampler_name: str
    scheduler: str


@dataclass(frozen=True)
class LivePreviewConfig:
    """Small, cacheable settings object; it never retains a MODEL or latent."""

    node_id: str
    output_fps: float
    max_frames: int
    preview_every: int
    webp_quality: int
    max_size: int


class _LatestPreviewEncoder:
    """One-slot worker: a newer preview replaces stale queued work."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending = None
        self._active = False
        self._closing = False
        self._cancelled = False
        self.submitted = 0
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run,
            name="wan22_live_preview",
            daemon=True,
        )
        self._thread.start()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def submit(self, job) -> bool:
        with self._condition:
            if self._closing:
                return False
            replaced = self._pending is not None
            if replaced:
                self.dropped += 1
            self._pending = job
            self.submitted += 1
            self._condition.notify()
            return replaced

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._active or self._pending is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self, *, cancelled: bool, timeout: float = 0.5) -> None:
        with self._condition:
            self._cancelled = bool(cancelled)
            if self._cancelled:
                self._pending = None
            self._closing = True
            self._condition.notify()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._cancelled:
                    return
                if self._pending is None and self._closing:
                    return
                job = self._pending
                self._pending = None
                self._active = True
            try:
                job()
            except Exception:
                logging.exception("[Wan22LivePreview] background preview encoding failed")
            finally:
                with self._condition:
                    self._active = False
                    self._condition.notify_all()


_SHARED_PREVIEW_ENCODER = None
_SHARED_PREVIEW_ENCODER_LOCK = threading.Lock()


def _shared_preview_encoder() -> _LatestPreviewEncoder:
    global _SHARED_PREVIEW_ENCODER
    with _SHARED_PREVIEW_ENCODER_LOCK:
        if _SHARED_PREVIEW_ENCODER is None:
            _SHARED_PREVIEW_ENCODER = _LatestPreviewEncoder()
        return _SHARED_PREVIEW_ENCODER


def _stage_preview_rgb_frames(
    x0: torch.Tensor,
    latent_format: Any,
    max_frames: int,
    max_size: int,
) -> tuple[torch.Tensor, int, Any | None]:
    """Stage selected Latent2RGB frames on CPU without blocking a CUDA callback."""
    if not isinstance(x0, torch.Tensor) or x0.ndim != 5:
        raise ValueError("live preview requires a five-dimensional video latent")

    rgb_factors = getattr(latent_format, "latent_rgb_factors", None)
    if rgb_factors is None:
        raise ValueError("the active latent format has no Latent2RGB factors")

    x0_view = x0.detach()
    reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
    if reshape is not None:
        x0_view = reshape(x0_view)
    if x0_view.ndim != 5:
        raise ValueError("the latent format reshape did not preserve video dimensions")

    source_frames = int(x0_view.shape[2])
    if source_frames < 1:
        raise ValueError("the video latent contains no temporal frames")
    frame_count = min(source_frames, max(1, int(max_frames)))
    indices = torch.linspace(
        0,
        source_frames - 1,
        frame_count,
        device=x0_view.device,
    ).round().to(dtype=torch.long)

    selected = x0_view[0].index_select(1, indices)
    factors = torch.as_tensor(
        rgb_factors,
        dtype=x0_view.dtype,
        device=x0_view.device,
    ).transpose(0, 1)
    bias_values = getattr(latent_format, "latent_rgb_factors_bias", None)
    bias = None
    if bias_values is not None:
        bias = torch.as_tensor(
            bias_values,
            dtype=x0_view.dtype,
            device=x0_view.device,
        )

    rgb = torch.nn.functional.linear(selected.movedim(0, -1), factors, bias=bias)
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    size_limit = max(0, int(max_size))
    if size_limit and max(height, width) > size_limit:
        scale = size_limit / max(height, width)
        target = (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )
        rgb = torch.nn.functional.interpolate(
            rgb.movedim(-1, 1),
            size=target,
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

    rgb = ((rgb + 1.0) * 127.5).clamp_(0, 255).to(dtype=torch.uint8)
    rgb = rgb.contiguous()
    if rgb.device.type == "cuda":
        staged = torch.empty(
            rgb.shape,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        staged.copy_(rgb, non_blocking=True)
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(device=rgb.device))
        return staged, source_frames, ready_event

    return rgb.to(device="cpu").contiguous(), source_frames, None


def _preview_rgb_frames(
    x0: torch.Tensor,
    latent_format: Any,
    max_frames: int,
) -> tuple[torch.Tensor, int]:
    """Synchronous wrapper used by diagnostics and the weight-free test suite."""
    frames, source_frames, ready_event = _stage_preview_rgb_frames(
        x0,
        latent_format,
        max_frames,
        0,
    )
    if ready_event is not None:
        ready_event.synchronize()
    return frames, source_frames


def _encode_preview_webp(
    rgb_frames: torch.Tensor,
    *,
    output_fps: float,
    temporal_ratio: int,
    source_frames: int,
    quality: int,
    max_size: int,
) -> tuple[str, int, int, int, float]:
    """Encode selected CPU frames while preserving the decoded video's duration."""
    if rgb_frames.device.type != "cpu" or rgb_frames.dtype != torch.uint8:
        raise ValueError("preview encoder accepts only CPU uint8 RGB frames")
    if rgb_frames.ndim != 4 or rgb_frames.shape[-1] != 3 or rgb_frames.shape[0] < 1:
        raise ValueError("preview encoder received an invalid RGB frame tensor")

    frames = [Image.fromarray(frame.numpy(), mode="RGB") for frame in rgb_frames]
    if max_size > 0:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        frames = [
            image.copy()
            if max(image.size) <= max_size
            else _downscale_preview(image, max_size, resampling)
            for image in frames
        ]

    ratio = max(1, int(temporal_ratio))
    decoded_frames = max(1, (max(1, int(source_frames)) - 1) * ratio + 1)
    duration_ms = max(
        1,
        int(round(1000.0 * decoded_frames / (float(output_fps) * len(frames)))),
    )
    effective_fps = len(frames) * float(output_fps) / decoded_frames

    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=int(quality),
        method=0,
    )
    return (
        base64.b64encode(buffer.getvalue()).decode("ascii"),
        frames[0].width,
        frames[0].height,
        decoded_frames,
        effective_fps,
    )


def _downscale_preview(image: Image.Image, max_size: int, resampling) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((max_size, max_size), resampling)
    return copy


class _LivePreviewSession:
    """Own one high-to-low preview timeline without touching either MODEL patcher."""

    def __init__(
        self,
        model: Any,
        config: LivePreviewConfig,
        total_steps: int,
        sigmas: torch.Tensor,
    ) -> None:
        server = getattr(PromptServer, "instance", None) if PromptServer is not None else None
        if server is None:
            raise RuntimeError("ComfyUI's prompt server is unavailable")
        client_id = getattr(server, "client_id", None)
        if client_id is None:
            raise RuntimeError("live preview requires an active ComfyUI browser client")
        if not config.node_id:
            raise RuntimeError("the live-preview node has no execution id")

        latent_format = model.model.latent_format
        if getattr(latent_format, "latent_rgb_factors", None) is None:
            raise RuntimeError("the active Wan latent format has no Latent2RGB factors")

        self.config = config
        self.total_steps = int(total_steps)
        self.sigmas = sigmas.detach().cpu()
        self.latent_format = latent_format
        self.temporal_ratio = max(
            1,
            int(getattr(latent_format, "temporal_downscale_ratio", 4)),
        )
        self.run_id = uuid.uuid4().hex
        self._server = server
        self._client_id = client_id
        self._cancelled = threading.Event()
        self._disabled_reason = None
        self._last_frame_count = 0
        self._last_effective_fps = 0.0
        self._submitted = 0
        self._dropped = 0
        self._send(
            {
                "node_id": config.node_id,
                "run_id": self.run_id,
                "stage": "starting",
                "global_step": 0,
                "global_total": self.total_steps,
                "output_fps": config.output_fps,
            }
        )
        self._encoder = _shared_preview_encoder()

    def _send(self, payload: dict[str, Any]) -> None:
        self._server.send_sync(
            "wan22_live_preview",
            payload,
            self._client_id,
        )

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is not None:
            return
        self._disabled_reason = reason
        logging.warning("[Wan22LivePreview] disabled for this run: %s", reason)
        try:
            self._send(
                {
                    "node_id": self.config.node_id,
                    "run_id": self.run_id,
                    "stage": "preview unavailable",
                    "global_total": self.total_steps,
                    "output_fps": self.config.output_fps,
                    "error": reason,
                }
            )
        except Exception:
            logging.debug("[Wan22LivePreview] could not send failure status", exc_info=True)

    def update(self, global_step: int, stage: str, x0: torch.Tensor) -> None:
        if self._disabled_reason is not None:
            return
        sent_step = int(global_step) + 1
        if sent_step % self.config.preview_every != 0 and sent_step < self.total_steps:
            return

        try:
            rgb_frames, source_frames, ready_event = _stage_preview_rgb_frames(
                x0,
                self.latent_format,
                self.config.max_frames,
                self.config.max_size,
            )
        except Exception as exc:
            self._disable(str(exc))
            return

        sigma = None
        if 0 <= global_step < self.sigmas.numel():
            sigma = float(self.sigmas[global_step])
        config = self.config
        temporal_ratio = self.temporal_ratio
        cancelled = self._cancelled

        def encode_and_send() -> None:
            if cancelled.is_set():
                return
            try:
                if ready_event is not None:
                    ready_event.synchronize()
                if cancelled.is_set():
                    return
                webp, width, height, decoded_frames, effective_fps = _encode_preview_webp(
                    rgb_frames,
                    output_fps=config.output_fps,
                    temporal_ratio=temporal_ratio,
                    source_frames=source_frames,
                    quality=config.webp_quality,
                    max_size=0,
                )
                if cancelled.is_set():
                    return
                self._last_frame_count = int(rgb_frames.shape[0])
                self._last_effective_fps = effective_fps
                self._send(
                    {
                        "node_id": config.node_id,
                        "run_id": self.run_id,
                        "stage": stage,
                        "global_step": sent_step,
                        "global_total": self.total_steps,
                        "sigma": sigma,
                        "frame_count": int(rgb_frames.shape[0]),
                        "decoded_frame_count": decoded_frames,
                        "output_fps": config.output_fps,
                        "effective_latent_fps": effective_fps,
                        "width": width,
                        "height": height,
                        "mime": "image/webp",
                        "webp_base64": webp,
                    }
                )
            except Exception as exc:
                self._disable(f"WebP encoding failed: {exc}")

        self._submitted += 1
        if self._encoder.submit(encode_and_send):
            self._dropped += 1

    def close(self, *, cancelled: bool) -> None:
        if cancelled:
            self._cancelled.set()

    def diagnostic_note(self) -> str:
        if self._disabled_reason is not None:
            return f"live preview unavailable ({self._disabled_reason})"
        return (
            f"live preview {self._last_frame_count or self.config.max_frames} latent frames "
            f"@ {self.config.output_fps:g} output fps / "
            f"{self._last_effective_fps:.2f} effective fps / "
            f"{self._dropped} stale dropped"
        )


def resolve_profile(
    profile: str,
    *,
    custom_steps: int = 4,
    custom_cfg_high: float = 1.0,
    custom_cfg_low: float = 1.0,
    custom_expected_shift: float = 5.0,
    custom_sampler: str = "euler",
    custom_scheduler: str = "simple",
) -> SamplingProfile:
    """Resolve a UI profile into the exact settings used for both experts."""
    if profile == PROFILE_LIGHTX2V:
        return SamplingProfile(4, 1.0, 1.0, 5.0, "euler", "simple")
    if profile == PROFILE_COMFYUI:
        return SamplingProfile(20, 3.5, 3.5, 8.0, "euler", "simple")
    if profile == PROFILE_CUSTOM:
        if custom_steps < 2:
            raise ValueError("Custom sampling needs at least 2 steps so both Wan experts can run.")
        if custom_cfg_high < 0.0 or custom_cfg_low < 0.0:
            raise ValueError("Custom CFG values must be zero or greater.")
        if not math.isfinite(custom_expected_shift) or custom_expected_shift <= 0.0:
            raise ValueError("Custom expected shift must be a finite value greater than zero.")
        if custom_sampler not in comfy.samplers.SAMPLER_NAMES:
            raise ValueError(f"Unknown sampler: {custom_sampler}")
        if custom_scheduler not in comfy.samplers.SCHEDULER_NAMES:
            raise ValueError(f"Unknown scheduler: {custom_scheduler}")
        return SamplingProfile(
            int(custom_steps),
            float(custom_cfg_high),
            float(custom_cfg_low),
            float(custom_expected_shift),
            custom_sampler,
            custom_scheduler,
        )
    raise ValueError(f"Unknown Wan 2.2 profile: {profile}")


def validate_sigma_schedule(sigmas: torch.Tensor, label: str) -> None:
    """Reject malformed schedules before either heavyweight expert is loaded."""
    if not isinstance(sigmas, torch.Tensor):
        raise ValueError(f"{label} sigma schedule is not a tensor.")
    if sigmas.ndim != 1 or sigmas.numel() < 3:
        raise ValueError(
            f"{label} sigma schedule must be one-dimensional with at least 3 values; "
            f"received shape {tuple(sigmas.shape)}."
        )
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError(f"{label} sigma schedule contains a non-finite value.")
    if not math.isclose(float(sigmas[-1]), 0.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(f"{label} sigma schedule must end at zero.")
    if not bool(torch.all(sigmas[:-1] > sigmas[1:])):
        raise ValueError(f"{label} sigma schedule must be strictly descending.")


def split_sigma_schedule(
    sigmas: torch.Tensor, boundary: float
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Split at the first below-boundary start; equality belongs to high noise.

    ``k`` is the number of high-noise transitions.  Both slices retain the
    shared hand-off sigma, so no denoising interval is skipped or repeated.
    """
    transition_count = sigmas.numel() - 1
    k = 0
    for sigma in sigmas[:-1]:
        value = float(sigma)
        if value > boundary or math.isclose(value, boundary, rel_tol=0.0, abs_tol=1e-7):
            k += 1
        else:
            break

    if k < 1 or k >= transition_count:
        raise ValueError(
            f"Sigma boundary {boundary:.3f} must leave at least one transition "
            "for each Wan expert."
        )
    return k, sigmas[: k + 1], sigmas[k:]


def _model_sampling(model: Any, label: str) -> Any:
    try:
        sampling = model.get_model_object("model_sampling")
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"{label} MODEL does not expose a model_sampling object.") from exc
    if sampling is None:
        raise ValueError(f"{label} MODEL does not expose a model_sampling object.")
    return sampling


def _validate_model_pair(
    high_model: Any,
    low_model: Any,
    high_sampling: Any,
    low_sampling: Any,
    expected_shift: float,
) -> None:
    def reports_clone(model, other) -> bool:
        is_clone = getattr(model, "is_clone", None)
        return bool(is_clone(other)) if callable(is_clone) else False

    if (
        high_model is low_model
        or reports_clone(high_model, low_model)
        or reports_clone(low_model, high_model)
    ):
        raise ValueError(
            "Connect distinct Wan 2.2 high-noise and low-noise expert base models; "
            "two clones of one expert are not valid."
        )
    if high_sampling is low_sampling:
        raise ValueError(
            "The two experts share one model_sampling object; patch each MODEL separately "
            "with ModelSamplingSD3."
        )

    shifts: list[float] = []
    for label, sampling in (("High-noise", high_sampling), ("Low-noise", low_sampling)):
        shift = getattr(sampling, "shift", None)
        try:
            shift = float(shift)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} MODEL has no readable shift. Patch it upstream with ModelSamplingSD3."
            ) from exc
        if not math.isfinite(shift) or not math.isclose(
            shift, expected_shift, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"{label} MODEL shift is {shift:g}; this profile requires "
                f"ModelSamplingSD3 shift {expected_shift:g}."
            )
        shifts.append(shift)

    if not math.isclose(shifts[0], shifts[1], rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("High-noise and low-noise MODEL shifts must match.")


def _calculate_schedule(
    model_sampling: Any, scheduler: str, steps: int, sampler_name: str
) -> torch.Tensor:
    """Match the schedule shape used by ComfyUI's core KSampler."""
    discard = getattr(
        getattr(comfy.samplers, "KSampler", object),
        "DISCARD_PENULTIMATE_SIGMA_SAMPLERS",
        set(),
    )
    if sampler_name in discard:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps + 1)
        return torch.cat((sigmas[:-2], sigmas[-1:])).cpu()
    return comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()


def _zero_noise_like(noise: Any) -> Any:
    if getattr(noise, "is_nested", False):
        tensors = [
            torch.zeros(t.shape, dtype=t.dtype, layout=t.layout, device="cpu")
            for t in noise.unbind()
        ]
        return comfy.nested_tensor.NestedTensor(tensors)
    return torch.zeros(noise.shape, dtype=noise.dtype, layout=noise.layout, device="cpu")


def _unified_callbacks(
    model: Any,
    total_steps: int,
    high_steps: int,
    sigmas: torch.Tensor,
    live_preview: LivePreviewConfig | None,
):
    preview_session = None
    if live_preview is not None:
        try:
            preview_session = _LivePreviewSession(
                model,
                live_preview,
                total_steps,
                sigmas,
            )
        except Exception as exc:
            logging.warning(
                "[Wan22LivePreview] unavailable; using progress only: %s",
                exc,
            )

    if preview_session is None and live_preview is None:
        callback = latent_preview.prepare_callback(model, total_steps)

        def update(global_step, x0, x):
            callback(global_step, x0, x, total_steps)

    elif preview_session is None:
        progress = comfy.utils.ProgressBar(total_steps)

        def update(global_step, _x0, _x):
            # Never fall back to an image-bearing callback when a browser client
            # could not be identified: ComfyUI treats a missing sid as broadcast.
            progress.update_absolute(global_step + 1, total_steps)

    else:
        progress = comfy.utils.ProgressBar(total_steps)
        preview_failed = False

        def update(global_step, x0, _x):
            nonlocal preview_failed
            # ProgressBar's hook is also ComfyUI's cancellation checkpoint.
            progress.update_absolute(global_step + 1, total_steps)
            if not preview_failed:
                stage = "high noise" if global_step < high_steps else "low noise"
                try:
                    preview_session.update(global_step, stage, x0)
                except Exception:
                    preview_failed = True
                    logging.exception(
                        "[Wan22LivePreview] callback failed; sampling will continue"
                    )

    def high_callback(step, x0, x, _local_total):
        update(step, x0, x)

    def low_callback(step, x0, x, _local_total):
        update(high_steps + step, x0, x)

    return high_callback, low_callback, preview_session


class Wan22LivePreview(io.ComfyNode):
    """Configure one low-overhead animated preview for both Wan experts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Wan22LivePreview",
            display_name="Wan 2.2 Live Preview",
            search_aliases=["wan video preview", "wan animated latent preview"],
            category="model/sampling",
            description=(
                "Streams selected Wan latent-time frames as one duration-correct "
                "animation across the combined sampler's high/low handoff."
            ),
            inputs=[
                io.Float.Input(
                    "output_fps",
                    default=16.0,
                    min=1.0,
                    max=60.0,
                    step=0.01,
                    tooltip=(
                        "Target decoded-video FPS; latent playback accounts "
                        "for temporal compression."
                    ),
                ),
                io.Int.Input(
                    "max_frames",
                    default=24,
                    min=1,
                    max=128,
                    tooltip="Maximum latent-time slices per animated update.",
                ),
                io.Int.Input(
                    "preview_every",
                    default=1,
                    min=1,
                    max=100,
                    tooltip="Update every N denoise steps; the final step is always shown.",
                ),
                io.Int.Input(
                    "webp_quality",
                    default=70,
                    min=30,
                    max=100,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_size",
                    default=512,
                    min=64,
                    max=2048,
                    step=16,
                    advanced=True,
                    tooltip=(
                        "Maximum encoded side; small latent previews are scaled "
                        "only by the browser."
                    ),
                ),
            ],
            outputs=[
                io.Custom(LIVE_PREVIEW_TYPE).Output(
                    display_name="live_preview",
                    tooltip="Connect once to Wan 2.2 Combined KSampler.",
                )
            ],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        output_fps=16.0,
        max_frames=24,
        preview_every=1,
        webp_quality=70,
        max_size=512,
    ):
        output_fps = float(output_fps)
        if not math.isfinite(output_fps) or output_fps <= 0.0:
            raise ValueError("Live-preview output_fps must be finite and greater than zero.")
        hidden = getattr(cls, "hidden", None)
        node_id = getattr(hidden, "unique_id", None)
        if node_id is None:
            raise ValueError("Live-preview node execution id is unavailable.")
        config = LivePreviewConfig(
            node_id=str(node_id),
            output_fps=output_fps,
            max_frames=max(1, int(max_frames)),
            preview_every=max(1, int(preview_every)),
            webp_quality=min(100, max(30, int(webp_quality))),
            max_size=max(64, int(max_size)),
        )
        return io.NodeOutput(config)


class Wan22CombinedKSampler(io.ComfyNode):
    """Run Wan 2.2's high- and low-noise experts over one shared schedule."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Wan22CombinedKSampler",
            display_name="Wan 2.2 Combined KSampler",
            search_aliases=["wan sampler", "wan 2.2 sampler", "dual ksampler"],
            category="model/sampling",
            description=(
                "Samples Wan 2.2 high- and low-noise experts as one continuous pass. "
                "Apply matched LoRAs first, then patch each MODEL with ModelSamplingSD3."
            ),
            inputs=[
                io.Model.Input(
                    "high_noise_model",
                    tooltip="Wan 2.2 high-noise expert, already patched by ModelSamplingSD3.",
                ),
                io.Model.Input(
                    "low_noise_model",
                    tooltip="Wan 2.2 low-noise expert, already patched by ModelSamplingSD3.",
                ),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Custom(LIVE_PREVIEW_TYPE).Input(
                    "live_preview",
                    optional=True,
                    tooltip=(
                        "Optional Wan 2.2 Live Preview controller. It does not clone "
                        "or patch either MODEL."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Combo.Input(
                    "task",
                    options=[TASK_I2V, TASK_T2V],
                    default=TASK_I2V,
                    tooltip=(
                        "Selects Wan's trained expert boundary: 0.900 for I2V "
                        "or 0.875 for T2V."
                    ),
                ),
                io.Combo.Input(
                    "profile",
                    options=[PROFILE_LIGHTX2V, PROFILE_COMFYUI, PROFILE_CUSTOM],
                    default=PROFILE_LIGHTX2V,
                    tooltip=(
                        "LightX2V assumes the matched high/low distilled LoRAs "
                        "are already applied upstream at their published strength."
                    ),
                ),
                io.Int.Input("custom_steps", default=4, min=2, max=1000, advanced=True),
                io.Float.Input(
                    "custom_cfg_high",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "custom_cfg_low",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "custom_expected_shift",
                    default=5.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Expected upstream ModelSamplingSD3 shift; the node "
                        "validates but does not repatch."
                    ),
                ),
                io.Combo.Input(
                    "custom_sampler",
                    options=comfy.samplers.SAMPLER_NAMES,
                    default="euler",
                    advanced=True,
                ),
                io.Combo.Input(
                    "custom_scheduler",
                    options=comfy.samplers.SCHEDULER_NAMES,
                    default="simple",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="LATENT",
                    tooltip="The continuously denoised high-to-low Wan latent.",
                ),
                io.String.Output(
                    display_name="diagnostics",
                    tooltip=(
                        "Resolved profile, sigmas, expert split, CFG, handoff, "
                        "and per-expert wall time."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        high_noise_model,
        low_noise_model,
        positive,
        negative,
        latent_image,
        seed,
        task,
        profile,
        custom_steps=4,
        custom_cfg_high=1.0,
        custom_cfg_low=1.0,
        custom_expected_shift=5.0,
        custom_sampler="euler",
        custom_scheduler="simple",
        live_preview=None,
    ):
        if task not in TASK_BOUNDARIES:
            raise ValueError(f"Unknown Wan 2.2 task: {task}")
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            raise ValueError("LATENT input must contain a samples tensor.")
        if live_preview is not None and not isinstance(live_preview, LivePreviewConfig):
            raise ValueError("live_preview must come from the Wan 2.2 Live Preview node.")

        settings = resolve_profile(
            profile,
            custom_steps=custom_steps,
            custom_cfg_high=custom_cfg_high,
            custom_cfg_low=custom_cfg_low,
            custom_expected_shift=custom_expected_shift,
            custom_sampler=custom_sampler,
            custom_scheduler=custom_scheduler,
        )

        high_sampling = _model_sampling(high_noise_model, "High-noise")
        low_sampling = _model_sampling(low_noise_model, "Low-noise")
        _validate_model_pair(
            high_noise_model,
            low_noise_model,
            high_sampling,
            low_sampling,
            settings.shift,
        )

        high_sigmas = _calculate_schedule(
            high_sampling, settings.scheduler, settings.steps, settings.sampler_name
        )
        low_sigmas = _calculate_schedule(
            low_sampling, settings.scheduler, settings.steps, settings.sampler_name
        )
        validate_sigma_schedule(high_sigmas, "High-noise")
        validate_sigma_schedule(low_sigmas, "Low-noise")
        if high_sigmas.shape != low_sigmas.shape or not torch.allclose(
            high_sigmas, low_sigmas, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(
                "High-noise and low-noise sigma schedules differ. Use matching "
                "ModelSamplingSD3 settings on both experts."
            )

        boundary = TASK_BOUNDARIES[task]
        effective_steps = int(high_sigmas.numel() - 1)
        high_steps, high_slice, low_slice = split_sigma_schedule(high_sigmas, boundary)
        low_steps = effective_steps - high_steps

        working = dict(latent_image)
        samples = comfy.sample.fix_empty_latent_channels(
            high_noise_model,
            working["samples"],
            working.get("downscale_ratio_spacial"),
            working.get("downscale_ratio_temporal"),
        )
        working["samples"] = samples
        batch_index = working.get("batch_index")
        noise = comfy.sample.prepare_noise(samples, int(seed), batch_index)
        noise_mask = working.get("noise_mask")
        sampler = comfy.samplers.sampler_object(settings.sampler_name)
        high_callback, low_callback, preview_session = _unified_callbacks(
            high_noise_model,
            effective_steps,
            high_steps,
            high_sigmas,
            live_preview,
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        sampling_succeeded = False
        try:
            high_started = time.perf_counter()
            samples = comfy.sample.sample_custom(
                high_noise_model,
                noise,
                settings.cfg_high,
                sampler,
                high_slice,
                positive,
                negative,
                samples,
                noise_mask=noise_mask,
                callback=high_callback,
                disable_pbar=disable_pbar,
                seed=int(seed),
            )
            high_seconds = time.perf_counter() - high_started

            low_started = time.perf_counter()
            samples = comfy.sample.sample_custom(
                low_noise_model,
                _zero_noise_like(noise),
                settings.cfg_low,
                sampler,
                low_slice,
                positive,
                negative,
                samples,
                noise_mask=noise_mask,
                callback=low_callback,
                disable_pbar=disable_pbar,
                seed=int(seed),
            )
            low_seconds = time.perf_counter() - low_started
            sampling_succeeded = True
        finally:
            if preview_session is not None:
                preview_session.close(cancelled=not sampling_succeeded)

        output = dict(working)
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = samples
        sigma_text = ",".join(f"{float(sigma):.4g}" for sigma in high_sigmas)
        cfg_note = " | native CFG=1 fast path" if (
            math.isclose(settings.cfg_high, 1.0) and math.isclose(settings.cfg_low, 1.0)
        ) else ""
        step_text = f"{effective_steps} steps"
        if effective_steps != settings.steps:
            step_text = f"{settings.steps} requested / {effective_steps} effective steps"
        preview_note = ""
        if preview_session is not None:
            preview_note = f" | {preview_session.diagnostic_note()}"
        elif live_preview is not None:
            preview_note = " | live preview unavailable (progress only)"
        diagnostics = (
            f"{task} | {profile} | {step_text} | "
            f"{settings.sampler_name}/{settings.scheduler} | shift {settings.shift:g} | "
            f"CFG {settings.cfg_high:g}/{settings.cfg_low:g}{cfg_note} | "
            f"high {high_steps} + low {low_steps} | "
            f"handoff sigma {float(high_sigmas[high_steps]):.6g} | sigmas [{sigma_text}]"
            f" | wall high {high_seconds:.2f}s / low {low_seconds:.2f}s "
            f"/ total {high_seconds + low_seconds:.2f}s{preview_note}"
        )
        return io.NodeOutput(output, diagnostics)

    sample = execute


class Wan22CombinedSamplerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Wan22CombinedKSampler, Wan22LivePreview]


async def comfy_entrypoint() -> Wan22CombinedSamplerExtension:
    return Wan22CombinedSamplerExtension()


__all__ = [
    "PROFILE_LIGHTX2V",
    "PROFILE_COMFYUI",
    "PROFILE_CUSTOM",
    "TASK_I2V",
    "TASK_T2V",
    "LIVE_PREVIEW_TYPE",
    "LivePreviewConfig",
    "SamplingProfile",
    "Wan22CombinedKSampler",
    "Wan22LivePreview",
    "Wan22CombinedSamplerExtension",
    "resolve_profile",
    "split_sigma_schedule",
    "validate_sigma_schedule",
    "comfy_entrypoint",
]
