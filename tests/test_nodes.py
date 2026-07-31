"""Weight-free contract tests for the Wan 2.2 combined sampler."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


class _Port:
    @classmethod
    def Input(cls, name, **kwargs):
        return {"name": name, **kwargs}

    @classmethod
    def Output(cls, name=None, **kwargs):
        if name is not None:
            kwargs.setdefault("name", name)
            kwargs.setdefault("display_name", name)
        return kwargs


class _CustomPort:
    def __init__(self, io_type):
        self.io_type = io_type

    def Input(self, name, **kwargs):
        return {"name": name, "type": self.io_type, **kwargs}

    def Output(self, name=None, **kwargs):
        output = {"type": self.io_type, **kwargs}
        if name is not None:
            output.setdefault("name", name)
            output.setdefault("display_name", name)
        return output


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *values, ui=None):
        self.result = values
        self.args = values
        self.ui = ui


class _IO:
    ComfyNode = object
    Schema = _Schema
    NodeOutput = _NodeOutput
    Model = _Port
    Conditioning = _Port
    Latent = _Port
    Int = _Port
    Float = _Port
    Combo = _Port
    String = _Port
    Boolean = _Port
    Image = _Port
    Mask = _Port
    Vae = _Port
    ClipVisionOutput = _Port
    Custom = _CustomPort
    Hidden = types.SimpleNamespace(unique_id="UNIQUE_ID")


def _install_runtime_stubs() -> None:
    comfy = types.ModuleType("comfy")
    sample = types.ModuleType("comfy.sample")
    samplers = types.ModuleType("comfy.samplers")
    utils = types.ModuleType("comfy.utils")
    nested_tensor = types.ModuleType("comfy.nested_tensor")
    model_management = types.ModuleType("comfy.model_management")
    ldm = types.ModuleType("comfy.ldm")
    sam3 = types.ModuleType("comfy.ldm.sam3")
    tracker = types.ModuleType("comfy.ldm.sam3.tracker")

    sample.fix_empty_latent_channels = lambda _model, value, *_ratios: value
    sample.prepare_noise = lambda value, _seed, _batch=None: torch.ones_like(value)
    sample.sample_custom = (
        lambda _model, _noise, _cfg, _sampler, _sigmas, _positive, _negative,
        value, **_kwargs: value
    )

    samplers.SAMPLER_NAMES = ["euler", "dpm_2"]
    samplers.SCHEDULER_NAMES = ["simple", "normal"]

    class KSampler:
        DISCARD_PENULTIMATE_SIGMA_SAMPLERS = {"dpm_2"}

    samplers.KSampler = KSampler
    samplers.sampler_object = lambda name: f"sampler:{name}"
    samplers.calculate_sigmas = lambda _sampling, _scheduler, steps: torch.linspace(
        1.0, 0.0, steps + 1
    )
    utils.PROGRESS_BAR_ENABLED = True
    utils.common_upscale = lambda value, width, height, _method, _crop=None, **_kwargs: F.interpolate(
        value, size=(height, width), mode="nearest"
    )
    utils.unpack_latents = lambda value, _shapes: value

    class ProgressBar:
        def __init__(self, total):
            self.total = total

        def update_absolute(self, _value, _total=None, _preview=None):
            return None

        def update(self, _value):
            return None

    utils.ProgressBar = ProgressBar
    nested_tensor.NestedTensor = lambda tensors: tensors
    model_management.throw_exception_if_processing_interrupted = lambda: None
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.intermediate_dtype = lambda: torch.float32
    model_management.get_torch_device = lambda: torch.device("cpu")
    model_management.load_model_gpu = lambda _model: None
    tracker.unpack_masks = lambda packed: packed

    comfy.sample = sample
    comfy.samplers = samplers
    comfy.utils = utils
    comfy.nested_tensor = nested_tensor
    comfy.model_management = model_management
    comfy.ldm = ldm
    ldm.sam3 = sam3
    sam3.tracker = tracker

    preview = types.ModuleType("latent_preview")
    preview.prepare_callback = lambda _model, _steps, _x0=None: lambda *_args: None

    node_helpers = types.ModuleType("node_helpers")
    node_helpers.conditioning_set_values = (
        lambda conditioning, _values, append=False: conditioning
    )
    node_helpers.conditioning_set_values_with_timestep_range = (
        lambda conditioning, _values, _start, _end: conditioning
    )

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: str(ROOT / "tests" / "_temp")

    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = object
    latest.io = _IO
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest

    sys.modules.update(
        {
            "comfy": comfy,
            "comfy.sample": sample,
            "comfy.samplers": samplers,
            "comfy.utils": utils,
            "comfy.nested_tensor": nested_tensor,
            "comfy.model_management": model_management,
            "comfy.ldm": ldm,
            "comfy.ldm.sam3": sam3,
            "comfy.ldm.sam3.tracker": tracker,
            "latent_preview": preview,
            "node_helpers": node_helpers,
            "folder_paths": folder_paths,
            "comfy_api": comfy_api,
            "comfy_api.latest": latest,
        }
    )


_install_runtime_stubs()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nodes as wan  # noqa: E402
import scail_core as core  # noqa: E402
import scail_identity as identity  # noqa: E402
import scail_nodes as scail  # noqa: E402


class _Sampling:
    def __init__(self, shift: float):
        self.shift = shift


class _Model:
    def __init__(self, shift: float):
        self.sampling = _Sampling(shift)

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return self.sampling

    def clone(self):
        raise AssertionError("The V3 node must not clone or retain heavyweight MODEL inputs.")

    def is_clone(self, _other):
        return False


class Wan22CombinedKSamplerTests(unittest.TestCase):
    def test_profile_resolution(self):
        light = wan.resolve_profile(wan.PROFILE_LIGHTX2V)
        self.assertEqual(
            light,
            wan.SamplingProfile(4, 1.0, 1.0, 5.0, "euler", "simple"),
        )

        standard = wan.resolve_profile(wan.PROFILE_COMFYUI)
        self.assertEqual(
            standard,
            wan.SamplingProfile(20, 3.5, 3.5, 8.0, "euler", "simple"),
        )

        custom = wan.resolve_profile(
            wan.PROFILE_CUSTOM,
            custom_steps=7,
            custom_cfg_high=1.25,
            custom_cfg_low=1.5,
            custom_expected_shift=6.0,
            custom_sampler="euler",
            custom_scheduler="normal",
        )
        self.assertEqual(
            custom,
            wan.SamplingProfile(7, 1.25, 1.5, 6.0, "euler", "normal"),
        )

    def test_split_math_keeps_boundary_equality_on_high_expert(self):
        sigmas = torch.tensor([1.0, 0.9, 0.8, 0.0])
        k, high, low = wan.split_sigma_schedule(sigmas, 0.9)
        self.assertEqual(k, 2)
        self.assertTrue(torch.equal(high, torch.tensor([1.0, 0.9, 0.8])))
        self.assertTrue(torch.equal(low, torch.tensor([0.8, 0.0])))
        self.assertEqual(float(high[-1]), float(low[0]))

        t2v_sigmas = torch.tensor([1.0, 0.875, 0.7, 0.0])
        t2v_k, _, _ = wan.split_sigma_schedule(t2v_sigmas, 0.875)
        self.assertEqual(t2v_k, 2)

    def test_published_and_reference_schedules_route_expected_experts(self):
        distilled = torch.tensor([1.0, 0.9375, 0.8333333, 0.625, 0.0])
        for boundary in (0.900, 0.875):
            k, high, low = wan.split_sigma_schedule(distilled, boundary)
            self.assertEqual((k, high.numel() - 1, low.numel() - 1), (2, 2, 2))

        reference = torch.tensor(
            [1.0, 0.975724, 0.941259, 0.888889, 0.800479, 0.615952, 0.0]
        )
        k, high, low = wan.split_sigma_schedule(reference, 0.900)
        self.assertEqual((k, high.numel() - 1, low.numel() - 1), (3, 3, 3))

    def test_schedule_validation_rejects_shape_order_and_nonzero_terminal(self):
        wan.validate_sigma_schedule(torch.tensor([1.0, 0.5, 0.0]), "valid")
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            wan.validate_sigma_schedule(torch.ones((1, 3)), "bad")
        with self.assertRaisesRegex(ValueError, "at least 3"):
            wan.validate_sigma_schedule(torch.tensor([1.0, 0.0]), "bad")
        with self.assertRaisesRegex(ValueError, "strictly descending"):
            wan.validate_sigma_schedule(torch.tensor([1.0, 0.5, 0.5, 0.0]), "bad")
        with self.assertRaisesRegex(ValueError, "end at zero"):
            wan.validate_sigma_schedule(torch.tensor([1.0, 0.5, 0.1]), "bad")

    def test_custom_scheduler_uses_its_effective_schedule_length(self):
        high_model = _Model(5.0)
        low_model = _Model(5.0)
        latent = {"samples": torch.zeros((1, 4, 2, 2, 2))}
        schedule = torch.tensor([1.0, 0.95, 0.92, 0.91, 0.89, 0.75, 0.3, 0.0])
        callback_totals = []
        sampled_sigmas = []

        def prepare_callback(_model, total):
            callback_totals.append(total)
            return lambda *_args: None

        def sample_custom(
            _model, _noise, _cfg, _sampler, sigmas, _positive, _negative, value, **_kwargs
        ):
            sampled_sigmas.append(sigmas.clone())
            return value

        with (
            mock.patch.object(
                wan.comfy.samplers,
                "calculate_sigmas",
                side_effect=[schedule.clone(), schedule.clone()],
            ),
            mock.patch.object(wan.comfy.sample, "sample_custom", side_effect=sample_custom),
            mock.patch.object(wan.latent_preview, "prepare_callback", side_effect=prepare_callback),
        ):
            result = wan.Wan22CombinedKSampler.execute(
                high_model,
                low_model,
                None,
                None,
                latent,
                0,
                wan.TASK_I2V,
                wan.PROFILE_CUSTOM,
                custom_steps=6,
                custom_scheduler="normal",
            )

        self.assertEqual(callback_totals, [7])
        self.assertEqual([item.numel() - 1 for item in sampled_sigmas], [4, 3])
        self.assertIn("6 requested / 7 effective steps", result.result[1])
        self.assertIn("high 4 + low 3", result.result[1])

    def test_live_preview_settings_and_duration_correct_webp(self):
        schema = wan.Wan22LivePreview.define_schema()
        self.assertEqual(schema.node_id, "Wan22LivePreview")
        self.assertEqual(schema.outputs[0]["type"], wan.LIVE_PREVIEW_TYPE)
        self.assertEqual(schema.hidden, [wan.io.Hidden.unique_id])

        previous_hidden = getattr(wan.Wan22LivePreview, "hidden", None)
        wan.Wan22LivePreview.hidden = types.SimpleNamespace(unique_id="12:7")
        try:
            result = wan.Wan22LivePreview.execute(16.0, 3, 1, 70, 512)
        finally:
            if previous_hidden is None:
                delattr(wan.Wan22LivePreview, "hidden")
            else:
                wan.Wan22LivePreview.hidden = previous_hidden

        config = result.result[0]
        self.assertEqual(
            config,
            wan.LivePreviewConfig("12:7", 16.0, 3, 1, 70, 512),
        )

        factors = [[0.0, 0.0, 0.0] for _ in range(16)]
        factors[0] = [1.0, 0.0, 0.0]
        latent_format = types.SimpleNamespace(
            latent_rgb_factors=factors,
            latent_rgb_factors_bias=[0.0, 0.0, 0.0],
            latent_rgb_factors_reshape=None,
            temporal_downscale_ratio=4,
        )
        x0 = torch.zeros((1, 16, 5, 2, 3))
        x0[0, 0, :, :, :] = torch.linspace(-1.0, 1.0, 5).view(5, 1, 1)
        frames, source_frames = wan._preview_rgb_frames(x0, latent_format, 3)
        self.assertEqual(tuple(frames.shape), (3, 2, 3, 3))
        self.assertEqual((frames.dtype, frames.device.type, source_frames), (torch.uint8, "cpu", 5))

        staged, staged_source_frames, ready_event = wan._stage_preview_rgb_frames(
            torch.zeros((1, 16, 3, 6, 9)),
            latent_format,
            3,
            4,
        )
        self.assertEqual(tuple(staged.shape), (3, 3, 4, 3))
        self.assertEqual(staged_source_frames, 3)
        self.assertIsNone(ready_event)

        encoded, width, height, decoded_frames, effective_fps = wan._encode_preview_webp(
            frames,
            output_fps=16.0,
            temporal_ratio=4,
            source_frames=source_frames,
            quality=70,
            max_size=512,
        )
        payload = base64.b64decode(encoded)
        self.assertEqual(payload[:4], b"RIFF")
        self.assertEqual(payload[8:12], b"WEBP")
        self.assertEqual((width, height, decoded_frames), (3, 2, 17))
        self.assertAlmostEqual(effective_fps, 3 * 16 / 17)

    def test_preview_encoder_keeps_latest_pending_update(self):
        encoder = wan._LatestPreviewEncoder()
        started = threading.Event()
        release = threading.Event()
        executed = []

        def first():
            executed.append("first")
            started.set()
            release.wait(timeout=2.0)

        encoder.submit(first)
        self.assertTrue(started.wait(timeout=2.0))
        encoder.submit(lambda: executed.append("stale"))
        encoder.submit(lambda: executed.append("latest"))
        release.set()
        encoder.close(cancelled=False, timeout=2.0)

        self.assertEqual(executed, ["first", "latest"])
        self.assertEqual(encoder.dropped, 1)

    def test_live_preview_session_emits_targeted_webp_event(self):
        factors = [[0.0, 0.0, 0.0] for _ in range(16)]
        factors[0] = [1.0, 0.0, 0.0]
        latent_format = types.SimpleNamespace(
            latent_rgb_factors=factors,
            latent_rgb_factors_bias=[0.0, 0.0, 0.0],
            latent_rgb_factors_reshape=None,
            temporal_downscale_ratio=4,
        )
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(latent_format=latent_format)
        )
        events = []

        class FakeServer:
            client_id = "client-7"

            def send_sync(self, event, payload, client_id):
                events.append((event, payload, client_id))

        config = wan.LivePreviewConfig("9:2", 16.0, 3, 1, 70, 512)
        with mock.patch.object(
            wan,
            "PromptServer",
            types.SimpleNamespace(instance=FakeServer()),
        ):
            session = wan._LivePreviewSession(
                model,
                config,
                4,
                torch.tensor([1.0, 0.95, 0.85, 0.5, 0.0]),
            )
            session.update(2, "low noise", torch.zeros((1, 16, 3, 2, 2)))
            session.close(cancelled=False)
            self.assertTrue(session._encoder.wait_idle(2.0))

        self.assertEqual([event[0] for event in events], ["wan22_live_preview"] * 2)
        self.assertTrue(all(event[2] == "client-7" for event in events))
        start, preview = events[0][1], events[1][1]
        self.assertEqual((start["node_id"], start["global_total"]), ("9:2", 4))
        self.assertEqual(preview["run_id"], start["run_id"])
        self.assertEqual(
            (preview["stage"], preview["global_step"], preview["frame_count"]),
            ("low noise", 3, 3),
        )
        payload = base64.b64decode(preview["webp_base64"])
        self.assertEqual((payload[:4], payload[8:12]), (b"RIFF", b"WEBP"))

    def test_live_preview_fails_closed_without_browser_client(self):
        latent_format = types.SimpleNamespace(
            latent_rgb_factors=[[0.0, 0.0, 0.0] for _ in range(16)],
            latent_rgb_factors_bias=None,
            latent_rgb_factors_reshape=None,
            temporal_downscale_ratio=4,
        )
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(latent_format=latent_format)
        )
        events = []

        class FakeServer:
            client_id = None

            def send_sync(self, *args):
                events.append(args)

        config = wan.LivePreviewConfig("9:2", 16.0, 3, 1, 70, 512)
        with (
            mock.patch.object(
                wan,
                "PromptServer",
                types.SimpleNamespace(instance=FakeServer()),
            ),
            self.assertRaisesRegex(RuntimeError, "active ComfyUI browser client"),
        ):
            wan._LivePreviewSession(
                model,
                config,
                4,
                torch.tensor([1.0, 0.95, 0.85, 0.5, 0.0]),
            )

        self.assertEqual(events, [])

    def test_unavailable_custom_preview_uses_progress_only(self):
        progress = []

        class ProgressBar:
            def __init__(self, total):
                self.total = total

            def update_absolute(self, value, total=None, preview=None):
                progress.append((value, total, preview))

        config = wan.LivePreviewConfig("9:2", 16.0, 3, 1, 70, 512)
        with (
            mock.patch.object(
                wan,
                "_LivePreviewSession",
                side_effect=RuntimeError("no browser client"),
            ),
            mock.patch.object(wan.comfy.utils, "ProgressBar", ProgressBar),
            mock.patch.object(wan.latent_preview, "prepare_callback") as native_callback,
        ):
            high_callback, low_callback, session = wan._unified_callbacks(
                object(),
                4,
                2,
                torch.tensor([1.0, 0.95, 0.85, 0.5, 0.0]),
                config,
            )
            high_callback(0, None, None, 2)
            low_callback(1, None, None, 2)

        self.assertIsNone(session)
        self.assertEqual(progress, [(1, 4, None), (4, 4, None)])
        native_callback.assert_not_called()

    def test_live_preview_sessions_share_one_bounded_worker_and_cancel_late_send(self):
        factors = [[0.0, 0.0, 0.0] for _ in range(16)]
        factors[0] = [1.0, 0.0, 0.0]
        latent_format = types.SimpleNamespace(
            latent_rgb_factors=factors,
            latent_rgb_factors_bias=None,
            latent_rgb_factors_reshape=None,
            temporal_downscale_ratio=4,
        )
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(latent_format=latent_format)
        )
        events = []
        started = threading.Event()
        release = threading.Event()
        calls = 0

        class FakeServer:
            client_id = "client-7"

            def send_sync(self, event, payload, client_id):
                events.append((event, payload, client_id))

        def slow_encode(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                release.wait(timeout=2.0)
            return "AAAA", 2, 2, 9, 16 / 3

        config_a = wan.LivePreviewConfig("11", 16.0, 3, 1, 70, 512)
        config_b = wan.LivePreviewConfig("12", 16.0, 3, 1, 70, 512)
        with (
            mock.patch.object(
                wan,
                "PromptServer",
                types.SimpleNamespace(instance=FakeServer()),
            ),
            mock.patch.object(wan, "_encode_preview_webp", side_effect=slow_encode),
        ):
            first = wan._LivePreviewSession(
                model,
                config_a,
                1,
                torch.tensor([1.0, 0.0]),
            )
            second = wan._LivePreviewSession(
                model,
                config_b,
                1,
                torch.tensor([1.0, 0.0]),
            )
            self.assertIs(first._encoder, second._encoder)
            first.update(0, "high noise", torch.zeros((1, 16, 3, 2, 2)))
            self.assertTrue(started.wait(timeout=2.0))
            first.close(cancelled=True)
            second.update(0, "high noise", torch.zeros((1, 16, 3, 2, 2)))
            release.set()
            self.assertTrue(second._encoder.wait_idle(2.0))

        previews = [payload for _, payload, _ in events if "webp_base64" in payload]
        self.assertEqual([payload["node_id"] for payload in previews], ["12"])

    def test_live_preview_is_continuous_across_both_experts(self):
        high_model = _Model(5.0)
        low_model = _Model(5.0)
        latent = {"samples": torch.zeros((1, 4, 3, 2, 2))}
        updates = []
        closes = []
        progress = []

        class FakeSession:
            def __init__(self, _model, _config, total, sigmas):
                self.total = total
                self.sigmas = sigmas

            def update(self, global_step, stage, x0):
                updates.append((global_step, stage, x0.shape[2]))

            def close(self, *, cancelled):
                closes.append(cancelled)

            def diagnostic_note(self):
                return "live preview test"

        class ProgressBar:
            def __init__(self, total):
                self.total = total

            def update_absolute(self, value, total=None, preview=None):
                progress.append((value, total, preview))

        def calculate_sigmas(_sampling, _scheduler, _steps):
            return torch.tensor([1.0, 0.95, 0.85, 0.5, 0.0])

        def sample_custom(
            _model, _noise, _cfg, _sampler, sigmas, _positive, _negative, value, **kwargs
        ):
            for step in range(sigmas.numel() - 1):
                kwargs["callback"](step, value, value, sigmas.numel() - 1)
            return value

        config = wan.LivePreviewConfig("44", 16.0, 24, 1, 70, 512)
        with (
            mock.patch.object(wan, "_LivePreviewSession", FakeSession),
            mock.patch.object(wan.comfy.utils, "ProgressBar", ProgressBar),
            mock.patch.object(wan.comfy.samplers, "calculate_sigmas", side_effect=calculate_sigmas),
            mock.patch.object(wan.comfy.sample, "sample_custom", side_effect=sample_custom),
            mock.patch.object(wan.latent_preview, "prepare_callback") as native_callback,
        ):
            result = wan.Wan22CombinedKSampler.execute(
                high_model,
                low_model,
                None,
                None,
                latent,
                0,
                wan.TASK_I2V,
                wan.PROFILE_LIGHTX2V,
                live_preview=config,
            )

        self.assertEqual(
            updates,
            [
                (0, "high noise", 3),
                (1, "high noise", 3),
                (2, "low noise", 3),
                (3, "low noise", 3),
            ],
        )
        self.assertEqual(progress, [(1, 4, None), (2, 4, None), (3, 4, None), (4, 4, None)])
        self.assertEqual(closes, [False])
        native_callback.assert_not_called()
        self.assertIn("live preview test", result.result[1])

    def test_cancelled_sampling_discards_pending_preview_work(self):
        high_model = _Model(5.0)
        low_model = _Model(5.0)
        latent = {"samples": torch.zeros((1, 4, 3, 2, 2))}
        closes = []

        class FakeSession:
            def __init__(self, *_args):
                pass

            def update(self, *_args):
                pass

            def close(self, *, cancelled):
                closes.append(cancelled)

        config = wan.LivePreviewConfig("44", 16.0, 24, 1, 70, 512)
        with (
            mock.patch.object(wan, "_LivePreviewSession", FakeSession),
            mock.patch.object(
                wan.comfy.samplers,
                "calculate_sigmas",
                return_value=torch.tensor([1.0, 0.95, 0.85, 0.5, 0.0]),
            ),
            mock.patch.object(
                wan.comfy.sample,
                "sample_custom",
                side_effect=RuntimeError("cancelled"),
            ),
            self.assertRaisesRegex(RuntimeError, "cancelled"),
        ):
            wan.Wan22CombinedKSampler.execute(
                high_model,
                low_model,
                None,
                None,
                latent,
                0,
                wan.TASK_I2V,
                wan.PROFILE_LIGHTX2V,
                live_preview=config,
            )

        self.assertEqual(closes, [True])

    def test_one_noise_two_expert_calls_metadata_and_unified_progress(self):
        high_model = _Model(5.0)
        low_model = _Model(5.0)
        latent_samples = torch.zeros((2, 4, 3, 8, 8))
        mask = torch.ones((2, 3, 8, 8))
        batch_index = [4, 9]
        marker = {"kept": True}
        latent = {
            "samples": latent_samples,
            "noise_mask": mask,
            "batch_index": batch_index,
            "custom_metadata": marker,
            "downscale_ratio_spacial": 8,
            "downscale_ratio_temporal": 4,
        }

        noise_calls = []
        sample_calls = []
        progress = []

        def prepare_noise(value, seed, batch):
            noise_calls.append((value, seed, batch))
            return torch.full_like(value, 7.0)

        def calculate_sigmas(_sampling, _scheduler, steps):
            self.assertEqual(steps, 4)
            return torch.tensor([1.0, 0.95, 0.85, 0.5, 0.0])

        def prepare_callback(_model, total):
            self.assertEqual(total, 4)

            def callback(step, _x0, _x, callback_total):
                progress.append((step, callback_total))

            return callback

        def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, value, **kwargs):
            sample_calls.append(
                {
                    "model": model,
                    "noise": noise.clone(),
                    "cfg": cfg,
                    "sampler": sampler,
                    "sigmas": sigmas.clone(),
                    "positive": positive,
                    "negative": negative,
                    "value": value.clone(),
                    **kwargs,
                }
            )
            for step in range(sigmas.numel() - 1):
                kwargs["callback"](step, None, None, sigmas.numel() - 1)
            return value + (1.0 if model is high_model else 2.0)

        with (
            mock.patch.object(wan.comfy.sample, "prepare_noise", side_effect=prepare_noise),
            mock.patch.object(wan.comfy.sample, "sample_custom", side_effect=sample_custom),
            mock.patch.object(wan.comfy.samplers, "calculate_sigmas", side_effect=calculate_sigmas),
            mock.patch.object(wan.latent_preview, "prepare_callback", side_effect=prepare_callback),
        ):
            result = wan.Wan22CombinedKSampler.execute(
                high_model,
                low_model,
                "positive",
                "negative",
                latent,
                123,
                wan.TASK_I2V,
                wan.PROFILE_LIGHTX2V,
            )

        output, diagnostics = result.result
        self.assertEqual(len(noise_calls), 1)
        self.assertIs(noise_calls[0][2], batch_index)
        self.assertEqual(noise_calls[0][1], 123)
        self.assertEqual(len(sample_calls), 2)
        self.assertIs(sample_calls[0]["model"], high_model)
        self.assertIs(sample_calls[1]["model"], low_model)
        self.assertTrue(torch.all(sample_calls[0]["noise"] == 7.0))
        self.assertTrue(torch.count_nonzero(sample_calls[1]["noise"]) == 0)
        self.assertTrue(
            torch.equal(sample_calls[0]["sigmas"], torch.tensor([1.0, 0.95, 0.85]))
        )
        self.assertTrue(
            torch.equal(sample_calls[1]["sigmas"], torch.tensor([0.85, 0.5, 0.0]))
        )
        self.assertTrue(torch.all(sample_calls[1]["value"] == 1.0))
        self.assertIs(sample_calls[0]["noise_mask"], mask)
        self.assertIs(sample_calls[1]["noise_mask"], mask)
        self.assertEqual(progress, [(0, 4), (1, 4), (2, 4), (3, 4)])
        self.assertTrue(torch.all(output["samples"] == 3.0))
        self.assertIs(output["noise_mask"], mask)
        self.assertIs(output["batch_index"], batch_index)
        self.assertIs(output["custom_metadata"], marker)
        self.assertNotIn("downscale_ratio_spacial", output)
        self.assertNotIn("downscale_ratio_temporal", output)
        self.assertIn("high 2 + low 2", diagnostics)
        self.assertIn("handoff sigma 0.85", diagnostics)
        self.assertIn("native CFG=1 fast path", diagnostics)
        self.assertIn("sigmas [1,0.95,0.85,0.5,0]", diagnostics)
        self.assertIn("wall high", diagnostics)

    def test_rejects_two_clones_of_one_underlying_expert(self):
        class CloneModel(_Model):
            def __init__(self, shift, base):
                super().__init__(shift)
                self.base = base

            def is_clone(self, other):
                return isinstance(other, CloneModel) and self.base is other.base

        base = object()
        latent = {"samples": torch.zeros((1, 4, 2, 2, 2))}
        with self.assertRaisesRegex(ValueError, "two clones of one expert"):
            wan.Wan22CombinedKSampler.execute(
                CloneModel(5.0, base),
                CloneModel(5.0, base),
                None,
                None,
                latent,
                0,
                wan.TASK_I2V,
                wan.PROFILE_LIGHTX2V,
            )

    def test_mismatched_shift_or_schedule_fails_before_noise(self):
        high_model = _Model(5.0)
        low_model = _Model(8.0)
        latent = {"samples": torch.zeros((1, 4, 2, 2, 2))}
        with self.assertRaisesRegex(ValueError, "requires ModelSamplingSD3 shift 5"):
            wan.Wan22CombinedKSampler.execute(
                high_model,
                low_model,
                None,
                None,
                latent,
                0,
                wan.TASK_I2V,
                wan.PROFILE_LIGHTX2V,
            )

        low_model = _Model(5.0)
        schedules = [
            torch.tensor([1.0, 0.95, 0.85, 0.5, 0.0]),
            torch.tensor([1.0, 0.94, 0.84, 0.5, 0.0]),
        ]
        with (
            mock.patch.object(wan.comfy.samplers, "calculate_sigmas", side_effect=schedules),
            mock.patch.object(wan.comfy.sample, "prepare_noise") as prepare_noise,
            self.assertRaisesRegex(ValueError, "sigma schedules differ"),
        ):
            wan.Wan22CombinedKSampler.execute(
                high_model,
                low_model,
                None,
                None,
                latent,
                0,
                wan.TASK_I2V,
                wan.PROFILE_LIGHTX2V,
            )
        prepare_noise.assert_not_called()

    def test_v3_schema_and_entrypoint(self):
        schema = wan.Wan22CombinedKSampler.define_schema()
        self.assertEqual(schema.node_id, "Wan22CombinedKSampler")
        self.assertEqual(schema.category, "model/sampling")
        self.assertEqual(
            [item["name"] for item in schema.inputs],
            [
                "high_noise_model",
                "low_noise_model",
                "positive",
                "negative",
                "latent_image",
                "live_preview",
                "seed",
                "task",
                "profile",
                "custom_steps",
                "custom_cfg_high",
                "custom_cfg_low",
                "custom_expected_shift",
                "custom_sampler",
                "custom_scheduler",
            ],
        )
        extension = asyncio.run(wan.comfy_entrypoint())
        self.assertEqual(
            asyncio.run(extension.get_node_list()),
            [wan.Wan22CombinedKSampler, wan.Wan22LivePreview],
        )

    def test_source_has_no_manual_model_lifecycle_or_lora_loading(self):
        source = (ROOT / "nodes.py").read_text(encoding="utf-8").lower()
        forbidden = (
            "unload_all_models",
            "soft_empty_cache",
            "cleanup_models(",
            "gc.collect",
            "load_lora",
            "load_torch_file",
            "add_wrapper",
            "torch.cuda.synchronize",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


class WanSCAILAutoExtendSamplerTests(unittest.TestCase):
    def test_chunk_plan_matches_scail_temporal_contract(self):
        self.assertEqual(
            scail.plan_scail_chunks(81).chunk_lengths,
            (81,),
        )
        self.assertEqual(
            scail.plan_scail_chunks(157).chunk_lengths,
            (81, 81),
        )
        plan = scail.plan_scail_chunks(121)
        self.assertEqual((plan.effective_frames, plan.chunk_lengths), (121, (81, 45)))

        trimmed = scail.plan_scail_chunks(160)
        self.assertEqual(
            (trimmed.effective_frames, trimmed.dropped_tail_frames, trimmed.chunk_lengths),
            (157, 3, (81, 81)),
        )
        capped = scail.plan_scail_chunks(500, max_frames=100)
        self.assertEqual(
            (capped.requested_frames, capped.effective_frames, capped.chunk_lengths),
            (100, 97, (81, 21)),
        )

    def test_chunk_plan_rejects_invalid_lengths(self):
        for kwargs, message in (
            ({"frame_count": 0}, "at least one frame"),
            ({"frame_count": 81, "chunk_length": 80}, "4n\\+1"),
            ({"frame_count": 81, "overlap": 4}, "4n\\+1"),
            (
                {"frame_count": 81, "chunk_length": 81, "overlap": 81},
                "must be smaller",
            ),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                scail.plan_scail_chunks(**kwargs)

    def test_scheduler_matches_basic_scheduler_contract(self):
        sampling = _Sampling(5.0)

        def calculate_sigmas(received, scheduler, steps):
            self.assertIs(received, sampling)
            self.assertEqual((scheduler, steps), ("simple", 16))
            return torch.linspace(2.0, 0.0, 17)

        with mock.patch.object(
            scail.comfy.samplers,
            "calculate_sigmas",
            side_effect=calculate_sigmas,
        ):
            sigmas = scail.build_scail_sigmas(sampling, "simple", 8, 0.5)

        self.assertEqual(sigmas.shape, (9,))
        self.assertAlmostEqual(float(sigmas[0]), 1.0)
        self.assertEqual(float(sigmas[-1]), 0.0)

    def test_auto_extend_executes_and_stitches_workflow_equivalent_chunks(self):
        calls = {"condition": [], "seeds": [], "color": 0}

        def fake_prepare(**kwargs):
            calls["condition"].append(kwargs)
            length = kwargs["length"]
            previous = kwargs["previous_frames"]
            adjusted = kwargs["video_frame_offset"]
            if previous is not None:
                adjusted = max(0, adjusted - previous.shape[0])
            latent = {
                "samples": torch.zeros(
                    (1, 16, ((length - 1) // 4) + 1, 1, 1)
                )
            }
            return "pos", "neg", latent, adjusted + length

        def fake_sample(**kwargs):
            calls["seeds"].append(kwargs["seed"])
            return kwargs["latent"]

        def fake_color(image_target, _image_reference, _strength):
            calls["color"] += 1
            return image_target

        class FakeVAE:
            def decode(self, samples):
                frames = (samples.shape[2] - 1) * 4 + 1
                return torch.zeros((frames, 2, 2, 3))

            def decode_tiled(self, _samples):
                raise AssertionError("Normal decode was selected.")

        model = _Model(5.0)
        pose = torch.zeros((121, 4, 4, 3))
        mask = torch.zeros_like(pose)
        with (
            mock.patch.object(scail, "prepare_scail_window", side_effect=fake_prepare),
            mock.patch.object(scail, "sample_scail_window", side_effect=fake_sample),
            mock.patch.object(scail, "reinhard_color_transfer", side_effect=fake_color),
            mock.patch.object(
                scail.comfy.samplers,
                "calculate_sigmas",
                return_value=torch.linspace(1.0, 0.0, 9),
            ),
            mock.patch.object(
                scail.comfy.samplers,
                "sampler_object",
                return_value="euler-sampler",
            ),
        ):
            output = scail.WanSCAILAutoExtendSampler.execute(
                model=model,
                positive="positive",
                negative="negative",
                vae=FakeVAE(),
                pose_video=pose,
                pose_video_mask=mask,
                width=512,
                height=896,
                seed=123,
                steps=6,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=1.0,
                expected_shift=5.0,
                chunk_length=81,
                overlap=5,
                max_frames=0,
                seed_mode=scail.SEED_FIXED,
                decode_mode=scail.DECODE_NORMAL,
                color_transfer=True,
                color_transfer_strength=1.0,
                replacement_mode=True,
                pose_strength=1.0,
                pose_start=0.0,
                pose_end=1.0,
                add_noise=True,
            )

        images, frame_count, diagnostics = output.result
        self.assertEqual((images.shape[0], frame_count), (121, 121))
        self.assertEqual([call["length"] for call in calls["condition"]], [81, 45])
        self.assertIsNone(calls["condition"][0]["previous_frames"])
        self.assertEqual(calls["condition"][1]["previous_frames"].shape[0], 5)
        self.assertEqual(calls["seeds"], [123, 123])
        self.assertEqual(calls["color"], 1)
        self.assertIn("chunks [81,45] overlap 5", diagnostics)
        self.assertIn("6 steps euler/simple", diagnostics)
        self.assertIn("shift 5", diagnostics)

    def test_short_segmentation_mask_fails_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "fewer frames"):
            scail.WanSCAILAutoExtendSampler.execute(
                model=_Model(5.0),
                positive=None,
                negative=None,
                vae=None,
                pose_video=torch.zeros((81, 2, 2, 3)),
                pose_video_mask=torch.zeros((77, 2, 2, 3)),
                width=512,
                height=896,
                seed=0,
                steps=6,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=1.0,
                expected_shift=5.0,
                chunk_length=81,
                overlap=5,
                max_frames=0,
                seed_mode=scail.SEED_FIXED,
                decode_mode=scail.DECODE_NORMAL,
                color_transfer=True,
                color_transfer_strength=1.0,
                replacement_mode=True,
                pose_strength=1.0,
                pose_start=0.0,
                pose_end=1.0,
                add_noise=True,
            )

    def test_scail_v3_schema(self):
        schema = scail.WanSCAILAutoExtendSampler.define_schema()
        self.assertEqual(schema.node_id, "WanSCAILAutoExtendSampler")
        self.assertEqual(schema.category, "model/sampling/wan")
        self.assertEqual(
            [item["display_name"] for item in schema.outputs],
            ["images", "frame_count", "diagnostics"],
        )
        inputs = {item["name"]: item for item in schema.inputs}
        self.assertEqual(inputs["steps"]["default"], 6)
        self.assertEqual(inputs["chunk_length"]["default"], 81)
        self.assertEqual(inputs["overlap"]["default"], 5)
        self.assertEqual(inputs["seed_mode"]["default"], scail.SEED_FIXED)

    def test_media_prep_preserves_full_batches_and_geometry(self):
        pose = torch.zeros((13, 10, 14, 3))
        reference = torch.zeros((2, 8, 12, 3))
        output = scail.WanSCAILMediaPrep.execute(
            pose,
            reference,
            64,
            96,
            "bicubic",
            "center crop",
            "stretch",
        )
        prepared_pose, prepared_reference, width, height, diagnostics = output.result
        self.assertEqual(prepared_pose.shape, (13, 96, 64, 3))
        self.assertEqual(prepared_reference.shape, (2, 96, 64, 3))
        self.assertEqual((width, height), (64, 96))
        self.assertIn("13 frame(s)", diagnostics)
        self.assertIn("2 view(s)", diagnostics)

    def test_scail_mask_contract_and_invalid_temporal_length(self):
        video = torch.zeros((5, 16, 24, 3))
        video[..., 2] = 1.0
        mask = core.extract_mask_to_28ch(video)
        self.assertEqual(mask.shape, (1, 2, 28, 2, 3))
        self.assertTrue(bool((mask[:, :, 3::7] == 1).all()))
        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            core.extract_mask_to_28ch(video[:4])

    def test_identity_control_preview_only_is_lightweight_and_schema_is_v3(self):
        schema = identity.WanSCAILIdentityControl.define_schema()
        self.assertEqual(schema.node_id, "WanSCAILIdentityControl")
        self.assertTrue(schema.is_output_node)
        inputs = {item["name"]: item for item in schema.inputs}
        self.assertEqual(inputs["max_identities"]["default"], 6)
        self.assertEqual(inputs["replacement_mode"]["default"], True)
        reference = torch.zeros((1, 32, 32, 3))
        pose = torch.zeros((9, 32, 32, 3))
        with mock.patch.object(
            identity,
            "_save_canvas_preview",
            return_value={"filename": "preview.png", "subfolder": "", "type": "temp"},
        ):
            output = identity.WanSCAILIdentityControl.execute(
                sam3_model=object(),
                reference_image=reference,
                pose_video=pose,
                refine_iterations=2,
                auto_detect=False,
                detection_threshold=0.5,
                max_identities=6,
                detect_interval=1,
                object_indices="0",
                sort_by="none",
                replacement_mode=True,
                markers='{"reference":[],"driving":[]}',
            )
        self.assertEqual(output.result[0].shape, pose.shape)
        self.assertEqual(output.result[2].shape, pose.shape)
        self.assertEqual(output.result[3].shape, reference.shape)
        self.assertEqual(output.result[4], 0)
        self.assertIn("preview only", output.result[5])
        self.assertIn("reference_preview", output.ui)

    def test_consolidated_scail_source_does_not_call_replaced_nodes(self):
        source = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ("scail_core.py", "scail_identity.py", "scail_nodes.py")
        )
        for token in (
            "from comfy_extras.nodes_scail",
            "import comfy_extras.nodes_scail",
            "from comfy_extras.nodes_custom_sampler",
            "import comfy_extras.nodes_custom_sampler",
            "from comfy_extras.nodes_post_processing",
            "import comfy_extras.nodes_post_processing",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
