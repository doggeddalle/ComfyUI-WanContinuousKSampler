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


class _Port:
    @classmethod
    def Input(cls, name, **kwargs):
        return {"name": name, **kwargs}

    @classmethod
    def Output(cls, **kwargs):
        return kwargs


class _CustomPort:
    def __init__(self, io_type):
        self.io_type = io_type

    def Input(self, name, **kwargs):
        return {"name": name, "type": self.io_type, **kwargs}

    def Output(self, **kwargs):
        return {"type": self.io_type, **kwargs}


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *values):
        self.result = values


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
    Custom = _CustomPort
    Hidden = types.SimpleNamespace(unique_id="UNIQUE_ID")


def _install_runtime_stubs() -> None:
    comfy = types.ModuleType("comfy")
    sample = types.ModuleType("comfy.sample")
    samplers = types.ModuleType("comfy.samplers")
    utils = types.ModuleType("comfy.utils")
    nested_tensor = types.ModuleType("comfy.nested_tensor")

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

    class ProgressBar:
        def __init__(self, total):
            self.total = total

        def update_absolute(self, _value, _total=None, _preview=None):
            return None

    utils.ProgressBar = ProgressBar
    nested_tensor.NestedTensor = lambda tensors: tensors

    comfy.sample = sample
    comfy.samplers = samplers
    comfy.utils = utils
    comfy.nested_tensor = nested_tensor

    preview = types.ModuleType("latent_preview")
    preview.prepare_callback = lambda _model, _steps: lambda *_args: None

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
            "latent_preview": preview,
            "comfy_api": comfy_api,
            "comfy_api.latest": latest,
        }
    )


_install_runtime_stubs()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nodes as wan  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
