# Wan 2.2 Combined KSampler

A ComfyUI custom node that runs Wan 2.2 A14B's high-noise and low-noise experts as one continuous denoising pass. It preserves a single latent and sigma schedule across the handoff, reducing duplicated sampler wiring and making runs easier to validate and compare.

Designed for repeatable Wan 2.2 I2V and T2V workflows, including LightX2V four-step distillation.

## Features

- One combined sampler for Wan's paired high- and low-noise experts
- Exact sigma-boundary routing: I2V `0.900`, T2V `0.875`
- One initial noise sample and continuous latent handoff
- Built-in LightX2V four-step profile: Euler/simple, CFG 1, shift 5
- Validation for duplicate experts, mismatched schedules, and incorrect shifts
- Optional animated live latent preview across both sampling stages
- Works with upstream native and GGUF model-loader workflows
- Leaves model loading, LoRA patching, caching, and VRAM policy to ComfyUI

## Installation

1. Copy or clone this directory into `ComfyUI/custom_nodes/Wan22CombinedKSampler`.
2. Restart ComfyUI.
3. Add **Wan 2.2 Combined KSampler** from `model/sampling`.
4. Optionally add **Wan 2.2 Live Preview** and connect it to the sampler's `live_preview` input.

The extension uses ComfyUI's V3 API. Update ComfyUI if startup reports that `comfy_api.latest` is unavailable.

## Quick start: LightX2V 4-step I2V

1. Load separate matching Wan 2.2 high- and low-noise experts.
2. Apply the corresponding high- and low-noise LightX2V LoRAs upstream, unless you use pre-distilled models.
3. Apply `ModelSamplingSD3` with shift **5** to both experts.
4. Connect the models, conditioning, and starting latent to **Wan 2.2 Combined KSampler**.
5. Set `task` to `I2V` and select `LightX2V 4-step distilled`.
6. Send the resulting latent directly to `VAEDecode`.

The included example workflow is at:

`example_workflows/Wan22_Combined_I2V_LightX2V_4step.json`

## Sampling behavior

Wan 2.2 switches experts by noise level, not by splitting the requested step count in half. This node creates one compatible sigma schedule, runs the high-noise expert through the appropriate boundary, and continues with the low-noise expert from the same latent without adding noise again.

For the default LightX2V profile:

| Setting | Value |
| --- | --- |
| Steps | 4 |
| CFG | 1.0 |
| Shift | 5 |
| Sampler / scheduler | Euler / simple |

Use matching high/low adapter files from the same LightX2V release. Do not cross the adapters, apply both to one expert, or apply them again to already-distilled models.

## Live preview

**Wan 2.2 Live Preview** provides an animated, duration-correct view of the video latent throughout both expert passes. It is intended for motion and composition feedback—not final colour or detail judgement.

Preview work is bounded and asynchronous. If VRAM is tight, lower `max_frames` or `max_size`, or increase `preview_every`.

## Model formats and memory

The sampler accepts ComfyUI `MODEL` inputs, so native and GGUF loaders can remain upstream. It does not load files, clone models, force cache cleanup, or control ComfyUI's offloading policy.

On 24 GB GPUs, one high-to-low expert swap is normally expected; keeping both 14B experts resident is often not practical. A combined sampler avoids adding extra transitions, but cannot remove the underlying model-memory requirement.

## Diagnostics and testing

The sampler's `STRING` output reports the resolved profile, effective schedule, high/low evaluation counts, handoff sigma, and stage timings. Use it when comparing fixed-seed runs.

Run the test suite from this project directory with ComfyUI's Python environment:

```powershell
python -m unittest discover -s tests -v
```

## Notes

- The original `testwan.json` workflow is preserved unchanged.
- The known LightX2V `..._1022` LoRA unused-key warnings are expected only for that specific adapter pair; investigate other missing-key warnings.
- Custom settings outside a distilled LoRA's training schedule are experimental.

## License

Add the license for this repository here.
