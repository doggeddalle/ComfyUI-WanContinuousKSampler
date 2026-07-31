# ComfyUI-WanContinuousKSampler

A ComfyUI node pack for repeatable Wan inference. It runs Wan 2.2 A14B's high-noise and low-noise experts as one continuous denoising pass and provides a consolidated SCAIL-2 character-control pipeline.

Designed for Wan 2.2 I2V/T2V workflows, LightX2V four-step distillation, and full-driving-video SCAIL-2 generation.

## Features

- One continuous Wan 2.2 high/low-noise sampler
- Exact expert routing: I2V `0.900`, T2V `0.875`
- One initial noise sample and continuous latent handoff
- Built-in LightX2V four-step profile: Euler/simple, CFG 1, shift 5
- Optional animated latent preview across both Wan experts
- Full-batch SCAIL media preparation
- Canvas-assisted SAM3 identity tracking and trained-palette masks
- Automatic SCAIL 81-frame windows with 5-frame overlap
- Pack-native SCAIL conditioning, sampling, decoding, stitching, and Reinhard stabilization
- Validation for model shifts, schedules, temporal shape, masks, and resolution
- Native and GGUF loader compatibility
- ComfyUI retains control of model loading, LoRAs, caching, and VRAM policy

## Installation

1. Copy or clone this directory into `ComfyUI/custom_nodes/ComfyUI-Wan22`.
2. Restart ComfyUI.
3. Add the Wan or SCAIL nodes from their normal ComfyUI categories.

The extension uses ComfyUI's V3 API. The SCAIL example also uses Video Helper Suite and KJNodes. KJNodes **Preview Animation** is used for lightweight mask/video previews.

## Nodes

| Node | Purpose |
| --- | --- |
| **Wan 2.2 Combined KSampler** | Continuous high-to-low Wan 2.2 denoising |
| **Wan 2.2 Live Preview** | Animated latent preview across both experts |
| **Wan SCAIL-2 Media Prep** | Resizes complete reference/video batches to one geometry |
| **Wan SCAIL-2 Identity Control** | SAM3 canvas tracking and both SCAIL identity masks |
| **Wan SCAIL-2 Auto-Extend Sampler** | Full-video SCAIL conditioning, generation, stitching, and stabilization |

## Quick start: LightX2V 4-step I2V

1. Load separate matching Wan 2.2 high- and low-noise experts.
2. Apply the matching high/low LightX2V LoRAs unless using pre-distilled models.
3. Apply `ModelSamplingSD3` with shift **5** to both experts.
4. Connect both models, conditioning, and the starting latent to **Wan 2.2 Combined KSampler**.
5. Select `I2V` and `LightX2V 4-step distilled`.
6. Send the output latent to `VAEDecode`.

Example:

`example_workflows/Wan22_Combined_I2V_LightX2V_4step.json`

## SCAIL-2 quick start

Load:

`example_workflows/Wan_SCAIL2_Fresh_Character_Control.json`

The workflow replaces the old resize, identity-tracker, colored-mask, conditioning, sampler, and color-transfer chain with three pack-owned nodes:

1. **Media Prep** preserves the full driving-video batch at a multiple-of-32 resolution.
2. **Identity Control** tracks matching reference/driving identities and renders both colored masks.
3. **Auto-Extend Sampler** generates the complete valid `4n+1` timeline in bounded windows.

Run **Identity Control** first to load its reference and driving canvases. Inspect both KJNodes animated mask previews before queuing the sampler.

### Prompting and identity order

Wan positive/negative prompts control the generated video. SAM3 prompts only identify what to segment and track.

- Use a short SAM3 prompt such as `person` on both reference and driving sides.
- For one character, keep `object_indices` at `0` and `sort_by` at `none`.
- For multiple characters, place matching markers in the same order on both canvases.

SCAIL's trained identity order is blue, red, green, magenta, cyan, then yellow. Text prompts find subjects; marker/order assignment determines which reference maps to which driving identity.

### Reference edges and RMBG

For replacement, keep `replacement_mode` enabled. The reference-mask preview should show a blue subject on black.

If a white halo appears:

1. Add positive points on details to keep.
2. Add negative points just outside the subject on the unwanted fringe—not on wanted hair, feathers, or body pixels.
3. Increase `refine_iterations` from `2` to `3` and inspect the mask again.

Adjust the driving canvas only when its mask includes unwanted background or other subjects. The two sides need the same identity count/order, not identical point placement.

RMBG is optional. Use it before **Media Prep** only when white/background pixels remain inside the selected reference boundary. Composite onto black, keep feathering minimal, and avoid eroding fine hair or feathers. If RMBG damages appearance details, retain the original image for CLIP Vision.

### Proven SCAIL defaults

| Setting | Value |
| --- | --- |
| Resolution / frame rate | 640 × 960 / 16 fps |
| Steps / CFG | 6 / 1.0 |
| Sampler / scheduler | Euler / simple |
| `ModelSamplingSD3` shift | 5 |
| Chunk / overlap | 81 / 5 frames |
| Seed mode | Fixed |
| Color stabilization | Reinhard LAB, strength 1 |
| Replacement mode | On |

Use tiled decode if normal VAE decode exceeds available VRAM. Completed chunks are moved to CPU; ComfyUI remains responsible for model residency and offloading.

## Wan 2.2 sampling behavior

Wan 2.2 switches experts by noise level, not by splitting the step count in half. The combined sampler creates one compatible sigma schedule, runs the high-noise expert through the task boundary, and continues with the low-noise expert from the same latent without adding noise again.

For LightX2V:

| Setting | Value |
| --- | --- |
| Steps | 4 |
| CFG | 1.0 |
| Shift | 5 |
| Sampler / scheduler | Euler / simple |

Use matching high/low adapters from the same release. Do not cross them or apply them again to already-distilled models.

## Preview, models, and memory

**Wan 2.2 Live Preview** is intended for motion and composition feedback, not final colour or detail judgement. Preview work is bounded and asynchronous.

The samplers accept standard ComfyUI `MODEL` inputs. They do not load files, clone heavyweight models, force cache cleanup, or control offloading.

On 24 GB GPUs, a high-to-low expert swap is normally expected because both 14B experts may not fit simultaneously.

## Diagnostics and testing

Diagnostic outputs report resolved settings, schedules, chunk plans, handoffs, trimming, and stage timings.

Run:

```powershell
python -m unittest discover -s tests -v
```

## Notes

- SCAIL output is trimmed to the nearest valid `4n+1` frame count.
- Width and height must be divisible by 32.
- The known LightX2V `..._1022` unused-key warnings are expected only for that adapter pair; investigate other missing-key warnings.
- Settings outside a distilled LoRA's training schedule are experimental.

## License

Licensed under the [Apache License 2.0](LICENSE).
