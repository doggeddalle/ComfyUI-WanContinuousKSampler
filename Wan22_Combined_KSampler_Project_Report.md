# Wan 2.2 Combined KSampler Project Report

**Reporting date:** 30 July 2026  
**Project version:** 0.2.0  
**Project directory:** `C:\Users\ac\Documents\Codex\wan`  
**Primary target:** Wan 2.2 A14B in ComfyUI, with LightX2V four-step distillation on a 24 GB RTX 3090

## Executive summary

This project rebuilt the Wan 2.2 high-noise/low-noise sampling path as a clean ComfyUI V3 extension. The main result is **Wan 2.2 Combined KSampler**, which owns the full denoising schedule, runs the two Wan experts in the correct order, and carries one latent continuously across their handoff. It replaces duplicated sampler controls and external split arithmetic with a single validated sampling contract.

A companion **Wan 2.2 Live Preview** node was also created. It presents an animated view of the active video latent throughout both expert passes, rather than showing one temporal slice or restarting at the expert boundary. The preview avoids model wrappers and full VAE decoding, uses bounded asynchronous work, and does not change ComfyUI's model-loading policy.

The work also included a detailed audit of the original workflow, a distilled LightX2V profile, interpretation of the supplied LoRA warnings, performance and VRAM analysis, deterministic workflow migration, documentation, and an 18-test validation suite.

The implementation is ready to install and begin fixed-seed video regression testing. It has loaded successfully against the local ComfyUI V3 API and has exercised its preview staging path on the RTX 3090, but it has not yet been installed into the live `custom_nodes` directory or used for a complete end-to-end render.

## Project goals and outcomes

| Goal | Outcome |
| --- | --- |
| Combine high-noise and low-noise generation | Implemented as one sampler node with two `MODEL` inputs and one continuous schedule. |
| Improve performance | Removed project-level forced cleanup, avoided model cloning and internal file loading, retained ComfyUI caching, and bounded preview work. |
| Prioritise distilled LoRAs | Added a first-class LightX2V four-step profile using its published sampler, CFG, shift, and expert pairing. |
| Reduce quality loss and instability | Corrected schedule ownership and expert routing, preserved the handoff sigma, reused one initial noise sample, and added validation against common wiring errors. |
| Feel native to ComfyUI | Built on the V3 extension API, standard ComfyUI types, upstream loaders and patchers, normal cache behaviour, and a separate optional preview controller. |
| Add meaningful Wan-specific capability | Added a duration-correct animated latent preview spanning both Wan experts, plus diagnostics designed for repeatable comparisons. |

## Assessment of the reference workflow

The preserved `testwan.json` workflow was treated as a reproducible description of the existing approach. Several structural issues were identified:

- High-noise and low-noise generation were expressed as two `KSamplerAdvanced` nodes with duplicated controls.
- The split was derived through midpoint arithmetic rather than from Wan's expert-routing boundary.
- The active settings used six Euler/simple steps and shift 8, which did not match the published LightX2V four-step recipe.
- The active path did not include the matching high-noise and low-noise LightX2V LoRAs.
- The high expert used Q4_K_M while the detail-focused low expert used Q4_0, making the two phases an uneven quality pair.
- A workflow-wide GPU cleanup operation appeared between sampling and VAE decode, deliberately discarding residency that ComfyUI could otherwise reuse.
- Native INT8 loaders were present as a disconnected alternative rather than part of the active sampling path.

These observations do not prove that one setting alone caused flicker. Temporal instability can also come from the source image, conditioning, quantisation, LoRA compatibility, VAE, resolution, frame count, or prompt. The new node therefore concentrates on making the sampling path deterministic, inspectable, and easier to compare under controlled conditions.

## Combined sampler implementation

### Node contract

**Wan 2.2 Combined KSampler** accepts:

- distinct high-noise and low-noise `MODEL` inputs;
- positive and negative conditioning;
- one starting video latent;
- a seed, task selection, and sampling profile;
- an optional live-preview configuration.

It returns the completed latent and a diagnostic string. Model loaders, LoRA application, and `ModelSamplingSD3` remain visible upstream, which keeps the node compatible with native, GGUF, and future model loaders.

### Distilled profile

The primary preset is **LightX2V 4-step distilled**:

| Setting | Value |
| --- | ---: |
| Steps | 4 |
| High-noise CFG | 1.0 |
| Low-noise CFG | 1.0 |
| ModelSamplingSD3 shift | 5.0 |
| Sampler | Euler |
| Scheduler | Simple |

CFG 1 also allows ComfyUI to use its native fast path. A 20-step convenience baseline and an explicitly labelled custom mode are available, but the custom controls do not silently alter models or imply compatibility with a distilled adapter.

### Exact expert routing

The full sigma schedule is calculated once for each model and must match. The sampler then routes evaluations using Wan's task boundary:

- I2V: `0.900`
- T2V: `0.875`

Equality remains with the high-noise expert. If the handoff index is `k`, the high expert receives `sigmas[:k+1]` and the low expert receives `sigmas[k:]`. The shared sigma is the destination of the high phase and the starting point of the low phase; it does not duplicate a model evaluation.

Only the first phase receives the prepared seed noise. The second phase continues from the resulting latent with zero additional noise. This preserves one denoising trajectory and prevents the two phases from behaving like separate generations.

### Validation and quality protections

The node rejects or reports:

- the same expert connected to both model inputs;
- two clones of one underlying expert;
- shared or mismatched model-sampling configuration;
- the wrong `ModelSamplingSD3` shift for the selected profile;
- differing, malformed, non-descending, or non-terminal sigma schedules;
- a boundary that leaves either expert with no work;
- malformed latent inputs;
- an invalid preview configuration source.

The diagnostic output records the resolved profile, effective step count, sampler and scheduler, CFG values, high/low evaluation counts, handoff sigma, full sigma list, and wall time around each expert call. Those stage times intentionally include any model transfer ComfyUI performs, making them useful for cold-versus-warm comparisons.

## Model loading and memory policy

The extension does not load weights from disk, apply LoRAs internally, clone heavyweight model patchers, force a global unload, empty the CUDA cache, or call a workflow cleanup node. ComfyUI remains responsible for caching, offloading, and eviction.

This policy matters on a 24 GB card. Two patched 14B experts generally cannot remain fully resident together, so one high-to-low expert transition is expected. A new queued generation may also need to replace the previously resident low expert with the high expert. The project aims to add no extra transitions beyond those required by the architecture.

Keeping loaders and LoRA nodes upstream also means changing only the seed can reuse their cached outputs. It avoids rereading model files or rebuilding patches simply because a sampler control changed.

## Live video preview

### Why a dedicated controller was used

KJNodes Model Preview Override is effective for a conventional single-model sampler, but Wan 2.2 uses two experts. Wrapping only one model ends the preview at the handoff; wrapping both creates two preview lifecycles. The new controller instead connects directly to the combined sampler and follows its unified global step count.

It does not patch, clone, or retain either model. Disconnecting it returns the sampler to ComfyUI's standard preview behaviour.

### Preview pipeline

For each requested update, the implementation:

1. Selects a bounded number of temporal slices from the five-dimensional video latent.
2. Applies Wan's lightweight Latent2RGB projection rather than running a full VAE or TAESD decode.
3. Limits preview dimensions on the device before transfer.
4. Copies the small RGB tensor into pinned CPU memory without blocking the CUDA stream.
5. Records a CUDA event and lets a background worker wait for readiness.
6. Encodes a duration-correct animated WebP and sends it to the frontend widget.

One shared latest-wins worker permits at most one active encode and one pending update. A newer pending frame set replaces stale work, so slow encoding cannot create an unbounded queue or a thread per sampling run. Cancellation tokens suppress late sends from an aborted generation.

Default controls are:

| Control | Default |
| --- | ---: |
| Output FPS | 16 |
| Maximum latent frames | 24 |
| Preview interval | Every denoising step |
| WebP quality | 70 |
| Maximum dimension | 512 px |

Wan uses a four-times temporal ratio. A 93-frame output contains 24 latent slices because `(24 - 1) × 4 + 1 = 93`. Displaying those slices for the same duration as a 93-frame, 16-fps video requires an effective latent rate of `24 × 16 / 93`, or approximately **4.129 fps**. The controller calculates this automatically rather than incorrectly playing the latent at the final output rate.

The browser extension adds an animated preview widget, reports the global high/low stage, sigma, step, latent and output frame counts, and effective frame rates. It supports qualified subgraph node IDs, rejects late events from retired runs, and revokes old Blob URLs.

Preview messages are sent only to the browser client that initiated the prompt. If an API-submitted prompt has no browser client ID, the extension uses progress-only callbacks and sends no generated frame data. This prevents ComfyUI's `sid=None` broadcast behaviour from exposing previews to unrelated connected clients.

The preview is intended for motion, timing, and composition feedback. Latent2RGB is an approximation and should not be used to judge final colour, texture, or fine detail.

## LoRA warning investigation

The attached verbose log contained a known warning pattern for the exact LightX2V `..._1022` I2V pair rather than a fatal loading exception.

| Adapter | Total keys | Unused keys | Compatible entries |
| --- | ---: | ---: | ---: |
| High-noise | 1,500 | 41 `diff_m` entries | 1,459 |
| Low-noise | 1,749 | 290 legacy image-attention entries | 1,459 |

The 41 high-noise warnings cover one `diff_m` modulation delta for each of 40 blocks plus the head. The low-noise warnings refer to the older `k_img`, `v_img`, `norm_k_img`, and `img_emb` image-attention path that is not present in the selected Wan 2.2 base architecture. The matching [LightX2V discussion](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/discussions/6) describes the same compatibility signature.

The later `Requested to load WAN21` line is ComfyUI's shared internal model-class label. It is not evidence that the loader substituted a Wan 2.1 checkpoint for the chosen Wan 2.2 expert.

Only this exact, understood signature should be treated as expected. Other missing-key messages must remain visible because they can reveal a crossed high/low adapter, an incompatible base, or a damaged conversion.

## Performance and VRAM observations

### GGUF baseline

One cold 640×480, 93-frame I2V run completed in **138 seconds** on the RTX 3090. The active Q4 GGUF route held at approximately **16–17 GB total VRAM**. This is a promising result for iterative GGUF use, but it is one measurement on one machine and not a general performance guarantee.

The combined node does not make an individual model forward pass faster. Its contribution is removing redundant sampler structure and avoiding unnecessary model-lifecycle interference.

### INT8 Convrot result

Each local INT8 Convrot expert is approximately 14.5 GB on disk before runtime memory is considered. Activations, attention workspaces, quantisation metadata, LoRA patches, temporary kernels, and any overlap during expert transfer must also fit in 24 GB. The observed execution-limit/OOM stall at 640×480×93 is therefore consistent with the available memory envelope. Cancelling the run was appropriate.

This is not a sampler handoff defect, and the project does not present INT4 as the fallback because its Wan quality loss was already considered unacceptable. Smaller dimensions, fewer frames, or more aggressive offloading could test whether INT8 runs, but that would be a separate memory experiment.

### Preview smoke measurement

A synthetic 24-slice Wan latent was exercised through the real CUDA staging path on the RTX 3090. After CUDA warm-up, staging took about **0.6 ms per update**, the worker-side event wait about **0.03 ms**, and animated WebP encoding about **52 ms** off the sampling thread. This confirms the asynchronous mechanism works, but a full generation with preview enabled and disabled is still required to measure end-to-end overhead.

## Reference workflow migration

The source workflow remains byte-for-byte unchanged:

`testwan.json`  
SHA-256: `F97B1400D5DE6FCBDEDEE38E4DD0773D7C359F05FED9B80EE406CC0B5E4C2B01`

A deterministic generator creates the migrated example:

`example_workflows/Wan22_Combined_I2V_LightX2V_4step.json`  
SHA-256: `D42109B9C8057582286BD820089D7369BAC4AA561DCFBA8987F031B61C2EA472`

The migrated workflow has 22 nodes and 28 links. It adds matching high/low LightX2V LoRA branches, sets both model-sampling shifts to 5, replaces the split sampler path with the combined sampler, connects the live-preview controller, routes diagnostics to a text preview, connects the final latent directly to VAE decode, and removes the forced GPU cleanup from the active path.

The migration script rebuilds link metadata and validates node IDs, socket order, widget order, LoRA filenames, shifts, output types, and all source/target link relationships. It also verifies that generation did not modify the reference file.

## Validation completed

The following checks passed:

- **18 unit tests** using the same embedded Python/PyTorch runtime as ComfyUI.
- Sampling profile resolution and sigma schedule validation.
- Published and reference boundary-routing cases.
- Single-noise, two-expert continuation and metadata preservation.
- Rejection of clone, shift, and schedule mismatches before noise preparation.
- Correct ComfyUI V3 schemas: 15 sampler inputs and two outputs; five preview controls and one custom output.
- Weight-free entrypoint loading with both nodes registered.
- Duration-correct animated WebP generation.
- Device-side frame selection and size limiting.
- Latest-wins queue replacement and shared-worker lifetime.
- Continuous callback numbering across both experts.
- Targeted client events, no-client fail-closed behaviour, and cancellation cleanup.
- Frontend JavaScript syntax validation.
- Real local ComfyUI schema import.
- Real RTX 3090 CUDA preview staging smoke test.
- Deterministic workflow regeneration and preservation of the original workflow hash.
- Static checks confirming the sampler contains no manual global unloading, CUDA synchronisation for timing, internal LoRA file loading, model wrapper, or forced cache cleanup.

Independent focused reviews found no remaining actionable defects in the sampler routing, V3 integration, cancellation, threading, preview privacy, VRAM handling, workflow migration, or frontend resource cleanup.

## Deliberate non-goals and current limitations

- The extension does not automatically identify model files or inspect whether distillation is already merged.
- It cannot repair incompatible experts, crossed or doubled LoRAs, quantisation loss, conditioning problems, or VAE artifacts.
- One expert swap remains normal on a 24 GB GPU.
- The live preview has a small but non-zero projection and transfer cost; `preview_every` and `max_frames` provide explicit control.
- Latent2RGB preview colour and detail are approximate.
- The 20-step profile is a convenience baseline, not a way to reverse an attached four-step distillation LoRA.
- ComfyUI's internal sampling APIs can evolve, so the suite should be rerun after ComfyUI upgrades.
- No fixed-seed A/B video set or end-to-end installed-node render has been completed yet.

## Recommended next phase

1. Install the project as `ComfyUI/custom_nodes/Wan22CombinedKSampler` and restart ComfyUI.
2. Open the migrated example and confirm both V3 nodes and the animated preview widget appear correctly.
3. Run a fixed-seed four-step GGUF render with preview disabled, then repeat with preview enabled to measure real overhead.
4. Compare the new workflow against the original while holding source image, prompt, seed, dimensions, frame count, VAE, LoRA versions, and encoder settings constant.
5. Record separate cold-start and warm-queue timing, including the sampler's high/low wall-time diagnostics and observed expert-transfer points.
6. Build a small visual regression set focused on motion continuity, flicker, identity stability, fine detail, and low-noise quantisation sensitivity.
7. Test a matched GGUF pair, prioritising additional precision for the low-noise/detail expert where VRAM allows.
8. After end-to-end validation, add a compatibility matrix for ComfyUI versions, loaders, model formats, and known distilled releases before packaging for wider distribution.

## Project deliverables

| File | Purpose |
| --- | --- |
| `nodes.py` | Combined sampler, live-preview controller, schedule validation, callbacks, and diagnostics. |
| `web/wan22_live_preview.js` | Animated frontend preview widget and event lifecycle. |
| `__init__.py` | ComfyUI entrypoint and web-directory export. |
| `pyproject.toml` | Project metadata and version 0.2.0. |
| `README.md` | Installation, design, workflow, warning, performance, and testing documentation. |
| `tests/test_nodes.py` | Eighteen-test backend validation suite. |
| `scripts/generate_example_workflow.py` | Deterministic reference-workflow migration and validation. |
| `example_workflows/Wan22_Combined_I2V_LightX2V_4step.json` | Ready-to-open migrated LightX2V I2V example. |
| `testwan.json` | Original reference workflow, preserved unchanged. |

## Conclusion

The project now has a coherent foundation for Wan 2.2 iteration on consumer hardware: a single validated high-to-low sampling path, a distillation-first default, ComfyUI-managed model residency, useful execution diagnostics, and a genuinely video-oriented live preview. The remaining work is empirical rather than architectural—install the extension, produce controlled fixed-seed renders, measure cold and warm behaviour, and tune model precision from observed quality rather than sampler uncertainty.
