Megatron native API refactor preview
===================================

Last updated: 10/29/2026.

This fork preview upgrades the Megatron integration to a fixed Core 0.21 / Bridge
0.8 development snapshot. The dependency manifest and lockfile identify the exact
revisions, including the companion Core and Bridge changes. GPU acceptance is a
separate gate from CPU interface and gradient tests.

Architecture
------------

Megatron-Bridge remains responsible for model construction, HF conversion, PEFT
hooks, and weight mapping. Core owns model execution, routing, recomputation and
distributed optimizer behavior. The verl adapter owns token layouts, RL masks,
log-probability/entropy output processors, rollout routing traces, and the
training/rollout/checkpoint lifecycle.

Both GPTModel and HybridModel expose the same output processor contract. The
model family selected by Bridge determines the architecture. Hybrid families can
use the native path without an assumption that one HF block equals one native
decoder layer. Router IDs come from Bridge's HF conversion metadata.

Changes in this preview
-----------------------

* Remove the MCore 0.12 MLA and checkpoint-writer compatibility implementations,
  old Bridge helper fallbacks, the legacy fused GPT forward, and the full MTP and
  TopKRouter replacements.
* Use one THD/BSHD layout adapter for eager and fused outputs. Output processors
  receive immutable per-forward context. Direct fused output-weight reads call
  Core's parameter readiness API before accessing tied or untied weights.
* Control MTP through native ``compute_mtp_loss``,
  ``mtp_loss_normalization_factor``, ``mtp_detach_heads`` and
  ``mtp_detach_output_weight``. Output-weight detachment is independent of the
  backbone/embedding detachment requested by ``model.mtp.detach_encoder``.
* Keep response-aware MTP masks and packed sequence boundaries in the input
  adapter. Dynamic CP uses the current microbatch's process group in native MTP.
* Use Core RouterReplay with a per-token replay mask for R3. Scope actions and
  recomputation queues to the current model and clear them on exceptions.
  THD and BSHD route layouts follow the model's actual token order.
* Gather PP/VPP routes using actual router IDs rather than a second copy of
  Megatron's pipeline-layer assignment rules.
* Use native recomputation and upstream pipeline offsets. Checkpoint operations
  materialize optimizer state before loading and restore phase placement even
  when the operation raises.

The remaining runtime shims address optional FlashAttention/CUDA-extension
imports and the ROCm Hadamard dependency. They are isolated in ``runtime.py`` and
do not replace Core model, router or autograd methods.

Dependency and platform boundary
-------------------------------

Use the branch's generated ``uv.lock`` for CUDA Linux x86_64/aarch64. Core,
Bridge, and Transformer Engine are pinned together. Transformer Engine is built from its fixed source revision against runtime Torch
and CUDA. The sync host therefore needs the CUDA toolkit and the build tools
listed in the Megatron extra. The October 29 refresh preserves upstream Transformers 5.16.1, vLLM 0.32.0,
DeepEP 1.2.1 and quantization-aware THD alignment. Combined vLLM/SGLang runs
need inference acceptance on this baseline.
Older images and custom providers need the same native contracts; model initialization reports
missing interfaces before training. Ascend and ROCm acceptance require their
own environment validation against this baseline.

Existing checkpoint layout v2 and HF export continue through the established
checkpoint manager. GPT-to-Hybrid checkpoint conversion follows the upstream
migration guide and the model family's Bridge mappings. Complete optimizer/RNG
resume across a model-architecture change requires independent validation.

Validation and release gates
----------------------------

CPU coverage checks callback values and gradients, parameter readiness, per-token
replay and microbatch queue order, MTP output-weight gradients and normalization,
Bridge layer mapping, padded token order, and exception cleanup. The focused
Core tests are runnable without distributed test fixtures; their normal imports
still exercise the real Core package.

The October 22 baseline on Slurm used the generated ``cpu`` extra to install and import Torch
2.13.0+cpu, Core 0.20.0 and Transformers 5.15.0. The focused verl suites pass
80 tests with one GPU-only skip; the companion Core suites pass 15 tests in
that same CPU environment. Transformer Engine and Apex are absent from this
CPU environment. These results do not validate the locked CUDA runtime.

The October 29 refresh passes 50 focused Core tests (one CUDA skip) and
29 focused verl tests. The latter also pass after installing the generated CPU
lock with Torch 2.13.0+cpu, Core 0.21.0 and Transformers 5.16.1. On H100,
8 router replay tests pass, including bit-exact CPU/CUDA route, score and
gradient checks. That GPU run uses the existing Torch 2.11 / TE 2.15 image;
it does not validate the new locked CUDA environment.

Before promoting this preview, run the following GPU acceptance matrix:

* Dense and MoE SFT plus RL actor/reference forwards, on TP/DP and PP/VPP meshes.
* THD and BSHD, static CP and dynamic CP, R2/R3, MTP enabled and disabled.
* Tied and untied output weights with overlapped parameter gathering.
* BF16 first, followed by FP8/MXFP8 and the configured ModelOpt QAT modes.
* Save/resume the optimizer, scheduler and RNG; compare the next update with an
  uninterrupted run. Include optimizer CPU offload and rollout phase offload.
* Check steady-state memory after removing the old backward replacement, then
  compare update time, peak memory and weight synchronization time.

Related work
------------

The baseline includes the merged cleanup series #8051, #8058, #8094 and #8172,
and the output-processor integration #6933. The preview changes the shared native
boundary used by open work #8029 (readiness), #7826 (BSHD fused MTP), #8031 (MTP
head gradients), #8074 (EP overlap), #7192 (Hybrid MTP) and #8235 (resume/offload).
Those changes should be coordinated before proposing an upstream implementation.

Track progress in `Modern Megatron integration RFC #8324
<https://github.com/verl-project/verl/issues/8324>`_. Current Core already includes
the Hybrid output callback (#7054); the refresh reuses that implementation. The
MoE recompute storage work in #8338 makes memory parity an explicit gate before
removing the legacy implementation. CPU interface coverage cannot establish that
memory gate.

An upstream tracking issue can use this document's architecture, implementation
list and remaining acceptance matrix. The fork PR is the review surface for the
prototype; public upstream submission follows human review and validation.
