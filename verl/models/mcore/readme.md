# Megatron integration

Updated 2026-10-22.

Megatron-Bridge provides model construction and HF weight conversion. This preview
uses the fixed Core/Bridge/Transformer Engine revisions in `pyproject.toml` and
`uv.lock`; the native API contracts replace the historical version fallbacks.

The shared input adapter supports THD and BSHD. Native model output processors
produce logits or fused PPO statistics; MTP, routing and recomputation execute in
Core. Router layer IDs follow Bridge conversion metadata so HybridModel's expanded
layer indices stay aligned with HF rollout traces.

See [the refactor preview](../../../docs/advance/megatron_native_backend.rst) for
migration requirements, companion API changes, related PRs and acceptance gates.

New model architectures and HF mappings belong in Megatron-Bridge. Custom providers
must expose `output_processor` / `output_processor_context`; MTP providers also
expose `compute_mtp_loss` and `mtp_loss_normalization_factor`.
