# Nemotron alignment: native Megatron/mlite

Training uses the native adapter in
[Megatron-LM #226](https://github.com/ISEEKYAN/Megatron-LM/pull/226).
Inference uses [vLLM #25](https://github.com/ISEEKYAN/vllm/pull/25), based on
Baiyan's `ds4-v9-rc1` with the required Mamba BI dependencies.

This verl change supplies only the integration:

- Optional byte-exact response logprob gate: `VERL_REQUIRE_BITWISE_LOGPROBS=1`.
- Distinct reproducible trajectory/session seeds under full determinism.
- Centralized multi-node vLLM DP launch without a spurious head-node start rank.
- `check_inference.py` for separate batch and prefill/decode logprob checks.

The previous Nemotron-specific FSDP engine, HF training monkeypatch adapter and
FSDP reproduction scripts have been removed. Generic verl FSDP support is
unchanged. The single-rank HF comparison oracle now belongs only to mlite tests.

## Native two-step configuration

Use `experimental/lite/examples/verl/scripts/run_nemotron_alignment.sh` from
the companion Megatron checkout on a two-node Ray cluster (four GB200 GPUs/node).
Set MODEL_PATH, TRAIN_FILES, OUTPUT_ROOT and TRAIN_BATCH_SIZE=32, and pass:

```bash
actor_rollout_ref.actor.use_dynamic_bsz=False \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
```

Training: TP1/PP2/CP2/EP4/denseDP2, one sequence per micro-batch, no offload.
Rollout: DP8/EP8, n2, CUDA Graph enabled, prefix caching disabled.
Prompt/response caps: 2048/8192; these are limits, not actual prompt lengths.
Both sides require `nemotron_shared_norms=True`.

The original two-step result had bitwise-equal response logprobs but zero
advantages/gradients. It is not evidence of effective learning or of the newly
migrated/cleaned code. Fresh validation and image provenance are recorded in the
linked PRs; the old FSDP Docker image is not the native reproduction image.
