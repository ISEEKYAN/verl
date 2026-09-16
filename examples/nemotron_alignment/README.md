# Nemotron-H alignment check

Experimental BF16 aligned forward with native backward. Requires the matching
vLLM Mamba exact-replay/graph stack and shared-normalization/EP changes; stock
vLLM alone is not sufficient. No native rebuild is needed with the tested image.

The verified target is52 layers, inference TP1/DP8/EP8 on two4-GPU nodes,
training TP1/EP4/CP2/PP2. CUDA Graph remains enabled. Inputs grow separately
from outputs, through2048 input and8192 output tokens.

Set explicit absolute paths; existing inference results are never overwritten:

```bash
export NEMOTRON_MODEL=/models/aligned-bf16
export NEMOTRON_RESULT=/results/inference.json
export VLLM_BATCH_INVARIANT=1 NEMOTRON_SHARED_NORMS=1
export NEMOTRON_EP_SLOT_DIAGNOSTIC=1
export NEMOTRON_DP=8 NEMOTRON_DP_LOCAL=4
export NEMOTRON_DP_MASTER=<node-zero-ip> NEMOTRON_DP_PORT=29931
export NEMOTRON_NODE_RANK=<zero-or-one>
export NEMOTRON_PROMPT_LENGTHS=2045,2046,2047,2048
export NEMOTRON_RESPONSE_LENGTH=8192 NEMOTRON_MAX_MODEL_LEN=12288
export NEMOTRON_TOKEN_BUDGET=128 NEMOTRON_TEMPERATURE=1
# Execute once per node, exposing four GPUs and the required RDMA/IMEX devices.
.venv/bin/python examples/nemotron_alignment/check_inference.py
```

After inference exits successfully on both nodes, score its saved tokens:

```bash
export NEMOTRON_REFERENCE=/results/inference-rank0.json
export NEMOTRON_EP=4 NEMOTRON_CP=2 NEMOTRON_PP=2
.venv/bin/python -m torch.distributed.run --nnodes=2 --nproc_per_node=4 \
  --node_rank="$NEMOTRON_NODE_RANK" --master_addr="$NEMOTRON_DP_MASTER" \
  --master_port=29933 examples/nemotron_alignment/check_training.py
```

Four seeded requests are compared across batch1/4 and decode/full-prefill.
Trainer gates include single, packed, reversed, repeated batch32 and dynamic32.
This is not128 unique prompts/n8, arbitrary unequal EP token counts, or an RL
convergence benchmark. The EP transport prioritizes exact reduction order,
not production throughput.

`NEMOTRON_BACKWARD_SMOKE=1` adds a short packed17/19 backward check, not a
long-backward claim. `NEMOTRON_EXPORT_UPDATE=1` additionally performs one SGD
step and requires a fresh `NEMOTRON_UPDATE_PATH`; full weights need about59GiB.
Exports stage locally before copying to the destination. Initial weights are
never modified. Optimizer/RNG checkpoint resume is not implemented here.
