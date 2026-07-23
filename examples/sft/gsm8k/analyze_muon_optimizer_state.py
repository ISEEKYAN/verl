#!/usr/bin/env python3
"""Estimate Muon vs AdamW optimizer-state footprint for a Megatron model chunk.

Run inside the proven CW container (needs megatron + emerging_optimizers).
This is a run-harness diagnostic helper — NOT part of the upstream muon PR diff.

Example:
  python examples/sft/gsm8k/analyze_muon_optimizer_state.py \
    --model-path /lustre/.../models/Qwen3-30B-A3B \
    --tp 2 --ep 8
"""
from __future__ import annotations

import argparse
import json


def _param_bytes(numel: int, dtype_bytes: int) -> int:
    return numel * dtype_bytes


def analyze_hf_config(model_path: str, tp: int, ep: int, pp: int = 1) -> dict:
    with open(f"{model_path}/config.json") as f:
        cfg = json.load(f)

    hidden = cfg.get("hidden_size", 0)
    n_layers = cfg.get("num_hidden_layers", 0)
    n_heads = cfg.get("num_attention_heads", 0)
    n_kv = cfg.get("num_key_value_heads", n_heads)
    intermediate = cfg.get("intermediate_size", 0)
    vocab = cfg.get("vocab_size", 0)
    n_experts = cfg.get("num_experts", 0)
    moe_intermediate = cfg.get("moe_intermediate_size", intermediate)

    world = tp * pp * ep
    # Per-rank shard counts (simplified; ignores PP splitting).
    experts_per_rank = max(n_experts // ep, 1) if n_experts else 0

    # QKV fused weight [out, hidden] with GQA
    qkv_out = (n_heads + 2 * n_kv) * (hidden // n_heads)
    qkv_per_rank = qkv_out // tp

    # Attention proj + MLP dense path
    attn_o_per_rank = hidden * hidden // tp
    # MoE expert: gate_up [hidden, 2*moe_intermediate] + down [moe_intermediate, hidden]
    expert_gate_up = hidden * (2 * moe_intermediate)
    expert_down = moe_intermediate * hidden

    matrix_params_per_rank = 0
    scalar_params_per_rank = 0

    for _ in range(n_layers):
        matrix_params_per_rank += qkv_per_rank * hidden  # qkv rows shard on dim0
        matrix_params_per_rank += attn_o_per_rank
        if n_experts:
            matrix_params_per_rank += experts_per_rank * (expert_gate_up + expert_down)
        else:
            matrix_params_per_rank += 2 * hidden * intermediate // tp

    # Embeddings + lm_head (scalar path for muon)
    scalar_params_per_rank += vocab * hidden // tp  # embed shard
    scalar_params_per_rank += vocab * hidden // tp  # lm_head shard
    # Per-layer layernorm scales (~4 * hidden per layer)
    scalar_params_per_rank += n_layers * 4 * hidden

    # Optimizer state bytes (fp32) per element
    adam_matrix = matrix_params_per_rank * 4 * 2  # m + v
    muon_matrix = matrix_params_per_rank * 4 * 1  # momentum only
    adam_scalar = scalar_params_per_rank * 4 * 2
    muon_scalar = scalar_params_per_rank * 4 * 2  # scalar still uses adam

    # NS scratch peak per largest local matrix (blockwise, fp32 intermediates ~3x min(M,N)^2)
    if n_experts:
        local_m = min(hidden, 2 * moe_intermediate)
        local_n = max(hidden, 2 * moe_intermediate)
    else:
        local_m = min(hidden, intermediate // tp)
        local_n = max(hidden, intermediate // tp)
    ns_scratch_per_matrix = 3 * (min(local_m, local_n) ** 2) * 4  # fp32 A,B temporaries (rough)
    # Upper bound if many matrices peak together (unlikely — sequential step)
    ns_scratch_upper = ns_scratch_per_matrix * 4

    return {
        "model_path": model_path,
        "model_type": cfg.get("model_type"),
        "parallel": {"tp": tp, "pp": pp, "ep": ep, "world": world},
        "params_per_rank": {
            "matrix_numel": matrix_params_per_rank,
            "scalar_numel": scalar_params_per_rank,
        },
        "optimizer_state_gb": {
            "adam_matrix_m_v": adam_matrix / 1024**3,
            "muon_matrix_momentum": muon_matrix / 1024**3,
            "adam_scalar_m_v": adam_scalar / 1024**3,
            "muon_scalar_m_v": muon_scalar / 1024**3,
            "adam_total": (adam_matrix + adam_scalar) / 1024**3,
            "muon_total_theoretical": (muon_matrix + muon_scalar) / 1024**3,
            "matrix_savings_gb": (adam_matrix - muon_matrix) / 1024**3,
        },
        "ns_scratch_mb_per_matrix": ns_scratch_per_matrix / 1024**2,
        "notes": [
            "Theoretical optimizer-state only; ignores DDP buffer padding, LayerWise duplicate buffers, "
            "and Megatron store_param_remainders packing for Adam.",
            "At DP=1, layer-wise sharding does not split layers across ranks.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--ep", type=int, default=8)
    p.add_argument("--pp", type=int, default=1)
    args = p.parse_args()
    result = analyze_hf_config(args.model_path, args.tp, args.ep, args.pp)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
