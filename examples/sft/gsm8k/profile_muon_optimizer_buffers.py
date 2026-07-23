#!/usr/bin/env python3
"""Runtime profiler: DDP param/grad buffer sizes + optimizer state bytes (Muon vs AdamW).

Run-harness diagnostic — NOT part of the upstream muon PR diff.
Invoke inside the proven CW container after model+optimizer construction, or from
the memory-profile sbatch post-hook.

Example (inside training, rank 0 only):
    from examples.sft.gsm8k.profile_muon_optimizer_buffers import profile_and_print
    profile_and_print(model_chunks, optimizer)
"""
from __future__ import annotations

import json
from typing import Any


def _tensor_bytes(t) -> int:
    if t is None:
        return 0
    return t.numel() * t.element_size()


def _buffer_report(model_chunks: list[Any]) -> dict:
    """Summarize per-DDP-buffer param/grad footprint."""
    buffers_out = []
    total_param_bytes = 0
    total_grad_bytes = 0
    for chunk_idx, chunk in enumerate(model_chunks):
        if not hasattr(chunk, "buffers"):
            continue
        for buf_idx, buf in enumerate(chunk.buffers):
            params = getattr(buf, "params", None) or getattr(buf, "params_list", [])
            is_lw = False
            if params:
                is_lw = bool(getattr(params[0], "is_managed_by_layer_wise_optimizer", False))
            param_numel = getattr(buf, "numel", None)
            if param_numel is None and params:
                param_numel = sum(p.numel() for p in params)
            grad_numel = param_numel
            param_bytes = param_numel * 2 if param_numel else 0  # bf16 default
            grad_bytes = grad_numel * 4 if grad_numel else 0  # fp32 grad default
            if hasattr(buf, "param_data") and buf.param_data is not None:
                param_bytes = _tensor_bytes(buf.param_data)
            if hasattr(buf, "grad_data") and buf.grad_data is not None:
                grad_bytes = _tensor_bytes(buf.grad_data)
            entry = {
                "chunk": chunk_idx,
                "buffer": buf_idx,
                "layer_wise_managed": is_lw,
                "use_distributed_optimizer": getattr(
                    getattr(buf, "ddp_config", None), "use_distributed_optimizer", None
                ),
                "param_bytes_gb": param_bytes / 1024**3,
                "grad_bytes_gb": grad_bytes / 1024**3,
                "num_params": len(params),
            }
            buffers_out.append(entry)
            total_param_bytes += param_bytes
            total_grad_bytes += grad_bytes
    return {
        "buffers": buffers_out,
        "total_param_buffer_gb": total_param_bytes / 1024**3,
        "total_grad_buffer_gb": total_grad_bytes / 1024**3,
        "layer_wise_buffer_count": sum(1 for b in buffers_out if b["layer_wise_managed"]),
        "distopt_buffer_count": sum(1 for b in buffers_out if not b["layer_wise_managed"]),
    }


def _optimizer_state_report(optimizer: Any) -> dict:
    """Walk ChainedOptimizer / LayerWise / DistOpt and sum optimizer state tensor bytes."""
    reports = []

    def _walk_state(opt, label: str):
        inner = getattr(opt, "optimizer", opt)
        state_bytes = 0
        main_param_bytes = 0
        n_params = 0
        for group in getattr(inner, "param_groups", []):
            for p in group.get("params", []):
                n_params += 1
                main_param_bytes += _tensor_bytes(p)
                st = inner.state.get(p, {})
                for v in st.values():
                    if hasattr(v, "numel"):
                        state_bytes += _tensor_bytes(v)
                mp = getattr(p, "main_param", None)
                if mp is not None and mp is not p:
                    main_param_bytes += _tensor_bytes(mp)
        reports.append(
            {
                "label": label,
                "type": type(opt).__name__,
                "n_params": n_params,
                "optimizer_state_gb": state_bytes / 1024**3,
                "main_param_extra_gb": main_param_bytes / 1024**3,
            }
        )
        for i, child in enumerate(getattr(opt, "chained_optimizers", []) or []):
            _walk_state(child, f"{label}/chained[{i}]")

    _walk_state(optimizer, "root")
    return {
        "sub_optimizers": reports,
        "total_optimizer_state_gb": sum(r["optimizer_state_gb"] for r in reports),
        "total_main_param_gb": sum(r["main_param_extra_gb"] for r in reports),
    }


def profile_optimizer_memory(model_chunks: list[Any], optimizer: Any) -> dict:
    """Return structured buffer + optimizer-state breakdown."""
    return {
        "ddp_buffers": _buffer_report(model_chunks),
        "optimizer": _optimizer_state_report(optimizer),
    }


def profile_and_print(model_chunks: list[Any], optimizer: Any) -> dict:
    report = profile_optimizer_memory(model_chunks, optimizer)
    print("=== muon buffer profile ===")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(
        "Import profile_and_print(model_chunks, optimizer) from training code; "
        "this module does not build a model standalone."
    )
