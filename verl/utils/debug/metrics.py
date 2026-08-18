# Copyright 2025 Individual Contributor: TomQunChaoA
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import socket
from pathlib import Path

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def dump_train_infer_input(data: DataProto, *, step: int) -> None:
    """Persist the exact teacher-forcing input before actor inference.

    Unlike the normal train/infer dump, this hook intentionally runs before
    ``old_log_probs`` exist so a CUDA failure inside actor inference still
    leaves a standalone-replay capsule.
    """
    directory = os.environ.get("VERL_TRAIN_INFER_PRE_FORWARD_DUMP_DIR")
    if not directory or int(os.environ.get("RANK", "0")) != 0:
        return

    def cpu_raw(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu().contiguous()

    batch = {
        key: cpu_raw(value)
        for key, value in data.batch.items()
        if isinstance(value, torch.Tensor)
    }
    payload = {
        "schema_version": 1,
        "step": int(step),
        "responses": batch.get("responses"),
        "response_mask": batch.get("response_mask"),
        "RL.vllm.rollout_log_probs": batch.get("rollout_log_probs"),
        "sample_indices": list(range(len(data))),
        "input_batch": {
            key: batch[key]
            for key in ("prompts", "input_ids", "attention_mask", "position_ids")
            if key in batch
        },
        "batch_meta_info": dict(data.meta_info),
        "provenance": {
            "producer": "verl.utils.debug.metrics.dump_train_infer_input",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "verl_commit": os.environ.get("VERL_COMMIT"),
            "run_stamp": os.environ.get("RUN_STAMP"),
        },
    }
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"step{int(step):05d}.pt"
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def _dump_train_infer_diff(
    *,
    rollout_log_probs: torch.Tensor,
    actor_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    responses: torch.Tensor,
    data: DataProto,
) -> None:
    path = os.environ.get("VERL_TRAIN_INFER_DIFF_DUMP")
    if not path:
        return
    # This comparison runs on the trainer driver. Keep the guard explicit so
    # future worker-side callers cannot race while writing the same artifacts.
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    mode = os.environ.get("VERL_TRAIN_INFER_DIFF_MODE", "full").strip().lower()
    if mode not in {"full", "compact"}:
        raise ValueError(
            "VERL_TRAIN_INFER_DIFF_MODE must be 'full' or 'compact', "
            f"got {mode!r}"
        )
    rollout = rollout_log_probs.detach().float().cpu()
    actor = actor_log_probs.detach().float().cpu()
    mask = response_mask.detach().bool().cpu()
    tokens = responses.detach().cpu()
    token_sample_limit = int(
        os.environ.get("VERL_TRAIN_INFER_TOKEN_SAMPLE_LIMIT", "8")
    )
    if token_sample_limit < 0:
        raise ValueError("VERL_TRAIN_INFER_TOKEN_SAMPLE_LIMIT must be non-negative")
    samples = []
    attention_mask = data.batch.get("attention_mask")
    prompt_width = (
        attention_mask.shape[1] - responses.shape[1]
        if attention_mask is not None
        else None
    )
    for index in range(rollout.shape[0]):
        valid = mask[index]
        rollout_values = rollout[index][valid]
        actor_values = actor[index][valid]
        logprob_diff = (rollout_values - actor_values).abs()
        probability_diff = (rollout_values.exp() - actor_values.exp()).abs()
        sample = {
            "sample_index": index,
            "token_ids": (
                tokens[index][valid].tolist()
                if mode == "full" or index < token_sample_limit
                else []
            ),
            "bitwise_equal_count": int(torch.eq(rollout_values, actor_values).sum()),
            "valid_token_count": int(valid.sum()),
            "prompt_token_count": (
                int(attention_mask[index, :prompt_width].sum().item())
                if prompt_width is not None
                else None
            ),
        }
        if mode == "full":
            sample.update(
                {
                    "rollout_log_probs": rollout_values.tolist(),
                    "actor_log_probs": actor_values.tolist(),
                    "logprob_abs_diff": logprob_diff.tolist(),
                    "probability_abs_diff": probability_diff.tolist(),
                }
            )
        else:
            sample.update(
                {
                    "logprob_abs_diff_max": float(logprob_diff.max().item())
                    if logprob_diff.numel()
                    else 0.0,
                    "logprob_abs_diff_sum": float(logprob_diff.sum().item()),
                    "probability_abs_diff_max": float(probability_diff.max().item())
                    if probability_diff.numel()
                    else 0.0,
                    "all_logprobs_finite": bool(
                        torch.isfinite(rollout_values).all()
                        and torch.isfinite(actor_values).all()
                        and torch.isfinite(logprob_diff).all()
                    ),
                    "token_ids_captured": index < token_sample_limit,
                }
            )
        samples.append(sample)
    record = {
        "schema_version": 2 if mode == "compact" else 1,
        "mode": mode,
        "shape": list(rollout.shape),
        "token_sample_limit": token_sample_limit,
        "samples": samples,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    raw_path = os.environ.get("VERL_TRAIN_INFER_RAW_DUMP")
    # Compact telemetry is the performance-safe default: never create the
    # legacy full-tensor .pt sidecar unless the caller explicitly requests a
    # real path. Full diagnostic mode preserves the historical sidecar.
    if mode == "compact" and not raw_path:
        return
    if not raw_path:
        jsonl_path = Path(path)
        raw_path = str(jsonl_path.with_suffix(".pt"))

    def cpu_raw(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu().contiguous()

    input_batch = {}
    for key in ("prompts", "input_ids", "attention_mask", "position_ids"):
        if key in data.batch:
            input_batch[key] = cpu_raw(data.batch[key])
    provenance = {
        "producer": "verl.utils.debug.metrics.calculate_debug_metrics",
        "sources": {
            "RL.vllm.rollout_log_probs": "rollout_log_probs",
            "RL.mlite.old_log_probs": "old_log_probs",
        },
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "verl_commit": os.environ.get("VERL_COMMIT"),
        "run_stamp": os.environ.get("RUN_STAMP"),
        "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
        "vllm_ds4_decode_kernel": os.environ.get("VLLM_DS4_DECODE_KERNEL"),
        "verl_full_determinism": os.environ.get("VERL_FULL_DETERMINISM"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    payload = {
        "schema_version": 1,
        "RL.vllm.rollout_log_probs": cpu_raw(rollout_log_probs),
        "RL.mlite.old_log_probs": cpu_raw(actor_log_probs),
        "responses": cpu_raw(responses),
        "response_mask": cpu_raw(response_mask),
        "sample_indices": list(range(responses.shape[0])),
        "input_batch": input_batch,
        "batch_meta_info": dict(data.meta_info),
        "provenance": provenance,
    }
    raw_directory = os.path.dirname(raw_path)
    if raw_directory:
        os.makedirs(raw_directory, exist_ok=True)
    temporary_path = f"{raw_path}.tmp.{os.getpid()}"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, raw_path)


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "loss_mask" in data.batch:
        logger.debug("loss mask found, use it to mask log probs")
        log_prob_mask = data.batch["loss_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    _dump_train_infer_diff(
        rollout_log_probs=rollout_old_log_probs,
        actor_log_probs=actor_old_log_probs,
        response_mask=response_mask,
        responses=responses,
        data=data,
    )

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
            "training/rollout_logprob_abs_diff_max": float("nan"),
            "training/rollout_logprob_bitwise_equal_fraction": float("nan"),
        }

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    rollout_log_probs_valid = torch.masked_select(rollout_old_log_probs, response_mask_bool)
    actor_log_probs_valid = torch.masked_select(actor_old_log_probs, response_mask_bool)
    logprob_abs_diff = torch.abs(rollout_log_probs_valid - actor_log_probs_valid)
    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        "training/rollout_logprob_abs_diff_max": torch.max(logprob_abs_diff).detach().item(),
        "training/rollout_logprob_bitwise_equal_fraction": torch.eq(
            rollout_log_probs_valid,
            actor_log_probs_valid,
        )
        .float()
        .mean()
        .detach()
        .item(),
    }
