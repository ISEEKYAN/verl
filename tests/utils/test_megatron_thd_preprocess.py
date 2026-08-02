# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
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
"""Regression coverage for packed THD preprocessing and reconstruction."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

import verl.utils.device as device_module


def _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size: int = 1, cp_rank: int = 0):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")

    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank
    parallel_state.get_context_parallel_group = lambda: object()
    parallel_state.get_tensor_model_parallel_world_size = lambda: 1

    class PackedSeqParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    packed_seq_params.PackedSeqParams = PackedSeqParams

    core.parallel_state = parallel_state
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)
    monkeypatch.setattr(device_module, "is_npu_available", False)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_thd_regression", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_tensor(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _run_contiguous_pipeline(monkeypatch, input_ids: torch.Tensor, cp_size: int) -> torch.Tensor:
    rank_outputs = []
    rank_zero_util = None
    rank_zero_params = None
    for cp_rank in range(cp_size):
        mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=cp_size, cp_rank=cp_rank)
        local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
            input_ids,
            cp_layout="contiguous",
        )
        rank_outputs.append(local_ids * 10 + local_positions)
        if cp_rank == 0:
            rank_zero_util = mcore_util
            rank_zero_params = packed_seq_params

    def fake_all_gather(outputs, local_output, group):
        del local_output, group
        for destination, source in zip(outputs, rank_outputs, strict=True):
            destination.copy_(source)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    return rank_zero_util.postprocess_thd_engine(
        rank_outputs[0],
        rank_zero_params,
        input_ids,
        batch_size=input_ids.shape[0],
        cp_layout="contiguous",
    )


def test_preprocess_thd_engine_pads_to_minimum_rows(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)
    input_ids = _nested_tensor([torch.arange(100, dtype=torch.long)])

    local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
        input_ids,
        min_local_rows=128,
    )

    assert local_ids.shape == (1, 128)
    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 128]
    assert torch.equal(local_ids[0, :100], torch.arange(100, dtype=torch.long))
    assert torch.equal(local_ids[0, 100:], torch.zeros(28, dtype=torch.long))
    assert torch.equal(local_positions[0, :100], torch.arange(100, dtype=torch.long))
    assert torch.equal(local_positions[0, 100:], torch.zeros(28, dtype=torch.long))


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_thd_engine_contiguous_matches_cp1_reference(monkeypatch, cp_size):
    input_ids = _nested_tensor(
        [
            torch.tensor([10, 11, 12, 13, 14], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
            torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.long),
        ]
    )
    cp1_reference = _run_contiguous_pipeline(monkeypatch, input_ids, cp_size=1)
    candidate = _run_contiguous_pipeline(monkeypatch, input_ids, cp_size=cp_size)

    assert len(candidate.unbind()) == len(cp1_reference.unbind())
    for candidate_row, reference_row in zip(candidate.unbind(), cp1_reference.unbind(), strict=True):
        assert torch.equal(candidate_row, reference_row)


@pytest.mark.parametrize(
    ("cp_rank", "expected_ids", "expected_positions"),
    [
        (0, [10, 11, 12, 13, 14], [0, 1, 2, 3, 4]),
        (1, [0, 20, 21, 22, 0], [0, 0, 1, 2, 0]),
    ],
)
def test_preprocess_thd_engine_contiguous_uses_global_packed_intervals(
    monkeypatch, cp_rank, expected_ids, expected_positions
):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=2, cp_rank=cp_rank)
    input_ids = _nested_tensor(
        [
            torch.tensor([10, 11, 12, 13, 14], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ]
    )

    local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
        input_ids,
        cp_layout="contiguous",
    )

    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 6, 10]
    # A zigzag-only factor of two would incorrectly pad these rows to 8 and 4.
    assert packed_seq_params.cu_seqlens_q_padded.tolist() != [0, 8, 12]
    assert torch.equal(local_ids[0], torch.tensor(expected_ids, dtype=torch.long))
    assert torch.equal(local_positions[0], torch.tensor(expected_positions, dtype=torch.long))


def test_preprocess_thd_engine_rejects_unknown_cp_layout(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=2)
    input_ids = _nested_tensor([torch.tensor([10, 11], dtype=torch.long)])

    with pytest.raises(ValueError, match="Unsupported context parallel layout: interleaved"):
        mcore_util.preprocess_thd_engine(input_ids, cp_layout="interleaved")


def test_preprocess_thd_engine_contiguous_rolls_each_sequence_independently(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=2, cp_rank=0)
    labels = _nested_tensor(
        [
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
            torch.tensor([5, 6, 7, 8], dtype=torch.long),
        ]
    )

    local_labels, _, _ = mcore_util.preprocess_thd_engine(
        labels,
        need_roll=True,
        cp_layout="contiguous",
    )

    assert torch.equal(local_labels[0], torch.tensor([2, 3, 4, 1], dtype=torch.long))


def test_postprocess_thd_engine_contiguous_inverts_global_intervals(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=2, cp_rank=0)
    input_ids = _nested_tensor(
        [
            torch.tensor([10, 11, 12, 13, 14], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ]
    )
    _, packed_seq_params, _ = mcore_util.preprocess_thd_engine(input_ids, cp_layout="contiguous")
    rank_outputs = [
        torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.float32),
        torch.tensor([[105, 106, 107, 108, 109]], dtype=torch.float32),
    ]

    def fake_all_gather(outputs, local_output, group):
        del local_output, group
        for destination, source in zip(outputs, rank_outputs, strict=True):
            destination.copy_(source)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    restored = mcore_util.postprocess_thd_engine(
        rank_outputs[0],
        packed_seq_params,
        input_ids,
        batch_size=2,
        cp_layout="contiguous",
    )

    assert torch.equal(restored[0], torch.tensor([100, 101, 102, 103, 104], dtype=torch.float32))
    assert torch.equal(restored[1], torch.tensor([106, 107, 108], dtype=torch.float32))


def test_postprocess_thd_engine_rejects_unknown_cp_layout(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=2)

    with pytest.raises(ValueError, match="Unsupported context parallel layout: interleaved"):
        mcore_util.postprocess_thd_engine(
            torch.empty(1, 1),
            packed_seq_params=None,
            input_ids=torch.empty(1),
            batch_size=1,
            post_process=False,
            cp_layout="interleaved",
        )
