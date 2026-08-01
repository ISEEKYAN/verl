# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

# Avoid importing ``verl.utils.modelopt.__init__``: this CPU contract test does
# not need the optional ModelOpt package used by the exporter.
modelopt_package = ModuleType("verl.utils.modelopt")
modelopt_package.__path__ = [str(Path(__file__).parents[3] / "verl" / "utils" / "modelopt")]
previous_modelopt_package = sys.modules.get("verl.utils.modelopt")
sys.modules["verl.utils.modelopt"] = modelopt_package

from verl.utils.modelopt.vllm_modelopt_patch import (  # noqa: E402
    prepare_modelopt_nvfp4_weight_stream,
)

if previous_modelopt_package is None:
    sys.modules.pop("verl.utils.modelopt", None)
else:
    sys.modules["verl.utils.modelopt"] = previous_modelopt_package


class _DenseQKVModel:
    def __init__(self):
        self.qkv = SimpleNamespace(
            # vLLM 0.23 QKVParallelLinear.output_sizes: q, k, v output axes.
            output_sizes=[4096, 512, 512],
            output_size_per_partition=5120,
            input_size_per_partition=2048,
            _modelopt_group_size=16,
            weight=torch.nn.Parameter(torch.empty(5120, 1024, dtype=torch.uint8), requires_grad=False),
            weight_scale=torch.nn.Parameter(torch.empty(5120, 128, dtype=torch.float8_e4m3fn), requires_grad=False),
        )

    def named_modules(self):
        yield "", self
        yield "model.layers.0.self_attn.qkv_proj", self.qkv


class _DenseRowParallelModel:
    def __init__(self):
        checkpoint_weight = torch.empty(1024, 2048, dtype=torch.uint8)
        self.o_proj = SimpleNamespace(
            # vLLM RowParallelLinear declares global K=input_size and N=output_size.
            input_size=4096,
            output_size=1024,
            input_size_per_partition=4096,
            output_size_per_partition=1024,
            tp_size=1,
            _modelopt_group_size=16,
            weight=torch.nn.Parameter(torch.empty(256, 2048, dtype=torch.int32), requires_grad=False),
            weight_scale=torch.nn.Parameter(torch.empty(256, 1024), requires_grad=False),
            _hf_param_meta={
                "weight": {
                    "shape": tuple(checkpoint_weight.shape),
                    "dtype": checkpoint_weight.dtype,
                    "device": "cpu",
                    "param_class": torch.nn.Parameter,
                    "input_dim": 1,
                    "output_dim": 0,
                },
                "weight_scale": {
                    "shape": (1024, 256),
                    "dtype": torch.float8_e4m3fn,
                    "device": "cpu",
                    "param_class": torch.nn.Parameter,
                    "input_dim": 1,
                    "output_dim": 0,
                },
                "weight_scale_2": {
                    "shape": (),
                    "dtype": torch.float32,
                    "device": "cpu",
                    "param_class": torch.nn.Parameter,
                },
                "input_scale": {
                    "shape": (),
                    "dtype": torch.float32,
                    "device": "cpu",
                    "param_class": torch.nn.Parameter,
                },
            },
            _marlin_tensor_refs={"weight": torch.empty(256, 2048, dtype=torch.int32)},
            _weight_loaders={},
        )

    def named_modules(self):
        yield "", self
        yield "model.layers.0.self_attn.o_proj", self.o_proj


class _CurrentCompressedDenseRowParallelModel(_DenseRowParallelModel):
    def __init__(self):
        super().__init__()
        self.o_proj._hf_param_meta = {
            "weight_packed": {
                "shape": (1024, 2048),
                "dtype": torch.uint8,
                "device": "cpu",
                "param_class": torch.nn.Parameter,
                "input_dim": 1,
                "output_dim": 0,
            },
            "weight_scale": {
                "shape": (1024, 256),
                "dtype": torch.float8_e4m3fn,
                "device": "cpu",
                "param_class": torch.nn.Parameter,
                "input_dim": 1,
                "output_dim": 0,
            },
            "weight_global_scale": {
                "shape": (),
                "dtype": torch.float32,
                "device": "cpu",
                "param_class": torch.nn.Parameter,
            },
            "input_global_scale": {
                "shape": (),
                "dtype": torch.float32,
                "device": "cpu",
                "param_class": torch.nn.Parameter,
            },
        }


def test_row_parallel_loader_restores_checkpoint_layout_and_public_axis_contract():
    model = _DenseRowParallelModel()
    source = torch.arange(1024, dtype=torch.int32).view(1024, 1).expand(1024, 2048).to(torch.uint8)

    prepared = list(
        prepare_modelopt_nvfp4_weight_stream(
            model,
            [("model.layers.0.self_attn.o_proj.weight", source)],
        )
    )

    name, loaded = prepared[0]
    assert name == "model.layers.0.self_attn.o_proj.weight"
    assert model.o_proj.weight.shape == (1024, 2048)
    assert model.o_proj.weight.input_dim == 1
    assert model.o_proj.weight.output_dim == 0
    # Exercise the real vLLM RowParallel loader contract: input_dim selects the K axis.
    param_data = model.o_proj.weight.data
    shard = loaded.narrow(model.o_proj.weight.input_dim, 0, param_data.shape[model.o_proj.weight.input_dim])
    assert shard.shape == param_data.shape
    param_data.copy_(shard)


def test_row_parallel_loader_rejects_wrong_row_column_direction():
    with pytest.raises(ValueError, match=r"dense NVFP4 source axis mismatch.*o_proj.weight.*2048.*1024.*1024.*2048"):
        list(
            prepare_modelopt_nvfp4_weight_stream(
                _DenseRowParallelModel(),
                [("model.layers.0.self_attn.o_proj.weight", torch.empty(2048, 1024, dtype=torch.uint8))],
            )
        )


def test_row_parallel_loader_rejects_missing_checkpoint_restore_contract():
    model = _DenseRowParallelModel()
    del model.o_proj._hf_param_meta

    with pytest.raises(ValueError, match=r"checkpoint metadata.*o_proj.weight"):
        list(
            prepare_modelopt_nvfp4_weight_stream(
                model,
                [("model.layers.0.self_attn.o_proj.weight", torch.empty(1024, 2048, dtype=torch.uint8))],
            )
        )


def test_unregistered_dense_path_rejects_missing_group_size_contract():
    model = _DenseRowParallelModel()
    del model.o_proj._modelopt_group_size

    def named_modules():
        yield "", model
        yield "model.layers.0.self_attn.unregistered_proj", model.o_proj

    model.named_modules = named_modules
    with pytest.raises(ValueError, match=r"group_size contract is missing.*unregistered_proj.weight_scale"):
        list(
            prepare_modelopt_nvfp4_weight_stream(
                model,
                [
                    (
                        "model.layers.0.self_attn.unregistered_proj.weight_scale",
                        torch.empty(1024, 256, dtype=torch.float8_e4m3fn),
                    )
                ],
            )
        )


@pytest.mark.parametrize(
    ("source_suffix", "target_suffix", "tensor"),
    [
        ("weight_scale", "weight_scale", torch.empty(1024, 256, dtype=torch.float8_e4m3fn)),
        ("weight_global_scale", "weight_scale_2", torch.tensor(1.0)),
        ("input_global_scale", "input_scale", torch.tensor(2.0)),
    ],
)
def test_row_parallel_loader_mirrors_every_quantized_name_surface(source_suffix, target_suffix, tensor):
    prepared = list(
        prepare_modelopt_nvfp4_weight_stream(
            _DenseRowParallelModel(),
            [(f"model.layers.0.self_attn.o_proj.{source_suffix}", tensor)],
        )
    )

    assert prepared[0][0] == f"model.layers.0.self_attn.o_proj.{target_suffix}"


@pytest.mark.parametrize(
    ("suffix", "tensor"),
    [
        ("weight_packed", torch.empty(1024, 2048, dtype=torch.uint8)),
        ("weight_scale", torch.empty(1024, 256, dtype=torch.float8_e4m3fn)),
        ("weight_global_scale", torch.tensor(1.0)),
        ("input_global_scale", torch.tensor(2.0)),
    ],
)
def test_current_compressed_dense_stream_preserves_declared_parameter_names(suffix, tensor):
    prepared = list(
        prepare_modelopt_nvfp4_weight_stream(
            _CurrentCompressedDenseRowParallelModel(),
            [(f"model.layers.0.self_attn.o_proj.{suffix}", tensor)],
        )
    )

    assert prepared[0][0] == f"model.layers.0.self_attn.o_proj.{suffix}"


@pytest.mark.parametrize(
    ("projection", "output_size"),
    [
        ("q_proj", 4096),
        ("k_proj", 512),
        ("v_proj", 512),
    ],
)
def test_dense_qkv_stream_preserves_projection_names_for_native_vllm_mapping(projection, output_size):
    updates = [
        (f"model.layers.0.self_attn.{projection}.weight_packed", torch.empty(output_size, 1024, dtype=torch.uint8)),
        (
            f"model.layers.0.self_attn.{projection}.weight_scale",
            torch.empty(output_size, 128, dtype=torch.float8_e4m3fn),
        ),
        (f"model.layers.0.self_attn.{projection}.weight_global_scale", torch.tensor(1.0)),
    ]

    prepared = list(prepare_modelopt_nvfp4_weight_stream(_DenseQKVModel(), updates))

    assert [name for name, _ in prepared] == [name for name, _ in updates]
    assert [tensor is source for (_, tensor), (_, source) in zip(prepared, updates, strict=True)] == [True, True, True]


def test_dense_qkv_stream_rejects_prefused_checkpoint_name():
    with pytest.raises(ValueError, match=r"pre-fused.*qkv_proj.weight_packed.*q_proj/k_proj/v_proj"):
        list(
            prepare_modelopt_nvfp4_weight_stream(
                _DenseQKVModel(),
                [("model.layers.0.self_attn.qkv_proj.weight_packed", torch.empty(5120, 1024, dtype=torch.uint8))],
            )
        )
