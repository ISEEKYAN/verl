# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _install_exporter_dependencies(monkeypatch):
    quant_utils = types.ModuleType("modelopt.torch.export.quant_utils")
    quant_utils.QUANTIZATION_NONE = "none"
    quant_utils.QUANTIZATION_NVFP4 = "nvfp4"
    quant_utils.QUANTIZATION_MXFP4 = "mxfp4"
    quant_utils.get_quantization_format = lambda module: "none"
    quant_utils.get_weight_block_size = lambda module: 0
    quant_utils.to_quantized_weight = lambda *args, **kwargs: None

    class FakeNVFP4QTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            return (torch.ones(weight.shape[-1] // block_size),)

    class FakeMXFP4QTensor:
        @classmethod
        def quantize(cls, weight, block_size):
            scale = torch.arange(weight.numel() // block_size, dtype=torch.uint8).reshape(-1, 1)
            return SimpleNamespace(_quantized_data=torch.empty(0, dtype=torch.uint8)), scale

    modules = {
        "modelopt": types.ModuleType("modelopt"),
        "modelopt.torch": types.ModuleType("modelopt.torch"),
        "modelopt.torch.export": types.ModuleType("modelopt.torch.export"),
        "modelopt.torch.export.quant_utils": quant_utils,
        "modelopt.torch.quantization": types.ModuleType("modelopt.torch.quantization"),
        "modelopt.torch.quantization.qtensor": types.ModuleType("modelopt.torch.quantization.qtensor"),
        "modelopt.torch.quantization.qtensor.nvfp4_tensor": types.ModuleType(
            "modelopt.torch.quantization.qtensor.nvfp4_tensor"
        ),
        "modelopt.torch.quantization.qtensor.mxfp4_tensor": types.ModuleType(
            "modelopt.torch.quantization.qtensor.mxfp4_tensor"
        ),
    }
    modules["modelopt.torch.quantization.qtensor.nvfp4_tensor"].NVFP4QTensor = FakeNVFP4QTensor
    modules["modelopt.torch.quantization.qtensor.mxfp4_tensor"].MXFP4QTensor = FakeMXFP4QTensor

    megatron_utils = types.ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model
    modules["verl.utils.megatron_utils"] = megatron_utils
    modelopt_package = types.ModuleType("verl.utils.modelopt")
    modelopt_package.__path__ = [str(Path(__file__).parents[3] / "verl" / "utils" / "modelopt")]
    modules["verl.utils.modelopt"] = modelopt_package

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("verl.utils.modelopt.qat_weight_exporter", None)
    return importlib.import_module("verl.utils.modelopt.qat_weight_exporter")


def test_mxfp4_export_packs_last_dimension_and_reshapes_block_scales(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    weight = torch.arange(2 * 96, dtype=torch.float32).reshape(2, 96)
    expected_packed = torch.arange(2 * 48, dtype=torch.uint8).reshape(2, 48)
    calls = []

    def fake_to_quantized_weight(weight_arg, scale, qformat, scale_2=None, block_size=None):
        calls.append((weight_arg, scale.clone(), qformat, scale_2, block_size))
        return expected_packed

    monkeypatch.setattr(exporter_module, "to_quantized_weight", fake_to_quantized_weight)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="mxfp4", block_size=32, weight_amax=None)

    result = list(exporter._quantize_mxfp4("model.layers.0.mlp.experts.0.gate_proj.weight", weight, meta))

    assert [name for name, _ in result] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
    ]
    assert torch.equal(result[0][1], expected_packed)
    assert result[1][1].shape == (2, 3)
    assert result[1][1].dtype == torch.uint8
    assert calls[0][2:] == ("mxfp4", None, 32)
    assert torch.equal(calls[0][1], result[1][1])


def test_mxfp4_export_rejects_non_ocp_block_size(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="mxfp4", block_size=16, weight_amax=None)

    with pytest.raises(ValueError, match="block size 32"):
        list(exporter._quantize_mxfp4("model.layers.0.mlp.experts.0.down_proj.weight", torch.ones(2, 32), meta))


def test_process_weights_iterator_dispatches_mxfp4(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="mxfp4", block_size=32, weight_amax=None)
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda name: meta)
    monkeypatch.setattr(
        exporter,
        "_quantize_mxfp4",
        lambda name, weight, metadata: iter([(name + ".mxfp4", weight)]),
        raising=False,
    )

    result = list(
        exporter.process_weights_iterator(iter([("model.layers.0.mlp.experts.0.up_proj.weight", torch.ones(1))]))
    )

    assert result[0][0].endswith(".mxfp4")


def test_export_only_mode_synthesizes_mxfp4_metadata_and_honors_ignores(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._metadata = {}
    exporter._registry = SimpleNamespace(_reverse_patterns=[])
    exporter._use_modelopt_fake_quant = False
    exporter.qat_mode = "mxfp4"
    exporter._block_size = 32
    exporter._ignore_patterns = ["lm_head", "embed_tokens", "re:.*mlp\\.gate$"]

    meta = exporter._resolve_quant_metadata("model.layers.0.mlp.experts.0.gate_proj.weight")

    assert meta.qformat == "mxfp4"
    assert meta.block_size == 32
    assert exporter._resolve_quant_metadata("model.layers.0.mlp.gate.weight") is None
    assert exporter._resolve_quant_metadata("model.embed_tokens.weight") is None


def test_export_only_mode_does_not_require_a_megatron_bridge(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._metadata = {}
    exporter._registry = exporter._get_mapping_registry(None)
    exporter._use_modelopt_fake_quant = False
    exporter.qat_mode = "w4a16"
    exporter._block_size = 16
    exporter._ignore_patterns = []

    meta = exporter._resolve_quant_metadata("model.layers.0.self_attn.q_proj.weight")

    assert exporter._registry is None
    assert meta.qformat == "nvfp4"


def test_hf_stream_exporter_does_not_require_megatron_parallel_state(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    config = SimpleNamespace(
        mode="w4a16",
        group_size=16,
        ignore_patterns=["lm_head"],
        apply_modelopt_fake_quant=False,
    )

    exporter = exporter_module.QATWeightExporter.from_hf_stream(config)

    assert exporter._actor_module == []
    assert exporter._metadata == {}
    assert exporter._registry is None
    assert exporter._pp_size == 1
    assert exporter._ep_size == 1
    assert exporter._resolve_quant_metadata("model.layers.0.self_attn.q_proj.weight").qformat == "nvfp4"
    assert exporter._resolve_quant_metadata("lm_head.weight") is None


def test_hf_stream_nvfp4_emits_checkpoint_parameter_not_marlin_compute_parameter(monkeypatch):
    """The compressed-tensors reload surface is ``weight_packed``, never ``weight``."""
    exporter_module = _install_exporter_dependencies(monkeypatch)

    class DenseQTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            del weights_scaling_factor_2
            return (torch.ones(weight.shape[0], weight.shape[1] // block_size, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", DenseQTensor)
    monkeypatch.setattr(
        exporter_module,
        "to_quantized_weight",
        lambda weight, *_args: torch.empty(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8),
    )
    exporter = exporter_module.QATWeightExporter.from_hf_stream(
        SimpleNamespace(
            mode="w4a16",
            group_size=16,
            ignore_patterns=[],
            apply_modelopt_fake_quant=False,
        ),
        model_config=SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=32,
            head_dim=128,
        ),
    )

    result = dict(
        exporter.process_weights_iterator(iter([("model.layers.0.self_attn.o_proj.weight", torch.ones(2048, 4096))]))
    )

    assert "model.layers.0.self_attn.o_proj.weight" not in result
    assert result["model.layers.0.self_attn.o_proj.weight_packed"].shape == (2048, 2048)
    assert result["model.layers.0.self_attn.o_proj.weight_scale"].shape == (2048, 256)


def test_export_helper_uses_hf_stream_path_for_non_modelopt_backend(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    qat_utils = importlib.import_module("verl.utils.modelopt.qat_utils")
    config = SimpleNamespace(
        mode="mxfp4",
        group_size=32,
        ignore_patterns=[],
        apply_modelopt_fake_quant=False,
    )
    sentinel = object()
    captured = {}

    class FakeExporter:
        @staticmethod
        def _get_model_config(modules):
            captured["modules"] = modules
            return "model-config"

        @classmethod
        def from_hf_stream(cls, qat_config, model_config=None):
            captured["qat_config"] = qat_config
            captured["model_config"] = model_config
            return cls()

        def process_weights_iterator(self, weights):
            captured["weights"] = weights
            return sentinel

    monkeypatch.setattr(exporter_module, "QATWeightExporter", FakeExporter)
    source = iter(())
    modules = [object()]

    result = qat_utils.export_qat_weights(source, modules, config, bridge=None)

    assert result is sentinel
    assert captured == {
        "modules": modules,
        "qat_config": config,
        "model_config": "model-config",
        "weights": source,
    }


def test_nvfp4_export_only_mode_computes_current_weight_amax(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    calls = []

    def fake_to_quantized_weight(weight, scale, qformat, scale_2=None, block_size=None):
        calls.append((scale_2.clone(), qformat, block_size))
        return torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8)

    monkeypatch.setattr(exporter_module, "to_quantized_weight", fake_to_quantized_weight)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    weight = torch.tensor([[1.0, -7.0] * 8])
    meta = exporter_module._QuantMeta(
        qformat="nvfp4",
        block_size=16,
        weight_amax=None,
        input_amax=torch.tensor(3.0),
    )

    result = list(
        exporter._quantize_nvfp4(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            weight,
            meta,
        )
    )

    assert calls[0][0].item() == pytest.approx(7.0 / (6.0 * 448.0))
    assert calls[0][1:] == ("nvfp4", 16)
    assert [name for name, _ in result] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.0.mlp.experts.0.gate_proj.weight_global_scale",
        "model.layers.0.mlp.experts.0.gate_proj.input_global_scale",
    ]


def test_fused_moe_w2_layout_converts_modelopt_packed_to_vllm_packed(monkeypatch):
    """The 48 axis is packed intermediate, not an expert or gate/up axis."""
    exporter_module = _install_exporter_dependencies(monkeypatch)
    packed = torch.arange(48 * 2048, dtype=torch.uint8).reshape(48, 2048)

    converted = exporter_module._NVFP4_PACKED_LAYOUT.to_vllm_packed(
        "model.layers.0.mlp.experts.0.down_proj.weight",
        torch.empty(96, 2048),
        packed,
        torch.empty(2048, 6, dtype=torch.float8_e4m3fn),
        group_size=16,
    )

    assert converted.shape == (2048, 48)
    assert converted.dtype == torch.uint8
    assert torch.equal(converted, packed.t())


def test_fused_moe_w2_layout_rejects_non_packed_contract(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)

    with pytest.raises(ValueError, match=r"dense NVFP4.*logical=.*packed=.*scale="):
        exporter_module._NVFP4_PACKED_LAYOUT.to_vllm_packed(
            "model.layers.0.mlp.experts.0.down_proj.weight",
            torch.empty(96, 2048),
            torch.empty(47, 2048, dtype=torch.uint8),
            torch.empty(2048, 6, dtype=torch.float8_e4m3fn),
            group_size=16,
        )


def test_nvfp4_export_uses_group_scale_contract_for_fused_moe_w2(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    packed = torch.arange(48 * 2048, dtype=torch.uint8).reshape(48, 2048)

    class FusedW2QTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            return (torch.ones(2048, 6, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", FusedW2QTensor)
    monkeypatch.setattr(exporter_module, "to_quantized_weight", lambda *args: packed)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=torch.tensor(1.0))

    result = list(exporter._quantize_nvfp4("expert.down_proj.weight", torch.ones(96, 2048), meta))

    assert result[0][1].shape == (2048, 48)
    assert torch.equal(result[0][1], packed.t())


def test_nvfp4_dense_export_rejects_row_column_axis_swap(monkeypatch):
    """A normal dense [N,K] tensor must stay [N,K/2], never use the W2 transpose escape hatch."""
    exporter_module = _install_exporter_dependencies(monkeypatch)

    class DenseQTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            del weights_scaling_factor_2
            return (torch.ones(weight.shape[0], weight.shape[1] // block_size, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", DenseQTensor)
    monkeypatch.setattr(
        exporter_module,
        "to_quantized_weight",
        lambda *_args, **_kwargs: torch.empty(2048, 1024, dtype=torch.uint8),
    )
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)

    with pytest.raises(ValueError, match=r"dense NVFP4.*logical=.*1024.*4096.*packed=.*2048.*1024"):
        list(exporter._quantize_nvfp4("model.layers.0.self_attn.o_proj.weight", torch.ones(1024, 4096), meta))


def test_projection_contract_rejects_hidden_size_as_attention_output_width(monkeypatch):
    """o_proj K is num_attention_heads * head_dim, even when it differs from hidden_size."""
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._config = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=4,
        intermediate_size=6144,
        moe_intermediate_size=96,
        vocab_size=32000,
    )

    with pytest.raises(
        ValueError,
        match=r"o_proj\.weight.*num_attention_heads \* head_dim.*expected=\(2048, 4096\).*actual=\(2048, 2048\)",
    ):
        exporter._validate_logical_weight_shape(
            "model.layers.0.self_attn.o_proj.weight",
            torch.empty(2048, 2048),
        )

    exporter._validate_logical_weight_shape(
        "model.layers.0.self_attn.o_proj.weight",
        torch.empty(2048, 4096),
    )


def test_projection_contract_reads_megatron_axis_field_names(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=3,
        kv_channels=4,
        num_query_groups=1,
        ffn_hidden_size=20,
        moe_ffn_hidden_size=6,
    )

    exporter._validate_logical_weight_shape(
        "model.layers.0.self_attn.o_proj.weight",
        torch.empty(8, 12),
    )
    with pytest.raises(ValueError, match=r"num_attention_heads \* head_dim.*expected=\(8, 12\)"):
        exporter._validate_logical_weight_shape(
            "model.layers.0.self_attn.o_proj.weight",
            torch.empty(8, 8),
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.self_attn.q_proj.weight", (12, 8)),
        ("model.layers.0.self_attn.k_proj.weight", (4, 8)),
        ("model.layers.0.self_attn.v_proj.weight", (4, 8)),
        ("model.layers.0.mlp.gate_proj.weight", (20, 8)),
        ("model.layers.0.mlp.up_proj.weight", (20, 8)),
        ("model.layers.0.mlp.down_proj.weight", (8, 20)),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", (6, 8)),
        ("model.layers.0.mlp.experts.0.up_proj.weight", (6, 8)),
        ("model.layers.0.mlp.experts.0.down_proj.weight", (8, 6)),
        ("model.layers.0.mlp.shared_expert.gate_proj.weight", (10, 8)),
        ("model.layers.0.mlp.shared_expert.up_proj.weight", (10, 8)),
        ("model.layers.0.mlp.shared_expert.down_proj.weight", (8, 10)),
        ("model.embed_tokens.weight", (30, 8)),
        ("lm_head.weight", (30, 8)),
    ],
)
def test_projection_contract_uses_each_configured_semantic_axis(monkeypatch, name, expected):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=3,
        head_dim=4,
        num_key_value_heads=1,
        intermediate_size=20,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=10,
        vocab_size=30,
    )

    exporter._validate_logical_weight_shape(name, torch.empty(expected))
    wrong = (expected[1], expected[0])
    if wrong == expected:
        wrong = (expected[0] - 1, expected[1])
    with pytest.raises(ValueError, match=r"semantic axis mismatch"):
        exporter._validate_logical_weight_shape(name, torch.empty(wrong))


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ],
)
def test_nvfp4_export_validates_every_ordinary_dense_weight_family(monkeypatch, name):
    """All non-MoE dense families use the same explicit [N,K/2] checkpoint contract."""
    exporter_module = _install_exporter_dependencies(monkeypatch)

    class DenseQTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            del weights_scaling_factor_2
            return (torch.ones(weight.shape[0], weight.shape[1] // block_size, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", DenseQTensor)
    monkeypatch.setattr(
        exporter_module,
        "to_quantized_weight",
        lambda weight, *_args: torch.empty(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8),
    )
    exporter = object.__new__(exporter_module.QATWeightExporter)
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)

    result = dict(exporter._quantize_nvfp4(name, torch.ones(12, 32), meta))

    assert result[name].shape == (12, 16)
    assert result[name.replace(".weight", ".weight_scale")].shape == (12, 2)


def test_nvfp4_process_iterator_preserves_projection_names_for_vllm_shard_loading(monkeypatch):
    """vLLM maps q/k/v checkpoint names into qkv_proj with an explicit shard_id."""
    exporter_module = _install_exporter_dependencies(monkeypatch)

    class DenseQKVQTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            del weights_scaling_factor_2
            return (torch.ones(weight.shape[0], weight.shape[1] // block_size, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", DenseQKVQTensor)
    monkeypatch.setattr(
        exporter_module,
        "to_quantized_weight",
        lambda weight, *_args: torch.empty(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8),
    )
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._use_compressed_tensors_weight_names = True
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda _name: meta)

    result = list(
        exporter.process_weights_iterator(
            iter(
                (
                    (f"model.layers.0.self_attn.{projection}.weight", torch.ones(output_size, 16))
                    for projection, output_size in (("q_proj", 8), ("k_proj", 2), ("v_proj", 2))
                )
            )
        )
    )

    assert [name for name, _ in result] == [
        "model.layers.0.self_attn.q_proj.weight_packed",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.q_proj.weight_global_scale",
        "model.layers.0.self_attn.k_proj.weight_packed",
        "model.layers.0.self_attn.k_proj.weight_scale",
        "model.layers.0.self_attn.k_proj.weight_global_scale",
        "model.layers.0.self_attn.v_proj.weight_packed",
        "model.layers.0.self_attn.v_proj.weight_scale",
        "model.layers.0.self_attn.v_proj.weight_global_scale",
    ]
    assert [tensor.shape for _, tensor in result] == [
        (8, 8),
        (8, 1),
        (),
        (2, 8),
        (2, 1),
        (),
        (2, 8),
        (2, 1),
        (),
    ]
    assert not any(".qkv_proj." in name for name, _ in result)


@pytest.mark.parametrize(
    ("weights", "scale_names"),
    [
        (
            (("q_proj", 1.0), ("k_proj", 2.0), ("v_proj", 4.0)),
            ("q_proj", "k_proj", "v_proj"),
        ),
        (
            (("gate_proj", 3.0), ("up_proj", 6.0)),
            ("gate_proj", "up_proj"),
        ),
    ],
)
def test_nvfp4_stream_export_shares_global_scale_for_fused_loader_owners(monkeypatch, weights, scale_names):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    observed_global_scales = []

    class DenseQTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            observed_global_scales.append(weights_scaling_factor_2.clone())
            return (torch.ones(weight.shape[0], weight.shape[1] // block_size, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", DenseQTensor)
    monkeypatch.setattr(
        exporter_module,
        "to_quantized_weight",
        lambda weight, *_args: torch.empty(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8),
    )
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._use_compressed_tensors_weight_names = True
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda _name: meta)
    parent = "model.layers.0.self_attn" if scale_names[0] == "q_proj" else "model.layers.0.mlp.experts.0"

    result = list(
        exporter.process_weights_iterator(
            iter((f"{parent}.{projection}.weight", torch.full((2, 16), value)) for projection, value in weights)
        )
    )

    global_scales = {name: tensor for name, tensor in result if name.endswith(".weight_global_scale")}
    expected_internal_scale = torch.tensor(max(value for _, value in weights) / (6.0 * 448.0))
    expected_checkpoint_divisor = 1.0 / expected_internal_scale
    assert list(global_scales) == [f"{parent}.{projection}.weight_global_scale" for projection in scale_names]
    assert all(torch.equal(scale, expected_checkpoint_divisor) for scale in global_scales.values())
    assert len(observed_global_scales) == len(weights)
    assert all(torch.equal(scale, expected_internal_scale) for scale in observed_global_scales)


def test_nvfp4_stream_export_rejects_incomplete_fused_global_scale_group(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._use_compressed_tensors_weight_names = True
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda _name: meta)

    with pytest.raises(ValueError, match=r"incomplete NVFP4 fused global-scale group.*v_proj"):
        list(
            exporter.process_weights_iterator(
                iter(
                    [
                        ("model.layers.0.self_attn.q_proj.weight", torch.ones(2, 16)),
                        ("model.layers.0.self_attn.k_proj.weight", torch.ones(2, 16)),
                    ]
                )
            )
        )


def test_nvfp4_dense_export_artifact_matches_offline_and_online_name_contracts(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)

    class DenseQTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            del weights_scaling_factor_2
            return (torch.ones(weight.shape[0], weight.shape[1] // block_size, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", DenseQTensor)
    monkeypatch.setattr(
        exporter_module,
        "to_quantized_weight",
        lambda weight, *_args: torch.empty(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8),
    )
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._use_compressed_tensors_weight_names = True
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda _name: meta)
    artifact = list(
        exporter.process_weights_iterator(iter([("model.layers.0.self_attn.o_proj.weight", torch.ones(12, 32))]))
    )

    expected_names = [
        "model.layers.0.self_attn.o_proj.weight_packed",
        "model.layers.0.self_attn.o_proj.weight_scale",
        "model.layers.0.self_attn.o_proj.weight_global_scale",
    ]
    # Native/offline loading resolves the canonical checkpoint suffix directly.
    offline_meta = {
        "weight_packed": {"shape": (12, 16), "dtype": torch.uint8},
        "weight_scale": {"shape": (12, 2), "dtype": torch.float8_e4m3fn},
        "weight_global_scale": {"shape": (), "dtype": torch.float32},
    }
    assert [name for name, _ in artifact] == expected_names
    assert [name.rsplit(".", 1)[1] in offline_meta for name, _ in artifact] == [True, True, True]
    artifact_by_name = dict(artifact)
    # compressed-tensors declares weight_global_scale as a divisor (1/scale),
    # unlike ModelOpt's weight_scale_2 multiplier used during quantization.
    assert artifact_by_name[expected_names[-1]].item() == pytest.approx(6.0 * 448.0)

    layer = SimpleNamespace(
        input_size=32,
        output_size=12,
        _modelopt_group_size=16,
        _hf_param_meta={
            name: {
                **contract,
                "device": "cpu",
                "param_class": torch.nn.Parameter,
            }
            for name, contract in offline_meta.items()
        },
    )
    model = SimpleNamespace(
        named_modules=lambda: iter(
            (
                ("", model),
                ("model.layers.0.self_attn.o_proj", layer),
            )
        )
    )
    from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_nvfp4_weight_stream

    online_artifact = list(prepare_modelopt_nvfp4_weight_stream(model, artifact))
    assert [name for name, _ in online_artifact] == expected_names


def test_nvfp4_compressed_tensors_stream_exports_input_scale_as_divisor(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)

    class DenseQTensor:
        @staticmethod
        def get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2):
            del weights_scaling_factor_2
            return (torch.ones(weight.shape[0], weight.shape[1] // block_size, dtype=torch.float8_e4m3fn),)

    monkeypatch.setattr(exporter_module, "NVFP4QTensor", DenseQTensor)
    monkeypatch.setattr(
        exporter_module,
        "to_quantized_weight",
        lambda weight, *_args: torch.empty(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8),
    )
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._use_compressed_tensors_weight_names = True
    meta = exporter_module._QuantMeta(
        qformat="nvfp4",
        block_size=16,
        weight_amax=None,
        input_amax=torch.tensor(3.0),
    )
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda _name: meta)

    artifact = dict(
        exporter.process_weights_iterator(iter([("model.layers.0.self_attn.o_proj.weight", torch.ones(12, 32))]))
    )

    assert artifact["model.layers.0.self_attn.o_proj.input_global_scale"].item() == pytest.approx((6.0 * 448.0) / 3.0)


def test_nvfp4_compressed_tensors_stream_rejects_nonpositive_divisor_source(monkeypatch):
    exporter_module = _install_exporter_dependencies(monkeypatch)
    exporter = object.__new__(exporter_module.QATWeightExporter)
    exporter._use_compressed_tensors_weight_names = True
    meta = exporter_module._QuantMeta(qformat="nvfp4", block_size=16, weight_amax=None)
    monkeypatch.setattr(exporter, "_resolve_quant_metadata", lambda _name: meta)

    with pytest.raises(ValueError, match="requires finite positive scale.*weight_global_scale"):
        list(exporter.process_weights_iterator(iter([("model.layers.0.self_attn.o_proj.weight", torch.zeros(12, 32))])))
