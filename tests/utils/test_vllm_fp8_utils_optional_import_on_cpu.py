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

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl/utils/vllm/vllm_fp8_utils.py"


def _module(name: str) -> types.ModuleType:
    return types.ModuleType(name)


def _load_without_fused_moe():
    vllm = _module("vllm")
    vllm.__version__ = "0.26.1"
    fused_moe_layer = _module("vllm.model_executor.layers.fused_moe.layer")
    linear = _module("vllm.model_executor.layers.linear")
    linear.LinearBase = torch.nn.Module
    fp8_kernel = _module("verl.utils.kernel.fp8_kernel")
    fp8_kernel.scaled_fp8_blockwise = lambda *args, **kwargs: (None, None)
    fakes = {
        "vllm": vllm,
        "vllm.model_executor": _module("vllm.model_executor"),
        "vllm.model_executor.layers": _module("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": _module("vllm.model_executor.layers.fused_moe"),
        "vllm.model_executor.layers.fused_moe.layer": fused_moe_layer,
        "vllm.model_executor.layers.linear": linear,
        "verl.utils.kernel": _module("verl.utils.kernel"),
        "verl.utils.kernel.fp8_kernel": fp8_kernel,
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location("verl_vllm_fp8_utils_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def test_missing_fusedmoe_keeps_non_fp8_import_available():
    module = _load_without_fused_moe()

    assert module.FusedMoE is None


def test_missing_fusedmoe_fails_loud_on_fp8_path():
    module = _load_without_fused_moe()

    with pytest.raises(RuntimeError, match="current vLLM version does not provide FusedMoE"):
        list(module.quant_weights([], model=None, quant_config=None))
