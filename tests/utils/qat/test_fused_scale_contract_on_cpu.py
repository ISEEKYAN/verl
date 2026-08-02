# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl.utils.qat.fused_scale_contract import fuse_nvfp4_global_scales


def _import_quantizer_with_fake_compressed_tensors(monkeypatch):
    """Load the mode guard without requiring the optional GPU package."""

    compressed_tensors = types.ModuleType("compressed_tensors")
    compressors = types.ModuleType("compressed_tensors.compressors")
    quantized_compressors = types.ModuleType("compressed_tensors.compressors.quantized_compressors")
    fp4_quantized = types.ModuleType("compressed_tensors.compressors.quantized_compressors.fp4_quantized")
    fp4_quantized.NVFP4PackedCompressor = object
    quantization = types.ModuleType("compressed_tensors.quantization")
    quant_args = types.ModuleType("compressed_tensors.quantization.quant_args")
    quant_args.FP4_E2M1_DATA = types.SimpleNamespace(max=1.0)
    quant_args.FP8_E4M3_DATA = types.SimpleNamespace(max=1.0, dtype=torch.float8_e4m3fn)
    quant_args.QuantizationArgs = object
    quant_args.QuantizationStrategy = types.SimpleNamespace(TENSOR_GROUP="tensor_group")
    quant_args.QuantizationType = types.SimpleNamespace(FLOAT="float")
    utils = types.ModuleType("compressed_tensors.quantization.utils")
    helpers = types.ModuleType("compressed_tensors.quantization.utils.helpers")
    helpers.generate_gparam = object

    for module in (
        compressed_tensors,
        compressors,
        quantized_compressors,
        fp4_quantized,
        quantization,
        quant_args,
        utils,
        helpers,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(sys.modules, "verl.utils.qat.quantizer", raising=False)
    return importlib.import_module("verl.utils.qat.quantizer")


def test_fused_nvfp4_scale_policy_is_explicitly_inverse_across_representations():
    # The two stored forms encode the same policy: a smaller reciprocal gparam
    # is the same choice as a larger divisor.
    reciprocal = fuse_nvfp4_global_scales(
        [torch.tensor([0.25]), torch.tensor([0.125])], representation="reciprocal_gparam"
    )
    divisor = fuse_nvfp4_global_scales([torch.tensor([4.0]), torch.tensor([8.0])], representation="divisor")

    torch.testing.assert_close(reciprocal, torch.tensor(0.125))
    torch.testing.assert_close(divisor, torch.tensor(8.0))
    torch.testing.assert_close(reciprocal * divisor, torch.tensor(1.0))


def test_fused_nvfp4_scale_policy_rejects_unknown_representation():
    try:
        fuse_nvfp4_global_scales([torch.tensor([1.0])], representation="ambiguous")
    except ValueError as error:
        assert "representation" in str(error)
    else:
        raise AssertionError("ambiguous NVFP4 scale representation must fail loudly")


def test_qat_quantizer_rejects_mxfp4_instead_of_emitting_nvfp4(monkeypatch):
    quantizer = _import_quantizer_with_fake_compressed_tensors(monkeypatch)

    with pytest.raises(ValueError, match="MXFP4 requires the ModelOpt QATWeightExporter"):
        quantizer.QATQuantizer(mode="mxfp4")
