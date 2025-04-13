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

from .config_converter import hf_to_mcore_config_dense, hf_to_mcore_config_qwen2moe, hf_to_mcore_config_dpskv3, hf_to_mcore_config_qwen2_5_vl, hf_to_mcore_config_llama4
from .config_converter import PretrainedConfig, TransformerConfig
import torch
import torch.nn as nn


def hf_to_mcore_config(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    MODEL_CONFIG_CONVERTER_REGISTRY = {
        "LlamaForCausalLM": hf_to_mcore_config_dense,
        "Qwen2ForCausalLM": hf_to_mcore_config_dense,
        "Qwen2MoeForCausalLM": hf_to_mcore_config_qwen2moe,
        "DeepseekV3ForCausalLM": hf_to_mcore_config_dpskv3,
        "Qwen2_5_VLForConditionalGeneration": hf_to_mcore_config_qwen2_5_vl,
        "Llama4ForConditionalGeneration": hf_to_mcore_config_llama4,
    }
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_CONFIG_CONVERTER_REGISTRY:
        raise ValueError(f"Model architectures {arch} converter are not supported for now. "
                         f"Supported architectures: {MODEL_CONFIG_CONVERTER_REGISTRY.keys()}")
    return MODEL_CONFIG_CONVERTER_REGISTRY[arch](hf_config, dtype)


from .model_initializer import init_mcore_model_dense, init_mcore_model_qwen2_moe, init_mcore_model_dpskv3, init_mcore_model_qwen2_5_vl, init_mcore_model_llama4


def init_mcore_model(
        tfconfig,
        hf_config,
        pre_process=None,
        post_process=None,
        share_embeddings_and_output_weights=False,
        value=False,
        **extra_kwargs  # may be used for vlm
) -> nn.Module:
    MODEL_INITIALIZER_REGISTRY = {
        "LlamaForCausalLM": init_mcore_model_dense,
        "Qwen2ForCausalLM": init_mcore_model_dense,
        "Qwen2MoeForCausalLM": init_mcore_model_qwen2_moe,
        "DeepseekV3ForCausalLM": init_mcore_model_dpskv3,
        "Qwen2_5_VLForConditionalGeneration": init_mcore_model_qwen2_5_vl,
        "Llama4ForConditionalGeneration": init_mcore_model_llama4,
    }
    assert len(hf_config.architectures) == 1, "Only one architecture is supported for now"
    arch = hf_config.architectures[0]
    if arch not in MODEL_INITIALIZER_REGISTRY:
        raise ValueError(f"Model architectures {arch} initializer are not supported for now. "
                         f"Supported architectures: {MODEL_INITIALIZER_REGISTRY.keys()}")
    return MODEL_INITIALIZER_REGISTRY[arch](tfconfig, hf_config, pre_process, post_process,
                                            share_embeddings_and_output_weights, value, **extra_kwargs)
