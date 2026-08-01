# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dense Q/K/V checkpoint-name recognition for vLLM 0.23 contracts."""

import re

_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
_FUSED_PROJECTION = "qkv_proj"
NVFP4_PACKED_WEIGHT_SUFFIX = "weight_packed"
NVFP4_GROUP_SCALE_SUFFIX = "weight_scale"
NVFP4_GLOBAL_SCALE_SUFFIX = "weight_global_scale"
NVFP4_INPUT_SCALE_SUFFIX = "input_global_scale"

# Canonical stream names are the compressed-tensors checkpoint names accepted
# by vLLM's native loader. The second entries are explicit legacy ModelOpt
# aliases used only when that schema is present in saved metadata.
NVFP4_DENSE_STREAM_SUFFIX_TO_PARAM_CANDIDATES = {
    "weight": ("weight",),
    NVFP4_PACKED_WEIGHT_SUFFIX: (NVFP4_PACKED_WEIGHT_SUFFIX,),
    NVFP4_GROUP_SCALE_SUFFIX: (NVFP4_GROUP_SCALE_SUFFIX,),
    NVFP4_GLOBAL_SCALE_SUFFIX: (NVFP4_GLOBAL_SCALE_SUFFIX, "weight_scale_2"),
    NVFP4_INPUT_SCALE_SUFFIX: (NVFP4_INPUT_SCALE_SUFFIX, "input_scale"),
}

_TENSOR_SUFFIXES = (
    "weight",
    NVFP4_PACKED_WEIGHT_SUFFIX,
    NVFP4_GROUP_SCALE_SUFFIX,
    NVFP4_GLOBAL_SCALE_SUFFIX,
    NVFP4_INPUT_SCALE_SUFFIX,
)
_NAME_PATTERN = re.compile(
    rf"^(?P<prefix>.*\.)(?P<projection>{'|'.join((*_PROJECTIONS, _FUSED_PROJECTION))})"
    rf"(?P<base_layer>\.base_layer)?\.(?P<suffix>{'|'.join(_TENSOR_SUFFIXES)})$"
)


def _match_dense_qkv_name(name: str):
    return _NAME_PATTERN.fullmatch(name)
