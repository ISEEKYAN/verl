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

"""Projection groups and shared-scale policy for fused NVFP4 vLLM owners."""

from typing import Optional

import torch

NVFP4_FUSED_GLOBAL_SCALE_GROUPS = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "gate_up": ("gate_proj", "up_proj"),
}


def resolve_nvfp4_fused_global_scale_group(name: str) -> Optional[tuple[str, tuple[str, ...], str]]:
    """Return ``(owner, ordered_members, member)`` for an HF weight name."""
    if not name.endswith(".weight"):
        return None
    module_name = name.removesuffix(".weight")
    if "." not in module_name:
        return None
    parent, projection = module_name.rsplit(".", 1)
    for group_name, members in NVFP4_FUSED_GLOBAL_SCALE_GROUPS.items():
        if projection in members:
            return (f"{parent}:{group_name}", members, projection)
    return None


def fuse_nvfp4_global_scales(scales: list[torch.Tensor], *, representation: str) -> torch.Tensor:
    """Choose the common fused scale without silently mixing scale conventions.

    FSDP stores the compressed-tensors reciprocal ``gparam``; ModelOpt's
    exporter first holds its divisor (``amax / FP4*FP8``).  They represent the
    same quantization policy, so the selected extrema are inverse: minimum for
    reciprocal gparams and maximum for divisors.
    """
    if not scales:
        raise ValueError("cannot fuse an empty NVFP4 global-scale group")
    values = torch.stack([scale.float().reshape(-1).amax() for scale in scales])
    if representation == "reciprocal_gparam":
        return values.amin()
    if representation == "divisor":
        return values.amax()
    raise ValueError(f"unknown NVFP4 global-scale representation: {representation}")
