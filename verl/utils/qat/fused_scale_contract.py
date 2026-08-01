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

"""Projection groups that share one NVFP4 global scale in fused vLLM owners."""

from typing import Optional

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
