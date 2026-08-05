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

"""Runtime compatibility patch for Kimi-K3 MLA with disabled DCP.

Remove this when vLLM's ``MLACommonImpl`` no longer resets the disabled-DCP
state to the legacy ``-1`` sentinel.
"""

import inspect
from functools import wraps

from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl

_EXPECTED_INIT_PARAMETERS = (
    "self",
    "num_heads",
    "head_size",
    "scale",
    "num_kv_heads",
    "alibi_slopes",
    "sliding_window",
    "kv_cache_dtype",
    "logits_soft_cap",
    "attn_type",
    "kv_sharing_target_layer_name",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "qk_head_dim",
    "v_head_dim",
    "kv_b_proj",
    "indexer",
    "q_pad_num_heads",
)
_PATCH_MARKER = "_verl_kimi_k3_disabled_dcp_patch"


def _wrap_mla_common_init(original_init):
    actual_parameters = tuple(inspect.signature(original_init).parameters)
    if actual_parameters != _EXPECTED_INIT_PARAMETERS:
        raise RuntimeError(
            "vLLM MLACommonImpl constructor signature changed; remove or update "
            f"the verl Kimi-K3 MLA patch (expected {_EXPECTED_INIT_PARAMETERS}, got {actual_parameters})."
        )

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "dcp_world_size"):
            raise RuntimeError(
                "vLLM MLACommonImpl no longer defines dcp_world_size; remove or update the verl Kimi-K3 MLA patch."
            )
        if self.dcp_world_size != -1:
            raise RuntimeError(
                "vLLM MLACommonImpl disabled-DCP sentinel changed; remove or update the verl Kimi-K3 MLA patch "
                f"(expected -1, got {self.dcp_world_size!r})."
            )

        self.dcp_world_size = 1
        self.dcp_rank = 0
        if (self.dcp_world_size, self.dcp_rank) != (1, 0):
            raise RuntimeError("Failed to install the verl Kimi-K3 disabled-DCP state on MLACommonImpl.")

    setattr(patched_init, _PATCH_MARKER, True)
    return patched_init


def apply_kimi_k3_mla_patch() -> None:
    """Install the Kimi-K3 disabled-DCP fix before a vLLM engine is created."""
    if not isinstance(MLACommonImpl, type):
        raise RuntimeError("vLLM MLACommonImpl is not a class; cannot install the verl Kimi-K3 MLA patch.")

    original_init = getattr(MLACommonImpl, "__init__", None)
    if not callable(original_init):
        raise RuntimeError("vLLM MLACommonImpl.__init__ is missing or not callable.")
    if getattr(original_init, _PATCH_MARKER, False):
        return

    MLACommonImpl.__init__ = _wrap_mla_common_init(original_init)
