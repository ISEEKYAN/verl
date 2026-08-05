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

"""CPU tests for the Kimi-K3 disabled-DCP vLLM runtime patch."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "verl/utils/vllm/kimi_k3_mla_patch.py"
_PACKAGE_INIT_PATH = _REPO_ROOT / "verl/utils/vllm/__init__.py"


def _make_mla_common_impl(*, set_sentinel: bool = True):
    class FakeMLACommonImpl:
        def __init__(
            self,
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
            indexer=None,
            q_pad_num_heads=None,
        ):
            if set_sentinel:
                self.dcp_world_size = -1

    return FakeMLACommonImpl


def _load_patch(mla_common_impl):
    mla_attention = types.ModuleType("vllm.model_executor.layers.attention.mla_attention")
    mla_attention.MLACommonImpl = mla_common_impl
    fakes = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.attention": types.ModuleType("vllm.model_executor.layers.attention"),
        "vllm.model_executor.layers.attention.mla_attention": mla_attention,
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location("verl_kimi_k3_mla_patch_under_test", _MODULE_PATH)
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


def _construct(impl_cls):
    return impl_cls(
        128,
        576,
        1.0,
        1,
        None,
        None,
        "auto",
        None,
        "decoder",
        None,
        1536,
        512,
        128,
        64,
        192,
        128,
        object(),
    )


def test_patch_changes_disabled_dcp_sentinel_to_single_rank_state():
    impl_cls = _make_mla_common_impl()
    unpatched = _construct(impl_cls)

    with pytest.raises(AssertionError, match="must be positive"):
        assert unpatched.dcp_world_size > 0, "cp_world_size must be positive"

    module = _load_patch(impl_cls)
    module.apply_kimi_k3_mla_patch()
    patched = _construct(impl_cls)

    assert patched.dcp_world_size == 1
    assert patched.dcp_rank == 0


def test_importing_verl_vllm_package_applies_patch_before_engine_creation():
    impl_cls = _make_mla_common_impl()
    package_name = "verl_vllm_package_under_test"
    mla_attention = types.ModuleType("vllm.model_executor.layers.attention.mla_attention")
    mla_attention.MLACommonImpl = impl_cls
    npu_patch = types.ModuleType(f"{package_name}.npu_vllm_patch")
    npu_patch.apply_npu_vllm_patches = lambda: None
    utils = types.ModuleType(f"{package_name}.utils")
    utils.TensorLoRARequest = object
    utils.VLLMHijack = object
    utils.is_version_ge = lambda *args, **kwargs: True
    fakes = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.attention": types.ModuleType("vllm.model_executor.layers.attention"),
        "vllm.model_executor.layers.attention.mla_attention": mla_attention,
        f"{package_name}.npu_vllm_patch": npu_patch,
        f"{package_name}.utils": utils,
    }
    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(
            package_name,
            _PACKAGE_INIT_PATH,
            submodule_search_locations=[str(_PACKAGE_INIT_PATH.parent)],
        )
        package = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        patched = _construct(impl_cls)
    finally:
        sys.modules.pop(package_name, None)
        sys.modules.pop(f"{package_name}.kimi_k3_mla_patch", None)
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    assert patched.dcp_world_size == 1
    assert patched.dcp_rank == 0


def test_patch_fails_loudly_when_vllm_constructor_contract_changes():
    class ChangedMLACommonImpl:
        def __init__(self, incompatible_argument):
            self.dcp_world_size = -1

    module = _load_patch(ChangedMLACommonImpl)

    with pytest.raises(RuntimeError, match="constructor signature changed"):
        module.apply_kimi_k3_mla_patch()


def test_patch_fails_loudly_when_vllm_dcp_sentinel_disappears():
    impl_cls = _make_mla_common_impl(set_sentinel=False)
    module = _load_patch(impl_cls)
    module.apply_kimi_k3_mla_patch()

    with pytest.raises(RuntimeError, match="dcp_world_size"):
        _construct(impl_cls)
