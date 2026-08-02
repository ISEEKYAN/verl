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

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch

_SHARED_THD_OPTION_NAMES = {"cp_layout", "local_cp_size", "min_local_rows", "use_fp8_padding"}


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_router_replay_utils(monkeypatch, observed_layouts):
    def preprocess(value, *, cp_layout="zigzag", **kwargs):
        observed_layouts.append(("preprocess", cp_layout, kwargs.get("min_local_rows")))
        rows = sum(row.shape[0] for row in value.unbind())
        return torch.zeros(1, rows, 1, 1), object(), None

    def postprocess(value, *args, cp_layout="zigzag", **kwargs):
        del args, kwargs
        observed_layouts.append(("postprocess", cp_layout))
        return value

    parallel_state = _module("megatron.core.parallel_state")
    schedules = _module("megatron.core.pipeline_parallel.schedules", get_schedule_table=lambda *args: [])
    pipeline_utils = _module(
        "megatron.core.pipeline_parallel.utils",
        is_vp_first_stage=lambda *args: True,
        is_vp_last_stage=lambda *args: True,
    )
    tensor_parallel = _module(
        "megatron.core.tensor_parallel",
        gather_from_sequence_parallel_region=lambda value, **kwargs: value,
        scatter_to_sequence_parallel_region=lambda value: value,
    )
    transformer_config = _module("megatron.core.transformer.transformer_config", TransformerConfig=object)
    transformer_layer = _module(
        "megatron.core.transformer.transformer_layer", get_transformer_layer_offset=lambda *args, **kwargs: 0
    )
    thd_preprocess = _module(
        "verl.models.mcore.thd_preprocess",
        build_thd_preprocess_options=lambda config, *, cp_layout, local_cp_size=None: {
            "use_fp8_padding": config.fp8 in ["e4m3", "hybrid"],
            "local_cp_size": local_cp_size,
            "min_local_rows": (
                config.csa_window_size
                if getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid"
                else None
            ),
            "cp_layout": cp_layout,
        },
    )
    mcore_util = _module(
        "verl.models.mcore.util",
        postprocess_packed_seqs=lambda *args, **kwargs: args[0],
        postprocess_thd_engine=postprocess,
        preprocess_packed_seqs=lambda value, *args, **kwargs: (value, object()),
        preprocess_thd_engine=preprocess,
    )
    router_patch = _module(
        "verl.utils.megatron.router_replay_patch",
        RouterReplay=types.SimpleNamespace(router_instances=[]),
        RouterReplayAction=object,
    )

    modules = {
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core", parallel_state=parallel_state),
        "megatron.core.parallel_state": parallel_state,
        "megatron.core.pipeline_parallel": _module("megatron.core.pipeline_parallel"),
        "megatron.core.pipeline_parallel.schedules": schedules,
        "megatron.core.pipeline_parallel.utils": pipeline_utils,
        "megatron.core.tensor_parallel": tensor_parallel,
        "megatron.core.transformer": _module("megatron.core.transformer"),
        "megatron.core.transformer.transformer_config": transformer_config,
        "megatron.core.transformer.transformer_layer": transformer_layer,
        "verl.models.mcore.thd_preprocess": thd_preprocess,
        "verl.models.mcore.util": mcore_util,
        "verl.utils.megatron.router_replay_patch": router_patch,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    import verl.utils.device as device_module

    monkeypatch.setattr(device_module, "get_device_name", lambda: "cpu")
    path = Path(__file__).parents[2] / "verl" / "utils" / "megatron" / "router_replay_utils.py"
    spec = importlib.util.spec_from_file_location("router_replay_cp_layout_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Router:
    recorded_topk_idx = torch.zeros(5, 1, dtype=torch.long)

    def set_target_indices(self, value, replay_mask=None):
        self.value = value
        self.replay_mask = replay_mask


def _nested_routes():
    return torch.nested.as_nested_tensor(
        [torch.zeros(3, 1, 1, dtype=torch.long), torch.zeros(2, 1, 1, dtype=torch.long)], layout=torch.jagged
    )


def test_dsv4_thd_preprocess_options_are_built_as_one_bundle(monkeypatch):
    util = _module("verl.models.mcore.util", ContextParallelLayout=str)
    monkeypatch.setitem(sys.modules, "verl.models.mcore.util", util)
    path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "thd_preprocess.py"
    spec = importlib.util.spec_from_file_location("verl.models.mcore.thd_preprocess_options_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = types.SimpleNamespace(fp8="hybrid", csa_window_size=128, experimental_attention_variant="dsv4_hybrid")

    assert module.build_thd_preprocess_options(config, cp_layout="contiguous", local_cp_size=4) == {
        "use_fp8_padding": True,
        "local_cp_size": 4,
        "min_local_rows": 128,
        "cp_layout": "contiguous",
    }


def test_dsv4_thd_router_replay_preserves_contiguous_cp_layout(monkeypatch):
    observed_layouts = []
    replay_utils = _load_router_replay_utils(monkeypatch, observed_layouts)
    router = _Router()
    config = types.SimpleNamespace(
        fp8=None,
        csa_window_size=1,
        context_parallel_size=2,
        experimental_attention_variant="dsv4_hybrid",
        num_layers=1,
    )
    monkeypatch.setattr(replay_utils.RouterReplayHelper, "get_micro_batch_router_list", lambda *args: [router])
    monkeypatch.setattr(replay_utils, "get_current_rank_layer_info", lambda *args: {"start": 0, "end": 1})
    monkeypatch.setattr(replay_utils, "is_moe_layer", lambda *args: True)

    replay_utils.set_router_replay_data(
        _nested_routes(),
        None,
        config,
        replay_mask=torch.nested.as_nested_tensor([torch.ones(3), torch.ones(2)], layout=torch.jagged),
        cp_layout="contiguous",
    )
    merged_routes = []
    replay_utils.merge_router_topk_indices(
        None,
        _nested_routes(),
        merged_routes,
        config,
        cp_layout="contiguous",
    )

    assert observed_layouts == [
        ("preprocess", "contiguous", 1),
        ("preprocess", "contiguous", 1),
        ("preprocess", "contiguous", 1),
        ("postprocess", "contiguous"),
    ]


def test_engine_passes_its_single_cp_layout_source_to_both_replay_directions():
    path = Path(__file__).parents[2] / "verl" / "workers" / "engine" / "megatron" / "transformer_impl.py"
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"merge_router_topk_indices", "set_router_replay_data"}
    ]

    assert {call.func.id for call in calls} == {"merge_router_topk_indices", "set_router_replay_data"}
    for call in calls:
        cp_layout = next(keyword.value for keyword in call.keywords if keyword.arg == "cp_layout")
        assert isinstance(cp_layout, ast.Name) and cp_layout.id == "cp_layout"
        local_cp_size = next(keyword.value for keyword in call.keywords if keyword.arg == "local_cp_size")
        assert isinstance(local_cp_size, ast.Name) and local_cp_size.id == "local_cp_size"


def test_all_production_thd_layout_calls_are_explicit():
    root = Path(__file__).parents[2] / "verl"
    missing = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name not in {"preprocess_thd_engine", "postprocess_thd_engine"}:
                continue
            has_layout = any(keyword.arg == "cp_layout" for keyword in node.keywords)
            expands_shared_options = any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "thd_preprocess_options"
                for keyword in node.keywords
            )
            if not has_layout and not expands_shared_options:
                missing.append(f"{path.relative_to(root.parent)}:{node.lineno}")

    assert missing == []


def _find_unshared_thd_preprocess_calls(source_by_path):
    unshared = []
    for path, source in source_by_path.items():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "preprocess_thd_engine"
            ):
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            expands_shared_options = any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "thd_preprocess_options"
                for keyword in node.keywords
            )
            manually_passed = keyword_names & _SHARED_THD_OPTION_NAMES
            if not expands_shared_options or manually_passed:
                unshared.append(f"{path}:{node.lineno}")
    return unshared


def test_router_replay_and_forward_cannot_diverge_on_thd_preprocess_options():
    root = Path(__file__).parents[2]
    paths = [
        root / "verl" / "models" / "mcore" / "model_forward.py",
        root / "verl" / "models" / "mcore" / "model_forward_fused.py",
        root / "verl" / "utils" / "megatron" / "router_replay_utils.py",
    ]
    sources = {path.relative_to(root): path.read_text() for path in paths}

    assert _find_unshared_thd_preprocess_calls(sources) == []


def test_thd_preprocess_option_audit_rejects_missing_forward_bundle():
    source = """
preprocess_thd_engine(value, **thd_preprocess_options)
preprocess_thd_engine(value, cp_layout=cp_layout, use_fp8_padding=use_fp8_padding)
"""

    assert _find_unshared_thd_preprocess_calls({"fault_injected_forward.py": source}) == ["fault_injected_forward.py:3"]
