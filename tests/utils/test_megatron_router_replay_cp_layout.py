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


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_router_replay_utils(monkeypatch, observed_layouts):
    def preprocess(value, *, cp_layout="zigzag", **kwargs):
        del kwargs
        observed_layouts.append(("preprocess", cp_layout))
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
        ("preprocess", "contiguous"),
        ("preprocess", "contiguous"),
        ("preprocess", "contiguous"),
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
            if not any(keyword.arg == "cp_layout" for keyword in node.keywords):
                missing.append(f"{path.relative_to(root.parent)}:{node.lineno}")

    assert missing == []
