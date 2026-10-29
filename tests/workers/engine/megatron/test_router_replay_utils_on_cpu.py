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

"""Replay scope and padded token order are independent of the global registry."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

from verl.utils.megatron import router_replay_utils as rr
from verl.utils.megatron.native import configure_router_layer_mapping


class Router(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 2))
        self.layer_number = layer
        self.router_replay = RouterReplay()


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(rr, "TopKRouter", Router)
    monkeypatch.setattr(rr, "device_name", "cpu")
    model = torch.nn.Module()
    model.decoder = torch.nn.ModuleList([Router(2), Router(4)])
    return model


def test_bridge_maps_expanded_hybrid_layers_back_to_hf(model):
    tasks = [
        SimpleNamespace(
            global_param_name=f"decoder.layers.{i}.mlp.router.weight",
            hf_param_names=(f"model.layers.{hf}.mlp.gate.weight",),
            param_weight=router.weight,
        )
        for i, hf, router in [(1, 0, model.decoder[0]), (3, 2, model.decoder[1])]
    ]
    configure_router_layer_mapping([model], tasks, 3)
    assert [layer for layer, _ in rr.iter_model_routers(model)] == [1, 3]
    assert model._verl_replay_moe_layer_ids == (0, 2)


def test_scope_cleans_up_on_failure_without_touching_other_models(model):
    unrelated = RouterReplay()
    unrelated.set_router_replay_action(RouterReplayAction.RECORD)
    with pytest.raises(RuntimeError, match="failed schedule"):
        with rr.router_replay_scope([SimpleNamespace(module=model)], RouterReplayAction.REPLAY_FORWARD):
            for _, router in rr.iter_model_routers(model):
                router.set_target_indices(torch.zeros(2, 1, dtype=torch.long), torch.ones(2))
            raise RuntimeError("failed schedule")
    for _, router in rr.iter_model_routers(model):
        assert router.router_replay_action is None
        assert router.target_topk_idx is None
        assert router.target_replay_mask is None
        assert not router.replay_backward_list
    assert unrelated.router_replay_action == RouterReplayAction.RECORD


def test_replay_disabled_does_not_walk_models(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Disabled replay must not inspect router state.")

    monkeypatch.setattr(rr, "iter_model_routers", forbidden)
    with rr.router_replay_scope([object()]):
        pass


def test_bshd_record_replay_roundtrip_uses_sequence_major_order(model, monkeypatch):
    from verl.models.mcore import util

    for name, value in {
        "get_tensor_model_parallel_world_size": 1,
        "get_context_parallel_world_size": 1,
        "get_context_parallel_rank": 0,
    }.items():
        monkeypatch.setattr(util.mpu, name, lambda value=value: value)
    monkeypatch.setattr(util.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(rr, "gather_from_sequence_parallel_region", lambda value, **kw: value)
    monkeypatch.setattr(rr, "scatter_to_sequence_parallel_region", lambda value, **kw: value)
    config = SimpleNamespace(fp8=None, num_layers=4, moe_layer_freq=[0, 1, 0, 1], moe_router_topk=2)
    tokens = torch.nested.as_nested_tensor([torch.arange(3), torch.arange(2)], layout=torch.jagged)
    original = []
    for index, router in enumerate(model.decoder):
        routes = torch.arange(12).reshape(6, 2) + 100 * index
        router.router_replay.record_indices(routes)
        original.append(routes)
    recorded = []
    rr.merge_router_topk_indices(None, tokens, recorded, config, model=model, data_format="bshd")
    rr.set_router_replay_data(recorded[0], None, config, model=model, data_format="bshd")
    # S-major rows: (s0,b0), (s0,b1), (s1,b0), (s1,b1), (s2,b0), (pad,b1).
    for router, expected in zip(model.decoder, original, strict=True):
        torch.testing.assert_close(router.router_replay.target_topk_idx[:5], expected[:5])
