# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Checkpoint placement is restored after successful and failed operations."""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.core")

from verl.workers.engine.megatron import transformer_impl as impl


@pytest.mark.parametrize("model_on_cpu", [False, True])
@pytest.mark.parametrize("optimizer_offload", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_scope_orders_materialization_and_restores_placement(
    monkeypatch, model_on_cpu, optimizer_offload, fail
):
    calls = []
    monkeypatch.setattr(impl, "get_megatron_module_device", lambda model: "cpu" if model_on_cpu else "cuda")
    for function, event in [
        ("load_megatron_model_to_gpu", "load_model"),
        ("load_megatron_optimizer", "load_optimizer"),
        ("offload_megatron_optimizer", "offload_optimizer"),
        ("offload_megatron_model_to_cpu", "offload_model"),
    ]:
        monkeypatch.setattr(impl, function, lambda *args, _event=event, **kwargs: calls.append(_event))
    engine = SimpleNamespace(module=[object()], optimizer=object(), _is_offload_optimizer=optimizer_offload)
    try:
        with impl.MegatronEngine._checkpoint_device_scope(engine):
            calls.append("checkpoint")
            if fail:
                raise RuntimeError("checkpoint failed")
    except RuntimeError:
        assert fail
    expected = (["load_model"] if model_on_cpu else []) + (["load_optimizer"] if optimizer_offload else [])
    expected += ["checkpoint"]
    expected += (["offload_optimizer"] if optimizer_offload else []) + (["offload_model"] if model_on_cpu else [])
    assert calls == expected
