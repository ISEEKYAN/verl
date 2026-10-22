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

"""Provider capabilities and Bridge layer mappings fail before distributed execution."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.megatron.native import (
    configure_router_layer_mapping,
    get_language_model,
    logits_output_processor,
    validate_native_model,
)


class Model(torch.nn.Module):
    def forward(
        self,
        *,
        output_processor=None,
        output_processor_context=None,
        compute_mtp_loss=True,
        mtp_loss_normalization_factor=1.0,
    ):
        return output_processor_context


def test_capabilities_follow_distributed_and_multimodal_wrappers():
    model = Model()
    wrapped = SimpleNamespace(module=SimpleNamespace(module=SimpleNamespace(language_model=model)))
    assert get_language_model(wrapped) is model
    validate_native_model(wrapped, mtp=True)


def test_missing_mtp_capability_is_rejected():
    class Legacy(Model):
        def forward(self, *, output_processor=None, output_processor_context=None):
            pass

    with pytest.raises(ValueError, match="compute_mtp_loss"):
        validate_native_model(Legacy(), mtp=True)


def test_logits_callback_retains_hidden_and_weight_gradients():
    class Output(torch.nn.Linear):
        def forward(self, hidden, weight=None, runtime_gather_output=None):
            return torch.nn.functional.linear(hidden, self.weight if weight is None else weight), None

    layer = Output(3, 5, bias=False)
    hidden = torch.randn(4, 2, 3, requires_grad=True)
    result = logits_output_processor(
        hidden_states=hidden,
        output_layer=layer,
        output_weight=None,
        runtime_gather_output=False,
        scale_logits=lambda logits: logits * 0.5,
    )
    reference = torch.nn.functional.linear(hidden, layer.weight).transpose(0, 1) * 0.5
    torch.testing.assert_close(result, reference)
    actual = torch.autograd.grad(result.sum(), (hidden, layer.weight), retain_graph=True)
    expected = torch.autograd.grad(reference.sum(), (hidden, layer.weight))
    for value, baseline in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, baseline)


def test_mapping_uses_bridge_metadata_and_ignores_mtp():
    model = torch.nn.Module()
    router = torch.nn.Linear(2, 3, bias=False)
    router.router_replay = object()
    model.decoder = torch.nn.ModuleList([router])
    task = SimpleNamespace(
        global_param_name="decoder.layers.3.mlp.router.weight",
        hf_param_names=("model.layers.1.mlp.gate.weight",),
        param_weight=router.weight,
    )
    mtp = SimpleNamespace(global_param_name="mtp.layers.0.mlp.router.weight")
    configure_router_layer_mapping([model], [task, mtp], 4)
    assert router._verl_router_layer_number == 2
    assert model._verl_replay_moe_layer_ids == (1,)
    assert model._verl_replay_hf_depth == 4
    with pytest.raises(ValueError, match="every local"):
        configure_router_layer_mapping([model], [], 4)
    task.hf_param_names = ("model.layers.1.mlp.gate.weight", "model.layers.2.mlp.gate.weight")
    with pytest.raises(ValueError, match="identify one HF layer"):
        configure_router_layer_mapping([model], [task], 4)
