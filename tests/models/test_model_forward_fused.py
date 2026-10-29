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

"""Native output processors preserve the forward method, token order and gradients."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR

from verl.models.mcore import model_forward_fused as mff
from verl.utils.megatron.native import get_language_model, validate_native_model


class NativeModel(torch.nn.Module):
    def forward(self, *, output_processor=None, output_processor_context=None):
        return output_processor_context


def test_configure_preserves_forward_and_supports_nested_wrappers():
    model = NativeModel()
    wrapper = SimpleNamespace(module=SimpleNamespace(module=SimpleNamespace(language_model=model)))
    forward = model.forward.__func__
    mff.configure_fused_forward(wrapper)
    mff.configure_fused_forward(wrapper)
    assert get_language_model(wrapper) is model
    assert model.forward.__func__ is forward
    assert not hasattr(model, "forward_backup")


def test_legacy_models_fail_before_training():
    with pytest.raises(ValueError, match="lacks native Megatron APIs"):
        validate_native_model(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("tied", [False, True])
def test_processor_preserves_batch_sequence_order_gradients_and_readiness(monkeypatch, batch_size, tied):
    torch.manual_seed(17)
    hidden = torch.randn(4, batch_size, 5, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(7, 5))
    labels = torch.arange(batch_size * 4).reshape(batch_size, 4) % 7
    ready = []
    setattr(weight, PARAM_READY_CALLBACK_ATTR, lambda: ready.append(True))
    output_layer = SimpleNamespace(weight=weight, tp_group=None)

    def eager_kernel(h, w, targets, temperature, reduction, group, **kwargs):
        assert ready, "Parameter readiness must be established before a direct weight read."
        logits = F.linear(h, w) / temperature
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).reshape(-1), entropy.reshape(-1)

    monkeypatch.setattr(mff, "linear_cross_entropy", eager_kernel)
    output = mff.fused_output_processor(
        hidden_states=hidden,
        output_layer=output_layer,
        output_weight=weight if tied else None,
        labels=labels,
        context=mff.FusedOutputProcessorContext(0.7),
        config=SimpleNamespace(sequence_parallel=False, config_logger_dir=""),
    )
    reference_logits = F.linear(hidden.transpose(0, 1), weight) / 0.7
    reference = reference_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(output["log_probs"], reference)
    grad = torch.autograd.grad(output["log_probs"].sum(), (hidden, weight), retain_graph=True)
    expected_grad = torch.autograd.grad(reference.sum(), (hidden, weight))
    for actual, expected in zip(grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, expected)
    assert ready == [True]


@pytest.mark.parametrize("backend,reduce_grad", [("triton", True), ("liger", False)])
def test_sp_gather_uses_explicit_group_and_backend_gradient_contract(monkeypatch, backend, reduce_grad):
    hidden = torch.randn(2, 1, 3)
    group = object()
    seen = []
    monkeypatch.setattr(mff, "gather_from_sequence_parallel_region", lambda value, **kw: seen.append(kw) or value)
    assert mff._gather_fused_hidden_states(hidden, True, backend, group) is hidden
    assert seen == [{"tensor_parallel_output_grad": reduce_grad, "group": group}]


@pytest.mark.parametrize("mtp_enabled", [False, True])
def test_bshd_adapter_passes_native_mtp_controls_and_preserves_output_layout(monkeypatch, mtp_enabled):
    from verl.models.mcore import util

    for name, value in {
        "get_tensor_model_parallel_world_size": 1,
        "get_context_parallel_world_size": 1,
        "get_context_parallel_rank": 0,
        "get_context_parallel_group": None,
    }.items():
        monkeypatch.setattr(util.mpu, name, lambda value=value: value)
    captured = {}

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_process = True
            self.config = SimpleNamespace(fp8=None, mtp_num_layers=1)

        def forward(self, input_ids, output_processor, output_processor_context, **kwargs):
            captured.update(kwargs)
            captured["context_labels"] = output_processor_context.labels
            # The callback receives Core's native sequence-major hidden layout.
            return output_processor(
                hidden_states=input_ids.T.unsqueeze(-1).float(),
                output_layer=None,
                output_weight=None,
                labels=kwargs.get("labels"),
                context=output_processor_context,
                config=self.config,
            )

    def processor(hidden_states, context, **kwargs):
        return {"log_probs": hidden_states.squeeze(-1).T + context.labels.float()}

    tokens = torch.nested.as_nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
    response_mask = torch.tensor([[1, 0], [1, 0]])
    response_attention = torch.tensor([[1, 1], [1, 0]])
    result = mff.gptmodel_forward_model_engine(
        Toy(),
        tokens,
        {},
        logits_processor_args={
            "label": tokens,
            "loss_mask": response_mask,
            "response_attention_mask": response_attention,
        },
        data_format="bshd",
        mtp_enable_train=mtp_enabled,
        mtp_loss_normalization_factor=1.75,
        output_processor=processor,
        output_processor_context=mff.FusedOutputProcessorContext(1.0),
    )
    assert captured["compute_mtp_loss"] is mtp_enabled
    assert captured["mtp_loss_normalization_factor"] == 1.75
    torch.testing.assert_close(captured["context_labels"], torch.tensor([[2, 3, 1], [5, 0, 4]]))
    if mtp_enabled:
        torch.testing.assert_close(captured["labels"], captured["context_labels"])
        torch.testing.assert_close(captured["loss_mask"], torch.tensor([[1, 0, 0], [1, 0, 0]]))
    else:
        assert "labels" not in captured
        assert "loss_mask" not in captured
    torch.testing.assert_close(result["log_probs"].values(), torch.tensor([3.0, 5.0, 4.0, 9.0, 5.0]))
