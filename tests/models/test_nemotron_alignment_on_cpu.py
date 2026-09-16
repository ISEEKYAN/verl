# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for inference-visible forward/native backward selection."""

import pytest
import torch

from verl.models.transformers.nemotron_h_alignment import visible_forward


def test_visible_value_and_native_gradient_with_frozen_input():
    """Forward bits come from inference; only trainable inputs receive native VJPs."""
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    weight = torch.tensor([4.0, 5.0])
    native = lambda a, b: a.square() * b
    visible = lambda a, b: native(a, b) + 0.125
    actual = visible_forward(visible, native, x, weight)
    assert torch.equal(actual, visible(x, weight))
    upstream = torch.tensor([0.5, -2.0])
    expected = torch.autograd.grad(native(x, weight), x, upstream)[0]
    assert torch.equal(torch.autograd.grad(actual, x, upstream)[0], expected)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_no_grad_forward_never_calls_native(requires_grad):
    """Scoring must not materialize the native backward's dense intermediates."""
    x = torch.tensor([2.0], requires_grad=requires_grad)

    def native(_):
        raise AssertionError("native forward is only needed for backward")

    with torch.no_grad():
        assert torch.equal(visible_forward(torch.square, native, x), x.square())


def test_parameter_mutation_before_backward_is_rejected():
    """A weight update cannot silently recompute the VJP with different weights."""
    x = torch.tensor([2.0], requires_grad=True)
    result = visible_forward(torch.square, torch.square, x)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()
