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


import pytest
import torch

pytest.importorskip("megatron")

import verl.utils.megatron_utils as megatron_utils


@pytest.mark.parametrize(
    ("tracker", "expected_scale"),
    [
        ({"loss_sums": object(), "calculate_per_token_loss": True}, 1.0),
        ({"loss_values": object()}, 0.25),
    ],
)
def test_mtp_metric_scale_matches_mcore_tracker(monkeypatch, tracker, expected_scale):
    captured = {}

    class _LoggingHelper:
        @staticmethod
        def track_mtp_metrics(*, loss_scale, total_loss_dict, **_kwargs):
            captured["scale"] = loss_scale
            total_loss_dict["mtp_1 loss"] = torch.tensor(loss_scale)

    _LoggingHelper.tracker = tracker
    monkeypatch.setattr(megatron_utils, "MTPLossLoggingHelper", _LoggingHelper)

    metrics = megatron_utils.get_megatron_mtp_loss(n_micro_batch=4)
    assert captured["scale"] == expected_scale
    assert metrics == {"mtp_losses/mtp_1_loss": expected_scale}
