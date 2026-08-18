# Copyright 2025 Individual Contributor: TomQunChaoA
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

import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics, dump_train_infer_input


class TestMetrics(unittest.TestCase):
    def test_calculate_debug_metrics(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor(
                    [
                        [-1.5085, -0.1200, -0.6650, -0.4823, -0.1426, -1.5557, -2.8532, -0.3919, -0.4294, -0.4700],
                        [-0.0585, -0.0573, -0.4681, -0.5187, -0.7451, -1.2737, -0.0682, -0.4284, -0.5754, -0.0611],
                    ]
                ),
                "old_log_probs": torch.tensor(
                    [
                        [-1.8636, -0.7863, -0.2136, -0.4376, -2.0257, -0.2579, -1.1547, -0.5203, -0.3802, -0.9872],
                        [-0.3507, -0.5426, -0.2725, -0.4637, -0.3577, -0.3733, -1.7560, -1.9542, -0.4229, -1.3098],
                    ]
                ),
                "loss_mask": torch.tensor([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]),
                "responses": torch.zeros((2, 10)),
            }
        )
        metrics = calculate_debug_metrics(data)
        print(metrics)
        assert metrics["training/rollout_probs_diff_valid"] == 1

    def test_calculate_debug_metrics_can_dump_token_level_diff(self):
        rollout = torch.tensor([[-1.0, -2.0]], dtype=torch.float16)
        actor = torch.tensor([[-1.0, -2.5]], dtype=torch.float16)
        data = DataProto.from_dict(
            tensors={
                "rollout_log_probs": rollout,
                "old_log_probs": actor,
                "loss_mask": torch.tensor([[1, 1]]),
                "responses": torch.tensor([[7, 8]]),
                "prompts": torch.tensor([[3, 4]]),
            },
            meta_info={"temperature": 1.0},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diff.jsonl")
            with mock.patch.dict(os.environ, {"VERL_TRAIN_INFER_DIFF_DUMP": path}):
                metrics = calculate_debug_metrics(data)

            record = json.loads(open(path, encoding="utf-8").read())
            sample = record["samples"][0]
            assert sample["token_ids"] == [7, 8]
            assert sample["bitwise_equal_count"] == 1
            assert sample["valid_token_count"] == 2
            assert sample["logprob_abs_diff"] == [0.0, 0.5]
            assert metrics["training/rollout_logprob_abs_diff_max"] == 0.5
            assert metrics["training/rollout_logprob_bitwise_equal_fraction"] == 0.5
            raw = torch.load(os.path.join(directory, "diff.pt"), weights_only=True)
            assert torch.equal(raw["RL.vllm.rollout_log_probs"], rollout)
            assert torch.equal(raw["RL.mlite.old_log_probs"], actor)
            assert raw["RL.vllm.rollout_log_probs"].dtype == torch.float16
            assert raw["responses"].tolist() == [[7, 8]]
            assert raw["response_mask"].dtype == torch.int64
            assert raw["input_batch"]["prompts"].tolist() == [[3, 4]]
            assert raw["batch_meta_info"] == {"temperature": 1.0}
            assert raw["provenance"]["sources"]["RL.mlite.old_log_probs"] == "old_log_probs"

    def test_train_infer_dump_is_rank_zero_only(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor([[-1.0]]),
                "old_log_probs": torch.tensor([[-1.0]]),
                "loss_mask": torch.tensor([[1]]),
                "responses": torch.tensor([[7]]),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diff.jsonl")
            with mock.patch.dict(
                os.environ,
                {"VERL_TRAIN_INFER_DIFF_DUMP": path, "RANK": "1"},
            ):
                calculate_debug_metrics(data)

            assert not os.path.exists(path)
            assert not os.path.exists(os.path.join(directory, "diff.pt"))

    def test_compact_train_infer_dump_keeps_aggregates_without_raw_sidecar(self):
        rollout = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        actor = torch.tensor([[-1.0, -2.5], [-2.0, -4.0]])
        data = DataProto.from_dict(
            tensors={
                "rollout_log_probs": rollout,
                "old_log_probs": actor,
                "loss_mask": torch.ones_like(rollout, dtype=torch.int64),
                "responses": torch.tensor([[7, 8], [9, 10]]),
                "attention_mask": torch.ones((2, 4), dtype=torch.int64),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diff.jsonl")
            with mock.patch.dict(
                os.environ,
                {
                    "VERL_TRAIN_INFER_DIFF_DUMP": path,
                    "VERL_TRAIN_INFER_DIFF_MODE": "compact",
                    "VERL_TRAIN_INFER_TOKEN_SAMPLE_LIMIT": "1",
                },
            ):
                calculate_debug_metrics(data)

            record = json.loads(open(path, encoding="utf-8").read())
            assert record["schema_version"] == 2
            assert record["mode"] == "compact"
            first, second = record["samples"]
            assert first["token_ids"] == [7, 8]
            assert second["token_ids"] == []
            assert first["logprob_abs_diff_max"] == 0.5
            assert first["logprob_abs_diff_sum"] == 0.5
            assert first["bitwise_equal_count"] == 1
            assert first["all_logprobs_finite"] is True
            assert "rollout_log_probs" not in first
            assert "actor_log_probs" not in first
            assert "logprob_abs_diff" not in first
            assert not os.path.exists(os.path.join(directory, "diff.pt"))

    def test_pre_forward_dump_survives_without_actor_log_probs(self):
        data = DataProto.from_dict(
            tensors={
                "rollout_log_probs": torch.tensor([[-1.0, -2.0]]),
                "responses": torch.tensor([[7, 8]]),
                "response_mask": torch.tensor([[1, 1]]),
                "input_ids": torch.tensor([[3, 4, 7, 8]]),
            },
            meta_info={"temperature": 1.0},
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"VERL_TRAIN_INFER_PRE_FORWARD_DUMP_DIR": directory, "RANK": "0"},
            ):
                dump_train_infer_input(data, step=7)

            payload = torch.load(
                os.path.join(directory, "step00007.pt"), weights_only=True
            )
            assert payload["step"] == 7
            assert payload["RL.vllm.rollout_log_probs"].tolist() == [[-1.0, -2.0]]
            assert payload["responses"].tolist() == [[7, 8]]
            assert payload["input_batch"]["input_ids"].tolist() == [[3, 4, 7, 8]]
            assert "RL.mlite.old_log_probs" not in payload


if __name__ == "__main__":
    unittest.main()
