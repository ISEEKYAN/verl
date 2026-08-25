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
import importlib.util
from pathlib import Path

import pytest


def _load_module(name: str, relative_path: str):
    path = Path(__file__).parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


order_hybrid_workers = _load_module(
    "worker_ordering",
    "verl/workers/rollout/worker_ordering.py",
).order_hybrid_workers


def test_hybrid_workers_are_ordered_by_hostname_domain_and_gpu():
    worker_infos = [
        ("t07-gpu1", "node-07", "nvl72d056-T07", "1"),
        ("t05-gpu1", "node-05", "nvl72d056-T05", "1"),
        ("next-gpu0", "node-next", "nvl72d057-T01", "0"),
        ("t07-gpu0", "node-07", "nvl72d056-T07", "0"),
        ("next-gpu1", "node-next", "nvl72d057-T01", "1"),
        ("t05-gpu0", "node-05", "nvl72d056-T05", "0"),
    ]

    assert order_hybrid_workers(worker_infos, gpus_per_node=2) == [
        "t05-gpu0",
        "t05-gpu1",
        "t07-gpu0",
        "t07-gpu1",
        "next-gpu0",
        "next-gpu1",
    ]


@pytest.mark.parametrize(
    "worker_infos, message",
    [
        ([("gpu0", "node", "host", 0)], "complete GPU nodes"),
        (
            [("gpu0-a", "node", "host", 0), ("gpu0-b", "node", "host", 0)],
            "duplicate GPU IDs",
        ),
    ],
)
def test_hybrid_worker_order_rejects_invalid_topology(worker_infos, message):
    with pytest.raises(RuntimeError, match=message):
        order_hybrid_workers(worker_infos, gpus_per_node=2)
