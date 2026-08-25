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
from collections.abc import Iterable
from typing import TypeVar


Worker = TypeVar("Worker")


def order_hybrid_workers(
    worker_infos: Iterable[tuple[Worker, str, str, int | str]],
    gpus_per_node: int,
) -> list[Worker]:
    """Order hybrid workers by physical hostname and local GPU index."""
    workers_by_node: dict[str, tuple[str, list[tuple[int, Worker]]]] = {}
    for worker, node_id, hostname, device_id in worker_infos:
        node = workers_by_node.setdefault(node_id, (hostname, []))
        if node[0] != hostname:
            raise RuntimeError(f"Ray node {node_id} reported conflicting hostnames")
        node[1].append((int(device_id), worker))

    incomplete_nodes = {
        hostname: len(workers)
        for hostname, workers in workers_by_node.values()
        if len(workers) != gpus_per_node
    }
    if incomplete_nodes:
        raise RuntimeError(
            "Hybrid rollout requires complete GPU nodes before forming replicas: "
            f"{incomplete_nodes}"
        )

    for hostname, workers in workers_by_node.values():
        device_ids = [device_id for device_id, _ in workers]
        if len(device_ids) != len(set(device_ids)):
            raise RuntimeError(f"Hybrid rollout found duplicate GPU IDs on {hostname}: {device_ids}")

    # GB200 nodes in one MNNVL/NVL8 domain share the prefix before "-T".
    # Keep those nodes contiguous, then assign ranks in local GPU order.
    ordered_nodes = sorted(
        workers_by_node.values(),
        key=lambda node: (node[0].split("-T", 1)[0], node[0]),
    )
    return [
        worker
        for _, workers in ordered_nodes
        for _, worker in sorted(workers, key=lambda item: item[0])
    ]
