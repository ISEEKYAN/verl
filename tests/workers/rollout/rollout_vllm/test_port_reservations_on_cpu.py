from __future__ import annotations

import inspect

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class _Reservation:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_engine_port_reservations_are_released_idempotently() -> None:
    server = object.__new__(vLLMHttpServer)
    reservations = [_Reservation() for _ in range(3)]
    server._master_sock, server._dp_rpc_sock, server._dp_master_sock = reservations

    server._release_engine_port_reservations()
    server._release_engine_port_reservations()

    assert [reservation.close_calls for reservation in reservations] == [1, 1, 1]
    assert server._master_sock is None
    assert server._dp_rpc_sock is None
    assert server._dp_master_sock is None


def test_reservations_are_released_before_async_llm_creation() -> None:
    source = inspect.getsource(vLLMHttpServer.run_server)

    assert source.index("self._release_engine_port_reservations()") < source.index(
        "engine_client = AsyncLLM.from_vllm_config"
    )
