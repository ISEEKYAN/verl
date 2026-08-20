from datetime import timedelta
from unittest.mock import patch

import pytest

from verl.utils.distributed import initialize_global_process_group_ray


@pytest.mark.parametrize(
    ("timeout_second", "expected_timeout"),
    [(None, None), (123, timedelta(seconds=123))],
)
def test_initialize_global_process_group_ray_omits_unset_timeout(
    monkeypatch, timeout_second, expected_timeout
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.distributed.init_process_group") as init_process_group,
    ):
        initialize_global_process_group_ray(timeout_second=timeout_second, backend="gloo")

    kwargs = init_process_group.call_args.kwargs
    if expected_timeout is None:
        assert "timeout" not in kwargs
    else:
        assert kwargs["timeout"] == expected_timeout
