import sys
from types import SimpleNamespace

import pytest

from verl.single_controller.base.worker import collect_determinism_evidence
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.batch_invariant import (
    apply_batch_invariant,
    resolve_batch_invariant,
    scope_batch_invariant_env,
)


def test_runtime_env_forwards_determinism_evidence_inputs(monkeypatch) -> None:
    expected = {
        "VERL_ACTOR_BATCH_INVARIANT": "0",
        "VERL_ROLLOUT_BATCH_INVARIANT": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_BATCH_INVARIANT_KERNEL_LIB": (
            "/workspace/_vllm_batch_invariant_C.so"
        ),
        "VERL_FULL_DETERMINISM": "1",
        "VERL_DETERMINISM_SEED": "42",
        "PYTHONHASHSEED": "42",
        "PYTHONPATH": "/workspace/verl:/workspace/mlite",
        "MLITE_CUDA_SYNC_BOUNDARIES": "1",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)

    runtime_env = get_ppo_ray_runtime_env()

    assert {key: runtime_env["env_vars"][key] for key in expected} == expected


@pytest.mark.parametrize(
    ("key", "value"),
    [("VERL_DISTRIBUTED_TIMEOUT_S", "123"), ("VLLM_CACHE_ROOT", "/cache/vllm")],
)
def test_runtime_env_preserves_optional_defaults(monkeypatch, key, value) -> None:
    monkeypatch.delenv(key, raising=False)
    assert key not in get_ppo_ray_runtime_env()["env_vars"]

    monkeypatch.setenv(key, value)
    assert get_ppo_ray_runtime_env()["env_vars"][key] == value


def test_role_scoped_batch_invariant_inputs_are_independent() -> None:
    env = {
        "VERL_ACTOR_BATCH_INVARIANT": "0",
        "VERL_ROLLOUT_BATCH_INVARIANT": "1",
        "VLLM_BATCH_INVARIANT": "1",
    }

    assert resolve_batch_invariant("actor", environ=env) == "0"
    assert resolve_batch_invariant("rollout", environ=env) == "1"


def test_rollout_full_determinism_does_not_enable_actor_batch_invariance() -> None:
    assert resolve_batch_invariant("actor", environ={}) == "0"
    assert resolve_batch_invariant("rollout", rollout_full_determinism=True, environ={}) == "1"


def test_scoping_neutralizes_global_value_after_capturing_roles() -> None:
    env = {"VERL_ACTOR_BATCH_INVARIANT": "0", "VLLM_BATCH_INVARIANT": "1"}

    assert scope_batch_invariant_env(rollout_full_determinism=True, environ=env) == ("0", "1")
    assert env == {
        "VERL_ACTOR_BATCH_INVARIANT": "0",
        "VERL_ROLLOUT_BATCH_INVARIANT": "1",
        "VLLM_BATCH_INVARIANT": "0",
    }


def test_legacy_batch_invariant_remains_a_fallback() -> None:
    env = {"VLLM_BATCH_INVARIANT": "1"}

    assert resolve_batch_invariant("actor", environ=env) == "1"
    assert resolve_batch_invariant("rollout", environ=env) == "1"


def test_apply_batch_invariant_prints_role_and_actual_value(capsys) -> None:
    env = {"VERL_ACTOR_BATCH_INVARIANT": "0", "VLLM_BATCH_INVARIANT": "1"}

    assert apply_batch_invariant("actor", environ=env, evidence_role="actor-worker") == "0"
    assert env["VLLM_BATCH_INVARIANT"] == "0"
    assert (
        capsys.readouterr().out
        == "VERL_BATCH_INVARIANT_EVIDENCE role=actor-worker VLLM_BATCH_INVARIANT=0\n"
    )


def test_invalid_explicit_batch_invariant_fails_closed() -> None:
    with pytest.raises(ValueError, match="VERL_ROLLOUT_BATCH_INVARIANT"):
        resolve_batch_invariant(
            "rollout",
            rollout_full_determinism=True,
            environ={"VERL_ROLLOUT_BATCH_INVARIANT": "yes"},
        )


def test_collect_determinism_evidence_reports_deep_gemm_getter(monkeypatch) -> None:
    deep_gemm = SimpleNamespace(
        __file__="/opt/deep_gemm/__init__.py",
        get_batch_invariant=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "deep_gemm", deep_gemm)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VERL_FULL_DETERMINISM", "1")
    monkeypatch.setenv("VERL_DETERMINISM_SEED", "42")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("PYTHONPATH", "/workspace/verl:/workspace/mlite:/vendor")

    evidence = collect_determinism_evidence()

    assert evidence["VLLM_BATCH_INVARIANT"] == "1"
    assert evidence["VERL_FULL_DETERMINISM"] == "1"
    assert evidence["seed"] == "42"
    assert evidence["deep_gemm_file"] == "/opt/deep_gemm/__init__.py"
    assert evidence["deep_gemm_batch_invariant"] is True
    assert evidence["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert evidence["PYTHONPATH_prefix"] == ["/workspace/verl", "/workspace/mlite", "/vendor"]
