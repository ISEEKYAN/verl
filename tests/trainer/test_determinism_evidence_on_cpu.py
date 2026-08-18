import sys
from types import SimpleNamespace

from verl.single_controller.base.worker import collect_determinism_evidence
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def test_runtime_env_forwards_determinism_evidence_inputs(monkeypatch) -> None:
    expected = {
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
