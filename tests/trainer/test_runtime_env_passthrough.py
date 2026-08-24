from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def test_vllm_decode_kernel_is_forwarded_only_when_set(monkeypatch):
    monkeypatch.delenv("VLLM_DS4_DECODE_KERNEL", raising=False)
    assert "VLLM_DS4_DECODE_KERNEL" not in get_ppo_ray_runtime_env()["env_vars"]

    monkeypatch.setenv("VLLM_DS4_DECODE_KERNEL", "sparse")
    assert get_ppo_ray_runtime_env()["env_vars"]["VLLM_DS4_DECODE_KERNEL"] == "sparse"
