from omegaconf import OmegaConf

from verl.trainer import main_ppo


def test_run_ppo_can_execute_controller_locally(monkeypatch) -> None:
    monkeypatch.setenv("VERL_LOCAL_TASK_RUNNER", "1")
    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    calls = []

    class LocalRunner:
        def run(self, config):
            calls.append(config)

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"full_determinism": False}},
            "reward": {
                "reward_model": {
                    "enable": False,
                    "rollout": {"full_determinism": False},
                }
            },
            "trainer": {"logger": []},
            "transfer_queue": {"enable": False},
            "ray_kwargs": {},
        }
    )
    main_ppo.run_ppo(
        config,
        task_runner_class=object(),
        local_task_runner_class=LocalRunner,
    )
    assert calls == [config]
