import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.modules.setdefault("transfer_queue", MagicMock())
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def test_step_once_stops_after_old_log_prob(monkeypatch) -> None:
    monkeypatch.setenv("VERL_STOP_AFTER_TRAIN_INFER_DIFF", "1")
    trainer = object.__new__(_StubTrainer)
    batch = SimpleNamespace(
        extra_info={},
        tags=[{"is_padding": False}],
        keys=["sample"],
        partition_id="train",
    )
    trainer.global_steps = 1
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(temperature=1.0),
        ),
        trainer=SimpleNamespace(critic_warmup=0),
    )
    trainer.replay_buffer = SimpleNamespace(
        sample=lambda **_kwargs: (batch, {"sample/count": 1})
    )
    trainer.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=object()
    )
    trainer.on_sample_begin = lambda: None
    trainer.on_sample_end = lambda: None
    trainer._balance_batch = lambda value, metrics: value
    trainer._compute_old_log_prob = lambda value, metrics: value

    def unexpected(*_args, **_kwargs):
        raise AssertionError("forward-only mode entered a post-diff training path")

    trainer._compute_ref_log_prob = unexpected
    trainer._compute_values = unexpected
    trainer._compute_advantage = unexpected
    trainer._update_critic = unexpected
    trainer._update_actor = unexpected
    trainer.use_reference_policy = True
    trainer.use_critic = True
    metrics = {}

    result = trainer._step_once(metrics, {}, sample_batch_size=1)

    assert result is batch
    assert metrics["sample/count"] == 1
    assert metrics["trainer/train_infer_only"] == 1
