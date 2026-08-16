from types import SimpleNamespace
from unittest.mock import Mock

from verl.trainer.ppo.ray_trainer import _finalize_train_infer_only


def test_finalize_train_infer_only_logs_and_closes_without_update(monkeypatch) -> None:
    monkeypatch.setenv("VERL_STOP_AFTER_TRAIN_INFER_DIFF", "1")
    actor = SimpleNamespace(async_calls_finalize_fn_exec=Mock())
    trainer = SimpleNamespace(
        global_steps=3,
        actor_rollout_wg=actor,
        _shutdown_dump_executor=Mock(),
    )
    tracker = Mock()
    progress = Mock()
    metrics = {"training/rollout_probs_diff_valid": 1}

    assert _finalize_train_infer_only(trainer, tracker, metrics, progress)
    assert metrics["trainer/train_infer_only"] == 1
    tracker.log.assert_called_once_with(data=metrics, step=3)
    progress.update.assert_called_once_with(1)
    progress.close.assert_called_once_with()
    actor.async_calls_finalize_fn_exec.assert_called_once_with(blocking=True)
    trainer._shutdown_dump_executor.assert_called_once_with()


def test_finalize_train_infer_only_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VERL_STOP_AFTER_TRAIN_INFER_DIFF", raising=False)
    assert not _finalize_train_infer_only(Mock(), Mock(), {}, Mock())
