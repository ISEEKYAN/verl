# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _session_sampling_params, _settle_session_tasks


def test_session_seeds_are_reproducible_without_collapsing_grpo_siblings():
    params = {"temperature": 1.0}
    trajectory = {"step": 1, "sample_index": 42, "validate": False}
    seeds = [_session_sampling_params(params, 7, trajectory, i)["seed"] for i in range(8)]
    replay = [_session_sampling_params(params, 7, trajectory, i)["seed"] for i in reversed(range(8))]
    assert seeds == replay[::-1]
    assert len(set(seeds)) == 8
    assert all(0 <= seed < (1 << 63) for seed in seeds)
    assert params == {"temperature": 1.0}
    for field, value in (("step", 2), ("sample_index", 43), ("validate", True)):
        assert _session_sampling_params(params, 7, {**trajectory, field: value}, 0)["seed"] != seeds[0]
    assert _session_sampling_params(params, 8, trajectory, 0)["seed"] != seeds[0]


def test_session_seeds_preserve_explicit_seeds_and_nondeterministic_sampling():
    trajectory = {"step": 1, "sample_index": 42, "validate": False}
    for seed in (None, 123):
        params = {"seed": seed, "temperature": 1.0}
        assert _session_sampling_params(params, 7, trajectory, 0) == params
    assert _session_sampling_params({"temperature": 1.0}, None, trajectory, 0) == {"temperature": 1.0}


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())
