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

import os
from collections.abc import Mapping, MutableMapping
from typing import Literal


BatchInvariantRole = Literal["actor", "rollout"]

_ROLE_ENV = {
    "actor": "VERL_ACTOR_BATCH_INVARIANT",
    "rollout": "VERL_ROLLOUT_BATCH_INVARIANT",
}


def _binary_env(name: str, value: str) -> str:
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0' or '1', got {value!r}")
    return value


def resolve_batch_invariant(
    role: BatchInvariantRole,
    *,
    rollout_full_determinism: bool = False,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a role-scoped vLLM batch-invariance value.

    Role-specific inputs have highest priority. For compatibility,
    ``rollout.full_determinism`` still enables rollout batch invariance and the
    legacy ``VLLM_BATCH_INVARIANT`` remains a fallback.
    """
    env = os.environ if environ is None else environ
    role_env = _ROLE_ENV[role]
    if role_env in env:
        return _binary_env(role_env, env[role_env])
    if role == "rollout" and rollout_full_determinism:
        return "1"
    if "VLLM_BATCH_INVARIANT" in env:
        return _binary_env("VLLM_BATCH_INVARIANT", env["VLLM_BATCH_INVARIANT"])
    return "0"


def scope_batch_invariant_env(
    *,
    rollout_full_determinism: bool = False,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, str]:
    """Capture role values and neutralize the process-global legacy input."""
    env = os.environ if environ is None else environ
    actor_value = resolve_batch_invariant("actor", environ=env)
    rollout_value = resolve_batch_invariant(
        "rollout",
        rollout_full_determinism=rollout_full_determinism,
        environ=env,
    )
    env["VERL_ACTOR_BATCH_INVARIANT"] = actor_value
    env["VERL_ROLLOUT_BATCH_INVARIANT"] = rollout_value
    env["VLLM_BATCH_INVARIANT"] = "0"
    return actor_value, rollout_value


def apply_batch_invariant(
    role: BatchInvariantRole,
    *,
    rollout_full_determinism: bool = False,
    environ: MutableMapping[str, str] | None = None,
    evidence_role: str | None = None,
) -> str:
    """Set the process-local vLLM input and print fail-closed evidence."""
    env = os.environ if environ is None else environ
    value = resolve_batch_invariant(
        role,
        rollout_full_determinism=rollout_full_determinism,
        environ=env,
    )
    env["VLLM_BATCH_INVARIANT"] = value
    print(
        "VERL_BATCH_INVARIANT_EVIDENCE "
        f"role={evidence_role or role} VLLM_BATCH_INVARIANT={value}",
        flush=True,
    )
    return value
