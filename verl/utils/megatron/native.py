# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Small contracts between the engine and native Megatron model implementations."""

import inspect
import re


def configure_router_layer_mapping(models, conversion_tasks, hf_depth):
    """Map native router weights to HF layer IDs using Bridge's conversion metadata.

    Hybrid models can split one HF block across several native layers. Conversion
    tasks are authoritative; inferring the mapping from native layer counts is not.
    """
    local_routers = {}
    owners = [get_language_model(model) for model in models]
    for owner in owners:
        for module in getattr(owner, "decoder", owner).modules():
            if getattr(module, "router_replay", None) is not None:
                local_routers[id(module.weight)] = module
    global_ids = set()
    resolved = set()
    for task in conversion_tasks:
        name = task.global_param_name
        if "mtp" in name.split(".") or not name.endswith("router.weight"):
            continue
        ids = {
            int(match.group(1))
            for hf_name in task.hf_param_names
            for match in re.finditer(r"(?:layers|blocks)\.(\d+)\.", hf_name)
        }
        if len(ids) != 1:
            raise ValueError(f"Bridge router mapping {name} must identify one HF layer, got {task.hf_param_names}.")
        layer_id = ids.pop()
        if not 0 <= layer_id < hf_depth:
            raise ValueError(f"Bridge mapped router {name} outside HF depth {hf_depth}.")
        global_ids.add(layer_id)
        module = local_routers.get(id(task.param_weight))
        if module is not None:
            module._verl_router_layer_number = layer_id + 1
            resolved.add(id(module.weight))
    if resolved != set(local_routers):
        raise ValueError("Bridge conversion metadata does not cover every local replay-enabled router.")
    for owner in owners:
        owner._verl_replay_moe_layer_ids = tuple(sorted(global_ids))
        owner._verl_replay_hf_depth = hf_depth


def get_language_model(model):
    """Resolve distributed wrappers and the language component of multimodal models."""
    while hasattr(model, "module"):
        model = model.module
    return getattr(model, "language_model", model)


def validate_native_model(model, *, mtp=False):
    """Reject incompatible providers before starting a distributed training step."""
    language_model = get_language_model(model)
    required = {"output_processor", "output_processor_context"}
    if mtp:
        required.update({"compute_mtp_loss", "mtp_loss_normalization_factor"})
    missing = required - inspect.signature(language_model.forward).parameters.keys()
    if missing:
        raise ValueError(
            f"{type(language_model).__name__} lacks native Megatron APIs {sorted(missing)}. "
            "Install this branch's locked Core/Bridge dependencies and update custom providers."
        )


def logits_output_processor(
    *, hidden_states, output_layer, output_weight, runtime_gather_output, scale_logits, **kwargs
):
    """Return logits while native MTP independently attaches its auxiliary objective."""
    logits, _ = output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    return scale_logits(logits).transpose(0, 1).contiguous()
