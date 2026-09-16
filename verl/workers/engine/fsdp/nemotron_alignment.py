# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP1 verl adapter for the aligned Nemotron-H forward."""

import torch
from transformers import AutoModelForCausalLM
from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

from verl.models.transformers.nemotron_h_alignment import (
    _context_parallel_group,
    install_training_alignment,
)
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


class NemotronAlignmentEngine(FSDPEngineWithLMHead):
    def __init__(self, *args, expert_parallel_size=1, pipeline_parallel_size=1, **kwargs):
        self.expert_parallel_size = expert_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_group = None
        self.expert_parameters = set()
        super().__init__(*args, **kwargs)

    def _init_device_mesh(self):
        if self.pipeline_parallel_size == 1:
            return super()._init_device_mesh()
        from torch.distributed.device_mesh import init_device_mesh

        cp = self.engine_config.ulysses_sequence_parallel_size
        if (
            self.pipeline_parallel_size != 2
            or self.expert_parallel_size != 4
            or cp != 2
            or torch.distributed.get_world_size() != 8
        ):
            raise ValueError("Pipeline adapter requires PP2/EP4/CP2 on eight ranks")
        mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("pp", "dp", "sp"))
        self.pipeline_group = mesh["pp"].get_group()
        self.ulysses_device_mesh = mesh["dp", "sp"]
        self.device_mesh = self.ulysses_device_mesh._flatten("fsdp")
        self.ulysses_parallel_group = mesh["sp"].get_group()
        self.ulysses_sequence_parallel_size = cp
        self.use_ulysses_sp = True

    def get_data_parallel_size(self):
        return (
            torch.distributed.get_world_size()
            // self.pipeline_parallel_size
            // self.engine_config.ulysses_sequence_parallel_size
        )

    def _build_module(self):
        cp_size = self.engine_config.ulysses_sequence_parallel_size
        world_size = torch.distributed.get_world_size() // self.pipeline_parallel_size
        ep_size = self.expert_parallel_size
        if (
            cp_size not in (1, 2)
            or ep_size not in (1, 4)
            or world_size != (cp_size if ep_size == 1 else ep_size)
            or self.engine_config.strategy != "fsdp2"
        ):
            raise ValueError("Supported: FSDP2 TP1 CP1/2 with EP1/DP1 or EP4")
        if (
            self.model_config.use_liger
            or self.model_config.lora_rank
            or self.model_config.enable_gradient_checkpointing
            or self.model_config.enable_activation_offload
        ):
            raise ValueError("Liger, LoRA and activation checkpoint/offload are unverified")
        init_batch_invariance()
        model, info = AutoModelForCausalLM.from_pretrained(
            self.model_config.local_path,
            config=self.model_config.hf_config,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete Nemotron weight load: {info}")
        install_training_alignment(model)
        if self.pipeline_group is not None:
            stage = torch.distributed.get_rank(self.pipeline_group)
            if len(model.model.layers) % 2:
                raise ValueError("PP2 currently requires an even layer count")
            split = len(model.model.layers) // 2
            model.model.layers = model.model.layers[stage * split : (stage + 1) * split]
            model.model._alignment_pp_group = self.pipeline_group
            model._alignment_pp_group = self.pipeline_group
            if stage == 0:
                model.model.norm_f = None
                model.lm_head = None
            else:
                model.model.embeddings = None
        if ep_size > 1:
            from transformers.models.nemotron_h.modeling_nemotron_h import (
                NemotronHMoE,
            )

            ep_group = self.device_mesh.get_group()
            ep_rank = torch.distributed.get_rank(ep_group)
            for moe in model.modules():
                if not isinstance(moe, NemotronHMoE):
                    continue
                experts = moe.experts
                for name in ("up_proj", "down_proj"):
                    full = getattr(experts, name)
                    if full.shape[0] % ep_size:
                        raise ValueError("Expert count must divide EP size")
                    local = torch.nn.Parameter(
                        full.chunk(ep_size, dim=0)[ep_rank].clone(),
                        requires_grad=full.requires_grad,
                    )
                    # Match dense FSDP's world-averaged gradient convention.
                    local.register_hook(lambda grad: grad / world_size)
                    setattr(experts, name, local)
                    self.expert_parameters.add(local)
                experts._alignment_ep_group = ep_group
        if self.pipeline_group is not None:
            # Both pipeline stages receive and score the same final logprobs.
            for parameter in model.parameters():
                parameter.register_hook(lambda grad: grad / 2)
        return model

    def _build_fsdp_module(self, module):
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        # Keep HF's strict-FP32 routing bias and the BF16 weights unchanged.
        module.to(torch.device("cuda", torch.cuda.current_device()))
        policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
        for layer in module.model.layers:
            fully_shard(
                layer,
                mesh=self.device_mesh,
                reshard_after_forward=False,
                mp_policy=policy,
                ignored_params=self.expert_parameters,
            )
        fully_shard(
            module,
            mesh=self.device_mesh,
            reshard_after_forward=False,
            mp_policy=policy,
            ignored_params=self.expert_parameters,
        )
        return module

    def forward_step(self, *args, **kwargs):
        token = _context_parallel_group.set(self.ulysses_parallel_group)
        try:
            return super().forward_step(*args, **kwargs)
        finally:
            _context_parallel_group.reset(token)
