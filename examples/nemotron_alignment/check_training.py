"""Compare the real training engine with a frozen rollout; optionally test updates."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.fsdp.nemotron_alignment import NemotronAlignmentEngine


def checkpoint_state(state):
    """Restore the original Nemotron checkpoint names and per-expert tensors."""
    result = {}
    for name, tensor in state.items():
        name = re.sub(r"^model\.", "backbone.", name)
        match = re.fullmatch(r"(.*\.experts)\.(up_proj|down_proj)", name)
        if match:
            for expert, weight in enumerate(tensor.unbind(0)):
                result[f"{match[1]}.{expert}.{match[2]}.weight"] = weight.clone()
        else:
            result[name] = tensor
    return result


def flatten_pipeline_checkpoint(destination):
    """Expose stage shards at the root for vLLM's non-recursive weight glob."""
    index_path = destination / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    renames = {}
    for relative in set(index["weight_map"].values()):
        path = Path(relative)
        if len(path.parts) == 1:
            continue
        target = destination / f"{path.parent.name}-{path.name}"
        assert not target.exists(), target
        (destination / path).rename(target)
        renames[relative] = target.name
    index["weight_map"] = {k: renames.get(v, v) for k, v in index["weight_map"].items()}
    index_path.write_text(json.dumps(index, indent=2))


def main():
    cp_size = int(os.environ.get("NEMOTRON_CP", "1"))
    ep_size = int(os.environ.get("NEMOTRON_EP", "1"))
    pp_size = int(os.environ.get("NEMOTRON_PP", "1"))
    world_size = max(cp_size, ep_size) * pp_size
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    assert torch.cuda.device_count() == int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    with tempfile.TemporaryDirectory(prefix="nemotron-dist-") as directory:
        dist_args = (
            {"init_method": "env://"}
            if world_size > 1
            else {"init_method": f"file://{directory}/store", "rank": 0, "world_size": 1}
        )
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local_rank), **dist_args)
        assert torch.distributed.get_world_size() == world_size
        try:
            model_config = HFModelConfig(
                path=os.environ["NEMOTRON_MODEL"],
                enable_gradient_checkpointing=False,
                use_fused_kernels=True,
                override_config={"attn_implementation": "eager"},
            )
            config = FSDPEngineConfig(
                strategy="fsdp2",
                forward_only=True,
                model_dtype="bf16",
                use_torch_compile=False,
                reshard_after_forward=False,
                ulysses_sequence_parallel_size=cp_size,
            )
            engine = NemotronAlignmentEngine(
                model_config,
                config,
                FSDPOptimizerConfig(),
                CheckpointConfig(),
                expert_parallel_size=ep_size,
                pipeline_parallel_size=pp_size,
            )
            engine.initialize()
            assert engine.optimizer is None and engine.lr_scheduler is None
            print("REAL_ENGINE_INITIALIZED", type(engine.module).__name__, flush=True)
            if pp_size > 1:
                print(
                    "REAL_ENGINE_PP_STAGE",
                    torch.distributed.get_rank(),
                    len(engine.module.model.layers),
                    "embedding",
                    engine.module.model.embeddings is not None,
                    "head",
                    engine.module.lm_head is not None,
                    flush=True,
                )
                assert len(engine.module.model.layers) == 26
            if ep_size > 1:
                assert engine.expert_parameters
                for parameter in engine.expert_parameters:
                    assert not hasattr(parameter, "to_local")
                    assert parameter.shape[0] == 128 // ep_size
                print("REAL_ENGINE_EP_SHARDS", torch.distributed.get_rank(), len(engine.expert_parameters), flush=True)
            rows = json.loads(Path(os.environ["NEMOTRON_REFERENCE"]).read_text())
            cases = [
                ("single", rows, 1, False),
                ("packed", rows, 4, False),
                ("reversed", rows[::-1], 4, False),
                ("batch32", rows * 8, 32, False),
                ("dynamic32", rows[::-1] * 8, 32, True),
            ]
            for name, batch, micro_size, dynamic in cases:
                sequences = [torch.tensor(r["prompt"] + r["tokens"], device="cuda") for r in batch]
                ids = torch.nested.nested_tensor(sequences, layout=torch.jagged)
                positions = torch.nested.nested_tensor_from_jagged(
                    torch.cat([torch.arange(len(s), device="cuda") for s in sequences]), ids.offsets()
                )
                masks = []
                for row, seq in zip(batch, sequences, strict=False):
                    mask = torch.zeros(len(seq), device="cuda")
                    mask[len(row["prompt"]) - 1 : -1] = 1
                    masks.append(mask)
                data = TensorDict(
                    {
                        "input_ids": ids,
                        "position_ids": positions,
                        "loss_mask": torch.nested.nested_tensor(masks, layout=torch.jagged),
                    },
                    [len(batch)],
                )
                tu.assign_non_tensor(
                    data,
                    temperature=1.0,
                    use_fused_kernels=True,
                    use_dynamic_bsz=dynamic,
                    max_token_len_per_gpu=max(1024, (max(map(len, sequences)) + cp_size - 1) // cp_size),
                    micro_batch_size_per_gpu=micro_size,
                    return_model_output=True,
                )
                with engine.eval_mode():
                    output = engine.infer_batch(data)["model_output"]["log_probs"]
                assert len(output.unbind()) == len(batch)
                for row, values in zip(batch, output.unbind(), strict=False):
                    actual = values[len(row["prompt"]) - 1 : -1]
                    reference = torch.tensor(row["rollout_logprobs"], device="cuda")
                    delta = (actual - reference).abs()
                    assert actual.shape == reference.shape
                    print(
                        "ENGINE_OLD_LOGPROBS",
                        name,
                        len(row["prompt"]),
                        int((delta != 0).sum()),
                        float(delta.max()),
                        flush=True,
                    )
                    assert torch.equal(actual, reference)
            print("NEMOTRON7_REAL_ENGINE_EXACT", flush=True)
            if os.environ.get("NEMOTRON_BACKWARD_SMOKE") == "1":
                engine.module.train()
                sequences = [
                    torch.tensor(row["prompt"][:length], device="cuda")
                    for row, length in zip(rows, (17, 19), strict=False)
                ]
                packed = torch.cat(sequences)[None]
                positions = torch.cat([torch.arange(len(s), device="cuda") for s in sequences])[None]
                labels = packed.roll(-1, dims=1)
                mask = torch.ones_like(packed, dtype=torch.bool)
                mask[0, 16] = mask[0, -1] = False
                from verl.models.transformers.nemotron_h_alignment import (
                    _context_parallel_group,
                )

                rank = torch.distributed.get_rank()
                valid_tokens = mask.sum()
                if cp_size > 1:
                    cp_rank = torch.distributed.get_rank(engine.ulysses_parallel_group)
                    packed, positions, labels, mask = (
                        tensor.chunk(cp_size, dim=1)[cp_rank].contiguous()
                        for tensor in (packed, positions, labels, mask)
                    )
                token = _context_parallel_group.set(engine.ulysses_parallel_group)
                try:
                    output = engine.module(
                        input_ids=packed, position_ids=positions, shift_labels=labels, use_cache=False
                    )
                finally:
                    _context_parallel_group.reset(token)
                # FSDP averages across the CP ranks: restore global token mean.
                loss = -output.log_probs[mask].sum() * cp_size / valid_tokens
                assert torch.isfinite(loss)
                loss.backward()
                missing, nonfinite = [], []
                for name, parameter in engine.module.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if parameter.grad is None:
                        missing.append(name)
                    else:
                        grad = parameter.grad
                        local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
                        if not torch.isfinite(local_grad).all():
                            nonfinite.append(name)
                print("PACKED_BACKWARD", float(loss.detach()), "missing", missing, "nonfinite", nonfinite, flush=True)
                assert not missing and not nonfinite
                print("NEMOTRON7_PACKED_BACKWARD_FINITE", flush=True)
                if os.environ.get("NEMOTRON_EXPORT_UPDATE") == "1":
                    from torch.distributed.tensor import DTensor

                    destination = Path(os.environ["NEMOTRON_UPDATE_PATH"])
                    assert not destination.exists(), destination

                    def local_tensor(p):
                        return p.to_local() if isinstance(p, DTensor) else p

                    before = {
                        name: local_tensor(p).detach().cpu().clone() for name, p in engine.module.named_parameters()
                    }
                    optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.001, foreach=False)
                    optimizer.step()
                    changed = sum(
                        int((local_tensor(p).detach().cpu() != before[name]).sum())
                        for name, p in engine.module.named_parameters()
                    )
                    del before
                    changed = torch.tensor(changed, device="cuda", dtype=torch.int64)
                    torch.distributed.all_reduce(changed)
                    assert changed.item() > 0
                    optimizer.zero_grad(set_to_none=True)
                    stage_id = torch.distributed.get_rank(engine.pipeline_group) if pp_size > 1 else 0
                    stage_group = engine.device_mesh.get_group()
                    stage_leader = torch.distributed.get_rank(stage_group) == 0
                    state = {}
                    for name, p in engine.module.state_dict().items():
                        full = p.full_tensor() if isinstance(p, DTensor) else p
                        if ep_size > 1 and name.endswith(("experts.up_proj", "experts.down_proj")):
                            assert not isinstance(p, DTensor) and p.shape[0] == 128 // ep_size
                            parts = [torch.empty_like(p) for _ in range(ep_size)]
                            torch.distributed.all_gather(parts, p, group=stage_group)
                            full = torch.cat(parts)
                            del parts
                        assert torch.isfinite(full).all(), name
                        if stage_leader:
                            if pp_size > 1 and name.startswith("model.layers."):
                                fields = name.split(".")
                                fields[2] = str(int(fields[2]) + stage_id * len(engine.module.model.layers))
                                name = ".".join(fields)
                            state[name] = full.detach().cpu().contiguous()
                    del full
                    if rank == 0:
                        destination.mkdir()
                    torch.distributed.barrier()
                    if stage_leader:
                        from huggingface_hub import save_torch_state_dict

                        stage = tempfile.mkdtemp(prefix="nemotron-export-")
                        print("LOCAL_EXPORT_STAGE", stage, flush=True)
                        engine.module.config.save_pretrained(stage)
                        save_torch_state_dict(checkpoint_state(state), stage, max_shard_size="2GB")
                        part = destination / f"part{stage_id}" if pp_size > 1 else destination
                        part.mkdir(exist_ok=True)
                        subprocess.run(["cp", "--reflink=never", "-r", stage + "/.", str(part)], check=True)
                        shutil.rmtree(stage)
                    torch.distributed.barrier()
                    if rank == 0 and pp_size > 1:
                        weight_map, total_size = {}, 0
                        for i in range(pp_size):
                            part = destination / f"part{i}"
                            index = json.loads((part / "model.safetensors.index.json").read_text())
                            assert not weight_map.keys() & index["weight_map"].keys()
                            weight_map.update({k: f"part{i}/{v}" for k, v in index["weight_map"].items()})
                            total_size += index["metadata"]["total_size"]
                        source_index = json.loads(
                            (Path(model_config.local_path) / "model.safetensors.index.json").read_text()
                        )
                        assert weight_map.keys() == source_index["weight_map"].keys()
                        (destination / "model.safetensors.index.json").write_text(
                            json.dumps(
                                {
                                    "metadata": {"total_size": total_size},
                                    "weight_map": weight_map,
                                },
                                indent=2,
                            )
                        )
                        shutil.copy2(destination / "part0/config.json", destination / "config.json")
                        flatten_pipeline_checkpoint(destination)
                    for filename in (
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "LICENSE",
                    ):
                        if rank == 0:
                            shutil.copy2(Path(model_config.local_path) / filename, destination / filename)
                    if rank == 0:
                        (destination / "update-receipt.json").write_text(
                            json.dumps(
                                {
                                    "source": model_config.local_path,
                                    "optimizer": "SGD",
                                    "lr": 0.001,
                                    "steps": 1,
                                    "cp": cp_size,
                                    "ep": ep_size,
                                    "pp": pp_size,
                                    "tp": 1,
                                    "changed_elements": changed.item(),
                                    "tensor_count": len(weight_map) if pp_size > 1 else len(state),
                                },
                                indent=2,
                            )
                        )
                    torch.distributed.barrier()
                    print("UPDATED_WEIGHTS_EXPORTED", str(destination), "changed_elements", changed.item(), flush=True)
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
