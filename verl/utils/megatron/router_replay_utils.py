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

"""
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

from contextlib import contextmanager

import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

from verl.models.mcore.util import (
    get_thd_quant_align_size,
    postprocess_bshd_engine,
    postprocess_packed_seqs,
    postprocess_thd_engine,
    preprocess_bshd_engine,
    preprocess_packed_seqs,
    preprocess_thd_engine,
)
from verl.utils.device import get_device_name

device_name = get_device_name()


def is_moe_layer(tf_config, layer_idx):
    moe_layer_freq = getattr(tf_config, "moe_layer_freq", None)

    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    else:
        raise ValueError(f"Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def _context_parallel_layout(tf_config) -> str:
    """Return the CP row layout used for THD packing, matching MegatronEngine.

    Router-replay tensors must be sharded exactly like the model's own token rows; otherwise the
    replayed routes are attached to the wrong tokens (or the shapes disagree outright, since the
    zigzag layout doubles the alignment). Keep this in sync with
    ``MegatronEngine._get_context_parallel_layout``.
    """
    if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid":
        return "contiguous"
    return "zigzag"


def merge_router_topk_indices(
    attention_mask,
    input_ids,
    mini_layer_topk_idx_list,
    tf_config,
    vp_rank=None,
    local_cp_size=None,
    model=None,
    data_format="thd",
    forced_max_seqlen=None,
):
    """
    Merge recorded router top-k indices across sequence-parallel ranks for all router instances,
    then pack/unpack them to align with the original (batch, seq_len) layout and append the result.

    Args:
        attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]. Used to determine
            the valid token positions during pack/unpack.
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]. Used together with
            attention_mask for sequence packing/unpacking.
        mini_layer_topk_idx_list (list): A Python list to which the merged top-k indices tensor will be appended.
        tf_config: Megatron/Transformer engine configuration object. Used to locate router instances for
            the current micro-batch.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.
        model: The forwarded model chunk. When given, routes are collected from its own routers instead
            of inferring their location in the process-global router registry.

    Returns:
        None: The function has side effects only; it appends a tensor of shape
        [1, dynamic_bs_all, layer_num, topk] to mini_layer_topk_idx_list.
    """
    with torch.no_grad():
        if not input_ids.is_nested and attention_mask is None:
            raise RuntimeError("router replay RECORD requires attention_mask for non-nested BSHD inputs.")
        vp_rank = 0 if vp_rank is None else vp_rank
        if model is None:
            raise ValueError("Router replay requires the forwarded model; global registry inference was removed.")
        model_routers = sorted(iter_model_routers(model), key=lambda item: item[0])
        layer_numbers = [layer_number for layer_number, _ in model_routers]
        if len(layer_numbers) != len(set(layer_numbers)):
            raise RuntimeError("Router replay found duplicate HF layer numbers in the forwarded model.")
        router_instances_list = [router for _, router in model_routers]

        if not router_instances_list:
            topk = getattr(tf_config, "moe_router_topk", None)
            if not isinstance(topk, int) or topk <= 0:
                raise RuntimeError(
                    "router replay RECORD cannot build an empty local route map without a positive "
                    f"moe_router_topk, got {topk!r}."
                )
            if input_ids.is_nested:
                empty_routes = [
                    torch.empty((tokens.shape[0], 0, topk), dtype=torch.int16) for tokens in input_ids.unbind()
                ]
                mini_layer_topk_idx_list.append(torch.nested.as_nested_tensor(empty_routes, layout=torch.jagged))
            else:
                batch_size, seq_len = attention_mask.shape[:2]
                mini_layer_topk_idx_list.append(
                    torch.empty((batch_size, seq_len, 0, topk), dtype=torch.int16, device="cpu")
                )
            return

        recorded_local_positions = [
            idx for idx, router in enumerate(router_instances_list) if router.recorded_topk_idx is not None
        ]
        missing = [idx for idx, router in enumerate(router_instances_list) if router.recorded_topk_idx is None]
        if missing:
            local_rank_info = {"layer_numbers": layer_numbers}
            raise RuntimeError(
                "router replay RECORD did not capture all local routers: "
                f"missing_local_positions={missing}, selected={len(router_instances_list)}, "
                f"registry={len(RouterReplay.get_instances(is_mtp_layer=False))}, "
                f"recorded_local_positions={recorded_local_positions}, local_layer_range={local_rank_info}, "
                f"pp_rank={mpu.get_pipeline_model_parallel_rank()}, vp_rank={vp_rank}"
            )

        layers_topk_idx = []
        for router in router_instances_list:
            layers_topk_idx.append(router.recorded_topk_idx.to(torch.int16))  # dynamic_bs, topk

        # layer_num, dynamic_bs, topk  -> dynamic_bs, layer_num, topk
        layers_topk_idx = torch.stack(layers_topk_idx).permute(1, 0, 2).to(device_name)
        layers_topk_idx = layers_topk_idx.contiguous().view(torch.uint8)
        # dynamic_bs, layer_num, topk -> 1, dynamic_bs_all, layer_num, topk
        layers_topk_idx = (
            gather_from_sequence_parallel_region(layers_topk_idx, tensor_parallel_output_grad=False)
            .unsqueeze(0)
            .contiguous()
        )

        fp8 = tf_config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        cp_layout = _context_parallel_layout(tf_config)
        min_local_rows = (
            tf_config.csa_window_size
            if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid"
            else None
        )

        if data_format == "bshd":
            if not input_ids.is_nested:
                raise ValueError("BSHD router replay requires the engine's jagged input IDs.")
            _, layout_mask, _ = preprocess_bshd_engine(
                input_ids, pre_process=True, use_fp8_padding=use_fp8_padding, forced_max_seqlen=forced_max_seqlen
            )
            batch_size = input_ids.shape[0]
            routes = layers_topk_idx.squeeze(0)
            routes = routes.reshape(-1, batch_size, *routes.shape[1:]).transpose(0, 1).contiguous()
            layers_topk_idx = postprocess_bshd_engine(routes, layout_mask)
            layers_topk_idx = torch.nested.nested_tensor_from_jagged(
                layers_topk_idx.values().contiguous().view(torch.int16), offsets=layers_topk_idx.offsets()
            )
        elif input_ids.is_nested:
            batch_size = input_ids.shape[0]
            _, packed_seq_params, _ = preprocess_thd_engine(
                input_ids,
                pre_process=True,
                quant_align_size=get_thd_quant_align_size(tf_config),
                min_local_rows=min_local_rows,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            layers_topk_idx = postprocess_thd_engine(
                layers_topk_idx,
                packed_seq_params,
                input_ids,
                batch_size,
                post_process=True,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            # Undo the uint8 view. Nested tensors have no view(dtype), so reinterpret
            # the packed values buffer and re-wrap it on the same offsets.
            layers_topk_idx = torch.nested.nested_tensor_from_jagged(
                layers_topk_idx.values().contiguous().view(torch.int16), offsets=layers_topk_idx.offsets()
            )
        else:
            batch_size, seq_len = attention_mask.shape[:2]
            _, packed_seq_params = preprocess_packed_seqs(
                input_ids, attention_mask, pre_process=True, quant_align_size=get_thd_quant_align_size(tf_config)
            )
            layers_topk_idx = postprocess_packed_seqs(
                layers_topk_idx, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
            )
            layers_topk_idx = layers_topk_idx.view(torch.int16)
        mini_layer_topk_idx_list.append(layers_topk_idx.cpu())


def build_r3_replay_mask(input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Build a full-sequence replay mask for rollout-captured routes.

    Response logprobs are read from shifted model rows, but those rows attend to
    earlier hidden states whose MoE routing also affects the result. Replay every
    causal row that can influence response logprobs and skip only the final
    response row, whose logits are not used by ``no_padding_2_padding``.
    """
    total_lens = input_ids.offsets().diff()
    response_lens = response_mask.sum(dim=-1).to(device=total_lens.device, dtype=total_lens.dtype)
    bs = total_lens.size(0)
    values = torch.tensor([True, False], dtype=torch.bool, device=total_lens.device).repeat(bs)
    replay_lens = torch.where(response_lens > 0, total_lens - 1, torch.zeros_like(total_lens))
    suffix_lens = total_lens - replay_lens
    counts = torch.stack([replay_lens, suffix_lens], dim=1).flatten()
    mask_values = torch.repeat_interleave(values, counts)
    return torch.nested.nested_tensor_from_jagged(mask_values, offsets=input_ids.offsets())


def align_r3_router_replay_data(layers_topk_idx: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Align rollout routes with full training inputs one sequence at a time.

    Autoregressive rollout records the routes used to predict each generated
    token, so it has no route for the final generated token. The R3 replay mask
    leaves that final model row native; append one ignored placeholder per
    affected sequence so route and model tensors still have identical shapes.
    """
    if not layers_topk_idx.is_nested or not input_ids.is_nested:
        raise TypeError("R3 router replay requires jagged route targets and input_ids")

    route_parts = list(layers_topk_idx.unbind())
    input_lens = [int(length) for length in input_ids.offsets().diff().tolist()]
    if len(route_parts) != len(input_lens):
        raise RuntimeError(
            f"R3 router replay has {len(route_parts)} route sequences for {len(input_lens)} input sequences"
        )

    aligned_parts = []
    for sample_id, (routes, input_len) in enumerate(zip(route_parts, input_lens, strict=True)):
        route_len = routes.shape[0]
        if route_len == input_len:
            aligned_parts.append(routes)
        elif route_len == input_len - 1:
            placeholder = torch.zeros((1, *routes.shape[1:]), dtype=routes.dtype, device=routes.device)
            aligned_parts.append(torch.cat((routes, placeholder), dim=0))
        else:
            raise RuntimeError(
                f"R3 router replay sample {sample_id} has {route_len} route rows for {input_len} input tokens; "
                "expected equal lengths or exactly one missing final route"
            )

    return torch.nested.as_nested_tensor(aligned_parts, layout=torch.jagged)


def set_router_replay_data(
    layers_topk_idx,
    attention_mask,
    tf_config,
    vp_rank=None,
    replay_mask=None,
    local_cp_size=None,
    model=None,
    data_format="thd",
    forced_max_seqlen=None,
):
    """
    Scatter the packed router top-k indices back to sequence-parallel ranks and update each local
    RouterReplay instance with target indices for replay mode.

    This function prepares the per-layer, per-sample top-k routing decisions (recorded during an earlier
    forward) so that subsequent replay passes can follow exactly the same routing.

    Args:
        layers_topk_idx (torch.Tensor): Router top-k indices with shape [bs, max_seq_len, layer_num, topk].
            This should be the merged output produced by merge_router_topk_indices.
        attention_mask (torch.Tensor): Attention mask [batch_size, seq_len] used for pack/unpack alignment.
        tf_config: Megatron/Transformer engine configuration object.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.
        replay_mask (Optional[torch.Tensor]): Optional per-token mask. Masked tokens use replayed routes;
            unmasked tokens keep native Megatron routes.
        model: The forwarded model (or list of VPP chunks). When given, targets are written to its own
            routers instead of a positional slice of the global RouterReplay.get_instances(is_mtp_layer=False) list.

    Returns:
        None: The function updates internal RouterReplay instances in-place.
    """
    if layers_topk_idx is None:
        raise RuntimeError("router_replay REPLAY requires routed_experts from the preceding RECORD forward.")

    with torch.no_grad():
        vp_rank = 0 if vp_rank is None else vp_rank
        fp8 = tf_config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        cp_layout = _context_parallel_layout(tf_config)
        min_local_rows = (
            tf_config.csa_window_size
            if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid"
            else None
        )

        replay_mask_rmpad = None
        if data_format == "bshd":
            if local_cp_size is not None:
                raise ValueError("BSHD router replay does not support dynamic CP.")
            routes, _, _ = preprocess_bshd_engine(
                layers_topk_idx,
                pre_process=True,
                use_fp8_padding=use_fp8_padding,
                forced_max_seqlen=forced_max_seqlen,
            )
            layers_topk_idx_rmpad = routes.transpose(0, 1).reshape(1, -1, *routes.shape[2:])
            if replay_mask is not None:
                mask, _, _ = preprocess_bshd_engine(
                    replay_mask,
                    pre_process=True,
                    use_fp8_padding=use_fp8_padding,
                    forced_max_seqlen=forced_max_seqlen,
                )
                replay_mask_rmpad = mask.transpose(0, 1).reshape(1, -1)
        elif layers_topk_idx.is_nested:
            layers_topk_idx_rmpad, _, _ = preprocess_thd_engine(
                layers_topk_idx,
                pre_process=True,
                quant_align_size=get_thd_quant_align_size(tf_config),
                min_local_rows=min_local_rows,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            if replay_mask is not None:
                replay_mask_rmpad, _, _ = preprocess_thd_engine(
                    replay_mask,
                    pre_process=True,
                    quant_align_size=get_thd_quant_align_size(tf_config),
                    min_local_rows=min_local_rows,
                    local_cp_size=local_cp_size,
                    cp_layout=cp_layout,
                )
        else:
            if attention_mask is None:
                raise RuntimeError("router replay REPLAY requires attention_mask for non-nested BSHD inputs.")
            layers_topk_idx_rmpad, _ = preprocess_packed_seqs(
                layers_topk_idx, attention_mask, pre_process=True, quant_align_size=get_thd_quant_align_size(tf_config)
            )
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

        # 1, dynamic_bs_split, layer_num, topk
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.to(device_name).squeeze(dim=0)
        ).unsqueeze(dim=0)
        replay_mask_rmpad_split = None
        if replay_mask_rmpad is not None:
            replay_mask_rmpad_split = scatter_to_sequence_parallel_region(
                replay_mask_rmpad.to(device_name).squeeze(dim=0)
            )

        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk
        # When dim-0 covers all layers (e.g. R3, or R2 with all-MoE models),
        # index by absolute layer_idx; otherwise (R2 with mixed dense/MoE),
        # dim-0 only contains MoE layers, index by MoE-layer ordinal.

        if model is None:
            raise ValueError("Router replay requires the forwarded model; global registry inference was removed.")
        from verl.utils.megatron.native import get_language_model

        owner = get_language_model(model)
        all_moe_ids = getattr(owner, "_verl_replay_moe_layer_ids", None)
        hf_depth = getattr(owner, "_verl_replay_hf_depth", tf_config.num_layers)
        route_depth = layers_topk_idx_reshape.shape[0]
        for layer_number, router in iter_model_routers(model):
            layer_idx = layer_number - 1
            if route_depth == hf_depth:
                idx = layer_idx
            elif all_moe_ids is not None and route_depth == len(all_moe_ids):
                idx = all_moe_ids.index(layer_idx)
            elif all_moe_ids is None:
                idx = sum(1 for i in range(layer_idx) if is_moe_layer(tf_config, i))
            else:
                raise RuntimeError(f"Replay has {route_depth} layers; expected {hf_depth} or {len(all_moe_ids)}.")
            if not 0 <= idx < route_depth:
                raise RuntimeError(f"Replay data does not cover forwarded layer {layer_number}.")
            router.set_target_indices(layers_topk_idx_reshape[idx].long(), replay_mask=replay_mask_rmpad_split)


def iter_model_routers(model):
    """Yield ``(layer_number, RouterReplay)`` for every replay-enabled router owned by ``model``.

    ``model`` is a single module or the list of virtual-pipeline chunks. The process-global
    ``RouterReplay.get_instances(is_mtp_layer=False)`` list can hold more routers than this model has local layers —
    a model that rebuilds its decoder registers a set it then discards — so a positional slice of it
    can address the wrong objects. Walking the forwarded model reaches each layer's own router.
    """
    from verl.utils.megatron.native import get_language_model

    for chunk in model if isinstance(model, list | tuple) else [model]:
        chunk = get_language_model(chunk)
        # Scope to the decoder: MTP layers restart layer numbering at 1, so they alias decoder
        # layers, and the replay tensor carries no MTP rows. ``mtp`` is a sibling of ``decoder``
        # on both GPTModel and HybridModel, the only two classes that build one.
        for module in getattr(chunk, "decoder", chunk).modules():
            router = getattr(module, "router_replay", None)
            if isinstance(module, TopKRouter) and router is not None and module.layer_number is not None:
                yield getattr(module, "_verl_router_layer_number", module.layer_number), router


def set_model_router_replay_action(model, router_replay_action):
    """Set the router replay action on the forwarded model's own routers.

    Mirrors the module walk in ``set_router_replay_data``. Toggling through the positional slice can
    miss the real routers, leaving them stuck in REPLAY_FORWARD so the backward recompute reads a
    stale cross-micro-batch ``target_topk_idx`` instead of popping its own entry from
    ``replay_backward_list``.
    """
    for _, router in iter_model_routers(model):
        router.set_router_replay_action(router_replay_action)


def reorder_and_merge_vpp_layers(
    micro_batch_tensor_list,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    """
    Reorder and merge per-VPP layer blocks into a contiguous layer dimension.

    Given a tensor shaped as [bs*vpp_size, max_token_len, layer_num_per_vpp, topk], this function:
    1) Builds the schedule table for virtual microbatches and reorders the first dimension so that entries
       belonging to the same model chunk (VPP stage) become contiguous.
    2) Reshapes and merges the (vpp_size, layer_num_per_vpp) into a single layer dimension, producing
       [bs, max_token_len, layer_num, topk].

    Args:
        micro_batch_tensor_list : the list of Input tensor.
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.

    Returns:
        torch.Tensor: Output tensor of shape [bs, max_token_len, layer_num, topk].

    Raises:
        ValueError: If input tensor dimensionality or expected sizes do not match.
        RuntimeError: If the computed output shape is unexpected or the schedule length mismatches.
    """
    # 1) Build schedule table: map each virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)
    if len(micro_batch_tensor_list) != len(schedule_table):
        raise RuntimeError(
            "R2 RECORD must capture every scheduled microbatch/chunk, including zero-layer chunks: "
            f"captured={len(micro_batch_tensor_list)}, expected={len(schedule_table)}"
        )

    # 2) Group by model_chunk_id to build reorder indices so entries of the same chunk become contiguous along dim 0
    tensor_by_chunk = [[] for _ in range(vpp_size)]
    mini_tensor_list = []

    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        tensor_by_chunk[chunk_id].append(micro_batch_tensor_list[vidx])

    if micro_batch_tensor_list[0].is_nested:
        for chunk_id in range(vpp_size):
            mini_tensor_list.append(merge_nested_router_maps(tensor_by_chunk[chunk_id]))
    else:
        for chunk_id in range(vpp_size):
            mini_tensor_list.append(torch.cat(tensor_by_chunk[chunk_id], dim=0))

    out = torch.cat(mini_tensor_list, dim=2)

    return out


def merge_nested_router_maps(router_maps) -> torch.Tensor:
    """Flatten recorded micro-batch routes into one int16 jagged tensor."""
    tensors = [tensor.to(torch.int16) for router_map in router_maps for tensor in router_map.unbind()]
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def pp_gather(local_layers_router_map, tf_config, model):
    """
    Gather local router maps from all PP ranks into a global router map.

    Args:
        local_layers_router_map (torch.Tensor): Local router map of shape
            [bs, max_seq_len, local_num_layers, topk].
        tf_config: Configuration providing pipeline_model_parallel_size.

    Returns:
        torch.Tensor: Global router map of shape [bs, max_seq_len, num_layers, topk] placed on CPU.
    """
    pp_size = tf_config.pipeline_model_parallel_size
    local_ids = [layer for chunk in model for layer, _ in sorted(iter_model_routers(chunk), key=lambda item: item[0])]
    if pp_size <= 1:
        order = torch.tensor(sorted(range(len(local_ids)), key=local_ids.__getitem__), dtype=torch.long)
        return (
            torch.nested.as_nested_tensor(
                [row.index_select(1, order.to(row.device)) for row in local_layers_router_map.unbind()],
                layout=torch.jagged,
            )
            if local_layers_router_map.is_nested
            else local_layers_router_map.index_select(2, order)
        )

    pp_group = mpu.get_pipeline_model_parallel_group()
    world_size = torch.distributed.get_world_size(pp_group)
    if world_size != pp_size:
        raise RuntimeError(f"pipeline group size {world_size} does not match configured PP size {pp_size}.")
    if local_layers_router_map.is_nested:
        # all_gather_object preserves each sender's CUDA device, so normalize
        # jagged route maps before concatenating routes from local PP ranks.
        local_layers_router_map = local_layers_router_map.to("cpu")
        layers_topk_idx_global_list = [None] * world_size
        torch.distributed.all_gather_object(layers_topk_idx_global_list, local_layers_router_map, pp_group)
    else:
        # PP stages may own different numbers of MoE layers, including zero. Exchange
        # layer counts, pad only that dimension for NCCL, then remove the padding.
        # NCCL has no int16 datatype, so payloads travel as bit-identical uint8 views.
        payload = local_layers_router_map.to(device_name).contiguous().view(torch.uint8)
        local_layer_count = torch.tensor([local_layers_router_map.shape[2]], dtype=torch.int64, device=payload.device)
        gathered_layer_counts = [torch.empty_like(local_layer_count) for _ in range(world_size)]
        torch.distributed.all_gather(
            tensor=local_layer_count,
            tensor_list=gathered_layer_counts,
            group=pp_group,
            async_op=False,
        )
        layer_counts = [int(count.item()) for count in gathered_layer_counts]
        max_layer_count = max(layer_counts)
        if max_layer_count == 0:
            layers_topk_idx_global_list = [local_layers_router_map for _ in range(world_size)]
        else:
            if payload.shape[2] != max_layer_count:
                padded_shape = list(payload.shape)
                padded_shape[2] = max_layer_count
                padded_payload = torch.zeros(padded_shape, dtype=payload.dtype, device=payload.device)
                padded_payload[:, :, : payload.shape[2], :].copy_(payload)
                payload = padded_payload
            gathered_payloads = [torch.empty_like(payload) for _ in range(world_size)]
            torch.distributed.all_gather(
                tensor=payload,
                tensor_list=gathered_payloads,
                group=pp_group,
                async_op=False,
            )
            layers_topk_idx_global_list = [
                tensor[:, :, :layer_count, :].contiguous().view(torch.int16)
                for tensor, layer_count in zip(gathered_payloads, layer_counts, strict=True)
            ]
    ids_by_rank = [None] * pp_size
    torch.distributed.all_gather_object(ids_by_rank, local_ids, pp_group)
    all_ids = [layer for rank_ids in ids_by_rank for layer in rank_ids]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Router replay found duplicate HF layer IDs across pipeline stages.")
    order = torch.tensor(sorted(range(len(all_ids)), key=all_ids.__getitem__), dtype=torch.long)
    global_map = torch.cat(layers_topk_idx_global_list, dim=2).cpu()
    if global_map.is_nested:
        return torch.nested.as_nested_tensor(
            [row.index_select(1, order) for row in global_map.unbind()], layout=torch.jagged
        )
    return global_map.index_select(2, order)


class RouterReplayHelper:
    """Query replay actions only on the model being forwarded."""

    @staticmethod
    def _has_action(model, action):
        if model is None:
            raise ValueError("Router replay requires a model instance.")
        return any(router.router_replay_action == action for _, router in iter_model_routers(model))

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank=None, model=None):
        return RouterReplayHelper._has_action(model, RouterReplayAction.RECORD)

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None, model=None):
        return RouterReplayHelper._has_action(model, RouterReplayAction.REPLAY_FORWARD)

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None, model=None):
        return RouterReplayHelper._has_action(model, RouterReplayAction.REPLAY_BACKWARD)


@contextmanager
def router_replay_scope(models, action=None):
    """Own replay state for one schedule invocation, including exception cleanup."""
    if action is None:
        yield
        return
    routers = [router for _, router in iter_model_routers(models)]
    for router in routers:
        router.set_router_replay_action(action)
    try:
        yield
    finally:
        for router in routers:
            router.clear_indices()
            router.clear_router_replay_action()
