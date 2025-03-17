# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaMA model with Megatron-style acceleration."""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from megatron.core import tensor_parallel
from megatron.core import ModelParallelConfig
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast

from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
from .layers import ParallelLlamaDecoderLayer, ParallelLlamaRMSNorm, ParallelLlamaDecoderLayerRmPad
"""
TODO: 
1. Add weight initialization. Here we need to be careful on TP weight init.
2. Add sequence parallel
3. Load checkpoint from meta LLama pretrained checkpoint
"""


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class ParallelLlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                   embedding_dim=config.hidden_size,
                                                                   **embedding_kwargs)

        self.layers = nn.ModuleList(
            [ParallelLlamaDecoderLayer(config, megatron_config) for _ in range(config.num_hidden_layers)])
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                              tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Args:
            input_ids: input ids. shape (batch_size, seq_length)
            attention_mask: attention_mask. shape (batch_size, seq_length)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        # embed positions

        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds)

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelLlamaForCausalLM(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.model = ParallelLlamaModel(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)

        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=config.hidden_size,
                                                            output_size=config.vocab_size,
                                                            bias=False,
                                                            gather_output=False,
                                                            skip_bias_add=False,
                                                            **column_kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)[0]

        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

        logits = logits.float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


class ParallelLlamaModelRmPad(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        self.megatron_config = megatron_config
        if megatron_config is not None:
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                   embedding_dim=config.hidden_size,
                                                                   **embedding_kwargs)

        self.layers = nn.ModuleList(
            [ParallelLlamaDecoderLayerRmPad(config, megatron_config) for _ in range(config.num_hidden_layers)])
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                sequence_length: int = None,
                indices: torch.Tensor = None,
                cu_seqlens: int = None,
                max_seqlen_in_batch: int = None) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

        # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
        inputs_embeds = inputs_embeds.transpose(0, 1)
        if self.megatron_config.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states,
                                          position_ids=position_ids,
                                          sequence_length=sequence_length,
                                          indices=indices,
                                          cu_seqlens=cu_seqlens,
                                          max_seqlen_in_batch=max_seqlen_in_batch)

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelLlamaForCausalLMRmPad(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.model = ParallelLlamaModelRmPad(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size
        self._init_head()

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.config.hidden_size,
                                                            output_size=self.config.vocab_size,
                                                            bias=False,
                                                            gather_output=False,
                                                            skip_bias_add=False,
                                                            **column_kwargs)

    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        logits = self.lm_head(hidden_states)[0]
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)  # (total_nnz_padded, 1, vocab_size)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""
        batch_size, sequence_length = input_ids.shape

        # remove padding here
        input_ids, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                              attention_mask)  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        if self.megatron_config.sequence_parallel:
            input_ids = sp_utils.pad_to_sequence_parallel(input_ids)

        input_ids = input_ids.transpose(0, 1)  # (1, total_nnz+pad)

        outputs = self.model(input_ids=input_ids,
                             position_ids=position_ids,
                             sequence_length=sequence_length,
                             indices=indices,
                             cu_seqlens=cu_seqlens,
                             max_seqlen_in_batch=max_seqlen_in_batch)

        hidden_states = outputs

        logits = self._forward_head(hidden_states)

        # remove padding from sequence parallel
        if self.megatron_config.sequence_parallel:
            totol_nnz = cu_seqlens[-1]
            logits = logits[:totol_nnz]  # (total_nnz_padded)

        logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension
        # add removed padding back
        logits = pad_input(logits, indices, batch_size,
                           seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class ParallelLlamaForValueRmPad(ParallelLlamaForCausalLMRmPad):

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(in_features=self.config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output = super().forward(input_ids, attention_mask, position_ids)
        output.logits = torch.squeeze(output.logits, dim=-1)
        return output


"""
Support pipeline parallelism
"""


class ParallelLlamaModelRmPadPP(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    This model definition supports pipeline parallelism. To support pp and vpp,
    - This model only contains layer in this pp stage and vpp chunk
    - When calling get_model in Megatron, this rank will instantiate all the vpp chunks in this pp.
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, pre_process, post_process):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.megatron_config = megatron_config
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        if pre_process:
            self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                       embedding_dim=config.hidden_size,
                                                                       **embedding_kwargs)
        else:
            self.embed_tokens = None

        # pp_rank = megatron_config.pipeline_model_parallel_rank
        pp_size = megatron_config.pipeline_model_parallel_size
        self.num_layer_per_pp = config.num_hidden_layers // pp_size
        vpp_size = megatron_config.virtual_pipeline_model_parallel_size

        if vpp_size is not None:
            self.num_layer_vpp_chunk = self.num_layer_per_pp // vpp_size
            self.num_layer_this_model = self.num_layer_vpp_chunk
            # vpp_rank = megatron_config.virtual_pipeline_model_parallel_rank
            # self.offset = vpp_rank * (
            #         config.num_hidden_layers // megatron_config.virtual_pipeline_model_parallel_size) + \
            #             (megatron_config.pipeline_model_parallel_rank * self.num_layer_vpp_chunk)
        else:
            self.num_layer_this_model = self.num_layer_per_pp
            # self.offset = pp_rank * self.num_layer_per_pp

        layers = []
        for i in range(self.num_layer_this_model):
            layer = ParallelLlamaDecoderLayerRmPad(config, megatron_config)
            # setattr(layer, 'hidden_layer_index', self.offset + i)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if post_process:
            self.norm = ParallelLlamaRMSNorm(config, megatron_config)
        else:
            self.norm = None

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: Optional[torch.LongTensor] = None,
                sequence_length: int = None,
                indices: torch.Tensor = None,
                cu_seqlens: int = None,
                max_seqlen_in_batch: int = None) -> Union[Tuple, BaseModelOutputWithPast]:
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        if self.pre_process:
            inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

            # vocab parallel embedding will not do sequence parallel reduce-scatter in open source megatron
            # so need to deal with it by handle here:
            # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
            inputs_embeds = inputs_embeds.transpose(0, 1)
            if self.megatron_config.sequence_parallel:
                inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

            hidden_states = inputs_embeds
        else:
            # self.hidden_states should be passed by Megatron
            hidden_states = self.input_tensor

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states,
                                          position_ids=position_ids,
                                          sequence_length=sequence_length,
                                          indices=indices,
                                          cu_seqlens=cu_seqlens,
                                          max_seqlen_in_batch=max_seqlen_in_batch)

            hidden_states = layer_outputs

        if self.post_process:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class ParallelLlamaForCausalLMRmPadPP(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, pre_process, post_process,
                 share_embeddings_and_output_weights):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.model = ParallelLlamaModelRmPadPP(config,
                                               megatron_config=megatron_config,
                                               pre_process=pre_process,
                                               post_process=post_process)
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        if post_process:
            self._init_head()

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        assert len(input_tensor) == 1
        self.model.set_input_tensor(input_tensor[0])

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.config.hidden_size,
                                                            output_size=self.config.vocab_size,
                                                            bias=False,
                                                            gather_output=False,
                                                            skip_bias_add=False,
                                                            **column_kwargs)

    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        # logits shape before forward_head hidden_states.shape: [4, 32, 4096]
        logits = self.lm_head(hidden_states)[0]
        # logits shape after forward_head logits.shape: [8, 32, 8]
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        return logits

    def forward(
        self,
        # original input
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # Note that input_ids, attention_mask and position_ids should be passed to every pp layer.
        # In the first pp, input_ids will be used, in other pp layers hidden_states will be used inside self.model
        batch_size, sequence_length = input_ids.shape
        # remove padding here
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                                    attention_mask)  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        if self.megatron_config.sequence_parallel:
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)

        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz+pad)

        outputs = self.model(input_ids=input_ids_rmpad,
                             position_ids=position_ids,
                             sequence_length=sequence_length,
                             indices=indices,
                             cu_seqlens=cu_seqlens,
                             max_seqlen_in_batch=max_seqlen_in_batch)

        if self.post_process:
            hidden_states = outputs
            # print(f'hidden_states.shape = {hidden_states.shape}') # torch.Size([4, 32, 4096])
            logits = self._forward_head(hidden_states)
            logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension # torch.Size([8, 32, 16])

            # remove padding from sequence parallel
            if self.megatron_config.sequence_parallel:
                totol_nnz = cu_seqlens[-1]
                logits = logits[:totol_nnz]  # (total_nnz_padded)
            # add removed padding back. If input is already rmpad, we let the caller pad_input
            logits = pad_input(logits, indices, batch_size,
                               seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:
            return outputs


class ParallelLlamaForValueRmPadPP(ParallelLlamaForCausalLMRmPadPP):

    def _init_head(self):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(in_features=self.config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits

    def forward(
        self,
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.post_process:
            output.logits = torch.squeeze(output.logits, dim=-1)
            return output
        else:
            return output


def gptmodel_forward(model, input_ids, attention_mask, position_ids, sequence_parallel, pack_seqs=False):
    if pack_seqs:
        from flash_attn.bert_padding import pad_input, unpad_input  # noqa
        from megatron.core.packed_seq_params import PackedSeqParams
        import copy
        batch_size, sequence_length = input_ids.shape
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                                    attention_mask)
        cu_seqlens_padded = None
        if sequence_parallel:
            original_total_nnz = input_ids_rmpad.shape[0]
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)
            total_nnz_new = input_ids_rmpad.shape[0]
            pad_size = total_nnz_new - original_total_nnz
            if pad_size > 0:
                cu_seqlens_padded = copy.deepcopy(cu_seqlens)
                cu_seqlens_padded[-1] = cu_seqlens_padded[-1] + pad_size
                seqlens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
                max_seqlen_in_batch = seqlens.max().cpu().item()

        packed_seq_params = PackedSeqParams(qkv_format='thd',
                                            cu_seqlens_q=cu_seqlens,
                                            max_seqlen_q=max_seqlen_in_batch,
                                            cu_seqlens_kv=cu_seqlens,
                                            max_seqlen_kv=max_seqlen_in_batch,
                                            cu_seqlens_q_padded=cu_seqlens_padded,
                                            cu_seqlens_kv_padded=cu_seqlens_padded)
        output = model(input_ids=input_ids_rmpad.T,
                       attention_mask=None,
                       position_ids=position_ids,
                       packed_seq_params=packed_seq_params)

        output = torch.squeeze(output, dim=0)  # (1,seq,1)->(seq,1)

        if sequence_parallel and pad_size > 0:
            output = output[:cu_seqlens[-1]]

        output = pad_input(output, indices, batch_size, seqlen=sequence_length)[..., 0]
    else:
        # output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        batch_size, sequence_length = input_ids.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids,sequence_parallel)
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length)

    return output

def remove_left_padding(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,sequence_parallel: bool=False):

    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2
    
    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape) # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()
    if sequence_parallel: 
        from megatron.core import parallel_state as mpu
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        seq_len = seq_len + pad_size
    shape[1] = seq_len
    
    new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, seq_len))
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        new_input_ids[i, :seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, :seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, :seq_lens[i]] = position_ids[i, attention_mask[i]]
    return new_input_ids, new_attention_mask, new_position_ids

def recover_left_padding(result, attention_mask: torch.Tensor, original_attention_mask: torch.Tensor, origin_seqlen: int):
    """
    Recover left padding from result
    return result
    """
    shape = list(result.shape)
    batch_size = shape[0]
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    return new_result

