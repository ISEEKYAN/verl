# MyRMSNorm from TE is not consistent with apex version,
# configured by gptmodel_option, we use our implemented
# MyRMSNorm, MyDotProductAttention, MyMLP, MySelfAttention

from megatron.core.models.gpt.gpt_layer_specs import TransformerConfig, TransformerBlockSubmodules, get_gpt_layer_with_transformer_engine_spec, get_gpt_layer_local_spec, get_num_layers_to_build, get_transformer_layer_offset
from megatron.core.models.gpt.gpt_layer_specs import LNImpl
from megatron.core.models.gpt.gpt_layer_specs import *

from verl.models.llama.megatron.layers.parallel_rmsnorm import MyRMSNorm
from verl.models.llama.megatron.layers.parallel_attention import MyDotProductAttention
from verl.models.llama.megatron.layers.parallel_mlp import MyMLP
from verl.models.llama.megatron.layers.parallel_attention import MySelfAttention
from verl.utils.model import gptmodel_option

def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn('The fp8 argument in "_get_mlp_module_spec" has been deprecated'
                      ' and will be removed soon. Please update your code accordingly.')
    MLPClass = MLP if not gptmodel_option.my_mlp else MyMLP

    fc1Class = TELayerNormColumnParallelLinear if not gptmodel_option.seperate_rms_norm_mlp else TEColumnParallelLinear

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLPClass,
            submodules=MLPSubmodules(
                linear_fc1=fc1Class if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec(
            use_te=use_te,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )

def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn('The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
                      ' and will be removed soon. Please update your code accordingly.')

    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    assert multi_latent_attention == False, 'multi_latent_attention is not supported for now'
    if multi_latent_attention:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TEColumnParallelLinear,
                        linear_q_up_proj=(TELayerNormColumnParallelLinear if qk_layernorm else TEColumnParallelLinear),
                        linear_kv_down_proj=TEColumnParallelLinear,
                        linear_kv_up_proj=(TELayerNormColumnParallelLinear if qk_layernorm else TEColumnParallelLinear),
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:

        # TENorm significantly harms convergence when used
        # for QKLayerNorm if TE Version < 1.9;
        # we instead use the Apex implementation.
        norm_class = MyRMSNorm if gptmodel_option.my_rmsnorm else TENorm
        qk_norm = norm_class
        qkvClass = TELayerNormColumnParallelLinear if not gptmodel_option.seperate_rms_norm_attention else TEColumnParallelLinear
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=norm_class if gptmodel_option.seperate_rms_norm_attention else IdentityOp,
                self_attention=ModuleSpec(
                    module=MySelfAttention if gptmodel_option.my_self_attention else SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=qkvClass,
                        core_attention=MyDotProductAttention
                        if gptmodel_option.my_core_attention else TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        k_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=norm_class if gptmodel_option.seperate_rms_norm_mlp else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
            ),
        )


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
) -> TransformerBlockSubmodules:
    assert use_transformer_engine == True
    layer_norm_impl = MyRMSNorm if gptmodel_option.my_rmsnorm else TENorm
    # Layer specs.
    dense_layer_spec = (get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
    ))
    moe_layer_spec = (get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
    ))

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}")
    else:
        raise ValueError(f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}")

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset:offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=layer_norm_impl)

    return block_spec
