#!/usr/bin/env bash
# GRPO scale demo | DeepSeek-V4 | vLLM rollout | Megatron Lite training | GPU
#
# Megatron Lite's mainline target is Megatron-LM's dev branch, while active
# development happens on https://github.com/ISEEKYAN/mlite before upstreaming.
# That checkout provides both megatron.lite and the verl_mlite backend glue:
#
#   git clone https://github.com/ISEEKYAN/mlite
#   pip install -e mlite/experimental/lite/examples/verl
#
# DeepSeek-V4 uses fused DSA kernels on H100. The critical DSA-only dependencies
# are nvidia-cutlass-dsl==4.5.2 and a develop-branch nvidia-cudnn-frontend build
# with IndexerForwardSm90 support; release 1.24.1 is not sufficient.
#
# OPTIMIZER selects the Megatron Lite optimizer path:
#   - dist_opt: vanilla Megatron distributed optimizer
#   - fsdp2: Megatron Lite FSDP2 wrapper, lower memory pressure, default

set -xeuo pipefail

########################### user-adjustable ###########################
MLITE_ROOT=${MLITE_ROOT:-$HOME/mlite}
MLITE_VERL_ROOT=${MLITE_VERL_ROOT:-${MLITE_ROOT}/experimental/lite/examples/verl}
MLITE_LITE_ROOT=${MLITE_LITE_ROOT:-${MLITE_ROOT}/experimental/lite}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4 HF checkpoint}

NNODES=${NNODES:-16}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-4}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_CP=${ACTOR_CP:-4}
ACTOR_ETP=${ACTOR_ETP:-1}
OPTIMIZER=${OPTIMIZER:-fsdp2}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-16}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROJECT_NAME=${PROJECT_NAME:-verl-mlite-deepseek_v4-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_v4_grpo_${OPTIMIZER}}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
export PYTHONPATH="${MLITE_VERL_ROOT}:${MLITE_LITE_ROOT}:${MLITE_ROOT}:${VERL_ROOT:-}:${MEGATRON_ROOT:-}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES
fi

########################### parameter arrays ###########################
ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.truncation=error
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor@actor_rollout_ref.actor=mlite_actor
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.engine.tp=${ACTOR_TP}
    actor_rollout_ref.actor.engine.pp=${ACTOR_PP}
    actor_rollout_ref.actor.engine.vpp=1
    actor_rollout_ref.actor.engine.ep=${ACTOR_EP}
    actor_rollout_ref.actor.engine.cp=${ACTOR_CP}
    actor_rollout_ref.actor.engine.etp=${ACTOR_ETP}
    actor_rollout_ref.actor.engine.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.attention_backend_override=flash
    actor_rollout_ref.actor.engine.impl_cfg.use_thd=True
    +actor_rollout_ref.actor.engine.impl_cfg.optimizer=${OPTIMIZER}
    +actor_rollout_ref.actor.engine.impl_cfg.recompute=[full]
    +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.free_cache_engine=True
)

TRAINER=(
    critic.enable=False
    trainer.logger=[console]
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.val_before_train=False
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=(
    hydra.searchpath=[pkg://verl_mlite.config]
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${EXTRA[@]}" \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
