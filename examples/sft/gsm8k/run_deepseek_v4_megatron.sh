#!/usr/bin/env bash
# GSM8K SFT scale demo | DeepSeek-V4 | Megatron Lite training | GPU
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
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}

LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}

TP=${TP:-1}
PP=${PP:-4}
EP=${EP:-8}
CP=${CP:-4}
ETP=${ETP:-1}
OPTIMIZER=${OPTIMIZER:-fsdp2}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROJECT_NAME=${PROJECT_NAME:-verl-mlite-deepseek_v4-gsm8k-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_v4_gsm8k_sft_${OPTIMIZER}}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
export PYTHONPATH="${MLITE_VERL_ROOT}:${MLITE_LITE_ROOT}:${MLITE_ROOT}:${VERL_ROOT:-}:${MEGATRON_ROOT:-}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES
fi

########################### parameter arrays ###########################
DATA=(
    data.train_files="${TRAIN_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}
    data.use_dynamic_bsz=True
    data.max_token_len_per_gpu=${MAX_LENGTH}
    data.max_length=${MAX_LENGTH}
    data.pad_mode=no_padding
    data.truncation=error
    data.messages_key=messages
)

MODEL=(
    model=hf_model
    model.path="${MODEL_PATH}"
    model.trust_remote_code=True
)

OPTIM=(
    optim=megatron
    optim.lr=${LR}
    optim.min_lr=${MIN_LR}
    optim.weight_decay=${WEIGHT_DECAY}
    optim.clip_grad=${CLIP_GRAD}
    optim.lr_warmup_steps=0
    optim.lr_decay_style=constant
    +optim.override_optimizer_config.offload_fraction=1.0
    +optim.override_optimizer_config.use_precision_aware_optimizer=True
    +optim.override_optimizer_config.decoupled_weight_decay=True
)

ENGINE=(
    hydra.searchpath=[pkg://verl_mlite.config]
    engine=mlite
    engine.tp=${TP}
    engine.pp=${PP}
    engine.vpp=1
    engine.ep=${EP}
    engine.cp=${CP}
    engine.etp=${ETP}
    engine.param_offload=${ALL_OFFLOAD}
    engine.optimizer_offload=${ALL_OFFLOAD}
    engine.grad_offload=${ALL_OFFLOAD}
    engine.attention_backend_override=flash
    engine.impl_cfg.use_thd=True
    +engine.impl_cfg.optimizer=${OPTIMIZER}
    +engine.impl_cfg.recompute=[full]
)

TRAINER=(
    trainer.logger=[console]
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
)

########################### launch ###########################
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NDEVICES_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m verl_mlite.launch verl.trainer.sft_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${OPTIM[@]}" \
    "${ENGINE[@]}" \
    "${TRAINER[@]}" \
    "$@"
