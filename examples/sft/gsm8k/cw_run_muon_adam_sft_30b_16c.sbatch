#!/usr/bin/env bash
# CW run harness (NOT part of upstream muon PR diff): Qwen3-30B-A3B MoE SFT A/B
# Muon vs AdamW on 16×H100 (2 nodes × 8 GPUs), verl Megatron + mbridge + emerging_optimizers.
#
# Proven env from job 14252936 (0.5B smoke) + snapshot muon-mbridge-386bf7af6-r2 (job 13629738).
# PYTHONPATH layout matches run_mbridge_muon_inside_container.sh (shepherd-verified 2026-07-22):
#   verl_muon_sft:mlite/experimental/lite:megatron-d64:emerging-optimizers:mbridge-f5d6e2e:nvrx-only-venv/site-packages
# Shepherd: copy to lustre then sbatch the lustre copy (multi-node srun needs shared path):
#   cp examples/sft/gsm8k/cw_run_muon_adam_sft_30b_16c.sbatch \
#      ${CODE_ROOT}/runtime/cw_run_muon_adam_sft_30b_16c.sbatch
#   ssh cw-dfw-cs-001-login-02 && sbatch ${CODE_ROOT}/runtime/cw_run_muon_adam_sft_30b_16c.sbatch
#
# Outputs (loss + perf/max_memory_* in jsonl):
#   ${RUN_ROOT}/job-${SLURM_JOB_ID}/gsm8k-sft-30b-{muon,adam}/*.jsonl
#
# Parallelism note: 16 GPUs ⇒ TP×PP×EP must equal 16 (DP=1). Default TP=2 PP=1 EP=8
# (same family as examples/gspo_trainer/run_qwen3_30b_a3b_megatron.sh on 8 GPUs uses TP=2 EP=8).

#SBATCH --job-name=muon-sft-30b-ab
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/muon-verl-sft-ab-30b/slurm-%j.out

set -euo pipefail

# Avoid accidental subset runs: env vars may leak from login-shell via --export=ALL.
# MUON_SFT_ARMS leaked as [adam] in job 14272863 — unset all legacy/subset names.
unset ARMS OPTIMIZER MUON_SFT_ARMS SFT_AB_ARMS

########################### paths (proven muon snapshot) ###########################
B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
CODE_ROOT=${CODE_ROOT:-${B}/code}
RUNTIME_ROOT=${RUNTIME_ROOT:-${CODE_ROOT}/runtime/muon-verl-sft-ab-30b}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-${CODE_ROOT}/runtime/muon-mbridge-386bf7af6-r2}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/bayan/code/megatron_lite/verl.complete.sqsh}
VERL_ROOT=${VERL_ROOT:-${CODE_ROOT}/verl_muon_sft}
MLITE_ROOT=${MLITE_ROOT:-${SNAPSHOT_ROOT}/mlite}
MEGATRON_ROOT=${MEGATRON_ROOT:-${SNAPSHOT_ROOT}/megatron-d64}
# Source checkouts (not site-packages wheels) — see job 13629738 run_mbridge_muon_inside_container.sh
EMERGING_OPT_ROOT=${EMERGING_OPT_ROOT:-${SNAPSHOT_ROOT}/emerging-optimizers}
MBRIDGE_ROOT=${MBRIDGE_ROOT:-${SNAPSHOT_ROOT}/mbridge-f5d6e2e}
NVRX_VENV=${NVRX_VENV:-${CODE_ROOT}/runtime/muon-p0p1-4d2a5b1df-mb-f5d6e2e-v2/nvrx-only-venv}
NVRX_SITE=${NVRX_SITE:-${NVRX_VENV}/lib/python3.12/site-packages}

MODEL_PATH=${MODEL_PATH:-${CODE_ROOT}/models/Qwen3-30B-A3B}
DATA_DIR=${DATA_DIR:-${RUNTIME_ROOT}/gsm8k_sft}
RUN_ROOT=${RUN_ROOT:-${RUNTIME_ROOT}/job-${SLURM_JOB_ID:-local}}

########################### training knobs ###########################
MAX_STEPS=${MAX_STEPS:-30}
LR=${LR:-1e-4}
MINLR=${MINLR:-1e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-2048}

# 16 GPUs: TP*PP*EP = 16
TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-8}
CP=${CP:-1}
ETP=${ETP:-1}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-${SLURM_NNODES:-2}}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL_PARALLEL=$((TP * PP * EP))
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
if (( WORLD_SIZE % MODEL_PARALLEL != 0 )); then
  echo "FATAL: world_size=${WORLD_SIZE} not divisible by TP*PP*EP=${MODEL_PARALLEL}" >&2
  exit 2
fi

for path in \
  "${CONTAINER_IMAGE}" \
  "${VERL_ROOT}" \
  "${MLITE_ROOT}/experimental/lite" \
  "${MEGATRON_ROOT}" \
  "${EMERGING_OPT_ROOT}" \
  "${MBRIDGE_ROOT}" \
  "${NVRX_VENV}/bin/activate" \
  "${NVRX_SITE}/nvidia_resiliency_ext-0.6.0.dist-info" \
  "${MODEL_PATH}/config.json"; do
  [[ -e "${path}" ]] || {
    echo "FATAL: required path missing: ${path}" >&2
    exit 2
  }
done

mkdir -p "${RUN_ROOT}" "${DATA_DIR}"

# Multi-node srun must exec a lustre path visible on every node. Under sbatch, BASH_SOURCE
# resolves to the node-local spool copy (only on submit node) — node2 gets exit 127.
SELF_PATH="${SELF_PATH:-${CODE_ROOT}/runtime/cw_run_muon_adam_sft_30b_16c.sbatch}"

if [[ "${IN_MUON_SFT_30B_CONTAINER:-0}" != "1" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "submit with sbatch on CW login node" >&2
    exit 2
  }
  [[ -f "${SELF_PATH}" ]] || {
    echo "FATAL: SELF_PATH missing — copy sbatch to lustre before multi-node run: ${SELF_PATH}" >&2
    exit 2
  }
  # MASTER_ADDR must be resolved on the SLURM host (scontrol is not available inside the container).
  export MASTER_ADDR
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
    export MASTER_ADDR
  fi
  [[ -n "${MASTER_ADDR}" ]] || {
    echo "FATAL: could not resolve MASTER_ADDR from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}" >&2
    exit 2
  }
  export CODE_ROOT RUNTIME_ROOT SNAPSHOT_ROOT CONTAINER_IMAGE VERL_ROOT SELF_PATH
  export MLITE_ROOT MEGATRON_ROOT EMERGING_OPT_ROOT MBRIDGE_ROOT NVRX_VENV NVRX_SITE
  export MODEL_PATH DATA_DIR RUN_ROOT MAX_STEPS LR MINLR
  export TRAIN_BATCH_SIZE MICRO_BATCH_SIZE MAX_TOKEN_LEN
  export TP PP EP CP ETP GPUS_PER_NODE NNODES MASTER_PORT MASTER_ADDR SFT_AB_ARMS_OVERRIDE
  exec srun --export=ALL \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${VERL_ROOT}" \
    env IN_MUON_SFT_30B_CONTAINER=1 bash "${SELF_PATH}"
fi

########################### in-container env ###########################
export PATH=/usr/local/bin:/usr/bin:/bin
hash -r
unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV ROCR_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export HYDRA_FULL_ERROR=1
export VERL_SFT_LOGGING_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-job cache on lustre (avoid /root/.triton stale handles across sequential torchrun arms).
export TRITON_CACHE_DIR="${RUN_ROOT}/.triton_cache"
export XDG_CACHE_HOME="${RUN_ROOT}/.xdg_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}"

# Activate nvrx-only venv (nvidia_resiliency_ext 0.6.0) before PYTHONPATH assembly.
source "${NVRX_VENV}/bin/activate"
export PYTHONPATH="${VERL_ROOT}:${MLITE_ROOT}/experimental/lite:${MEGATRON_ROOT}:${EMERGING_OPT_ROOT}:${MBRIDGE_ROOT}:${NVRX_SITE}:${PYTHONPATH:-}"

# MASTER_ADDR is computed on the SLURM host and passed via --export=ALL; do not call scontrol here.
if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "FATAL: MASTER_ADDR not set — must be resolved before container entry (scontrol unavailable in container)" >&2
  exit 2
fi
export MASTER_PORT NNODES GPUS_PER_NODE
export NODE_RANK="${SLURM_NODEID}"

echo "=== muon-sft-30b-ab node=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} ==="
echo "PYTHONPATH head: ${PYTHONPATH%%:*}"
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

########################### node-0 prep + preflight ###########################
PREP_DONE="${DATA_DIR}/.prep_done"
PREFLIGHT_DONE="${RUN_ROOT}/.preflight_ok"

if [[ "${NODE_RANK}" == "0" ]]; then
  if [[ ! -f "${PREP_DONE}" ]]; then
    echo "=== [node0] gsm8k data prep ==="
    python "${VERL_ROOT}/examples/data_preprocess/gsm8k_multiturn_sft.py" \
      --local_dir "${DATA_DIR}"
    touch "${PREP_DONE}"
  fi

  if [[ ! -f "${PREFLIGHT_DONE}" ]]; then
    echo "=== [node0] preflight: HF config + mbridge qwen3_moe support (fail-loud) ==="
    export MODEL_PATH
    python - <<'PY'
import json
import os
import sys

model_path = os.environ["MODEL_PATH"]
with open(os.path.join(model_path, "config.json")) as f:
    cfg = json.load(f)
model_type = cfg.get("model_type")
print(f"HF model_type={model_type}")
if model_type != "qwen3_moe":
    print(f"FATAL: expected qwen3_moe, got {model_type}", file=sys.stderr)
    sys.exit(3)

from transformers import AutoConfig
hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"AutoConfig class={hf_cfg.__class__.__name__}")

# mbridge: fail loud if qwen3_moe is not registered (proven smoke used qwen3_5_moe only).
try:
    import mbridge
    print(f"mbridge={getattr(mbridge, '__file__', mbridge)}")
except Exception as exc:
    print(f"FATAL: cannot import mbridge: {exc}", file=sys.stderr)
    sys.exit(4)

# Best-effort registry probe; training init is the real gate.
bridge_hint = getattr(mbridge, "SUPPORTED_MODELS", None) or getattr(mbridge, "MODEL_REGISTRY", None)
if bridge_hint is not None:
    keys = list(bridge_hint.keys()) if hasattr(bridge_hint, "keys") else str(bridge_hint)
    print(f"mbridge registry keys sample: {str(keys)[:500]}")
    if isinstance(bridge_hint, dict) and "qwen3_moe" not in bridge_hint and "qwen3_5_moe" in bridge_hint:
        print(
            "WARNING: mbridge registry has qwen3_5_moe but not qwen3_moe — "
            "may fail at model build; continuing to fail-loud at train init.",
            file=sys.stderr,
        )
print("PREFLIGHT_OK")
PY
    touch "${PREFLIGHT_DONE}"
  fi
fi

# Barrier: wait for node0 prep/preflight
for _ in $(seq 1 120); do
  [[ -f "${PREP_DONE}" && -f "${PREFLIGHT_DONE}" ]] && break
  sleep 5
done
[[ -f "${PREP_DONE}" && -f "${PREFLIGHT_DONE}" ]] || {
  echo "FATAL: timed out waiting for node0 prep/preflight" >&2
  exit 5
}

########################### 5-minute GPU diagnostic (all nodes) ###########################
if [[ "${SKIP_GPU_DIAG:-0}" != "1" ]]; then
  echo "=== [node${NODE_RANK}] GPU diagnostic (5-min iron rule) ==="
  nvidia-smi -L || true
  python - <<'PY'
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"gpu{i}: {props.name} total_mem_gb={props.total_memory/1024**3:.1f}")
        x = torch.randn(1024, 1024, device=f"cuda:{i}")
        print(f"gpu{i}: alloc_ok {x.shape}")
else:
    raise SystemExit("FATAL: CUDA not visible inside container")
PY
fi

########################### launch helper ###########################
run_sft_arm() {
  local optimizer="$1"
  local arm_port="$2"
  local exp_name="gsm8k-sft-30b-${optimizer}"
  local save_path="${RUN_ROOT}/${exp_name}"
  mkdir -p "${save_path}"
  export VERL_FILE_LOGGER_ROOT="${save_path}"

  echo "=== [node${NODE_RANK}] SFT arm optimizer=${optimizer} port=${arm_port} save=${save_path} exp=${exp_name} ==="

  local -a optim_args=(
    optim=megatron
    "optim.optimizer=${optimizer}"
    "optim.lr=${LR}"
    "optim.min_lr=${MINLR}"
    optim.lr_warmup_steps=5
    optim.weight_decay=0.1
    "optim.betas=[0.9,0.95]"
    optim.clip_grad=1.0
    optim.lr_warmup_init=0
    optim.lr_decay_style=cosine
  )
  if [[ "${optimizer}" == "muon" ]]; then
    optim_args+=(
      "optim.muon_tp_mode=${MUON_TP_MODE:-blockwise}"
      "optim.muon_scalar_optimizer=${MUON_SCALAR_OPT:-adam}"
      "optim.muon_num_ns_steps=${MUON_NS_STEPS:-5}"
      # Fair A/B vs AdamW: shard Muon optimizer state across DP ranks (Megatron emerging path).
      optim.use_layer_wise_distributed_optimizer=True
    )
  fi

  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${arm_port}" \
    -m verl.trainer.sft_trainer \
    "data.train_files=${DATA_DIR}/train.parquet" \
    "data.val_files=${DATA_DIR}/test.parquet" \
    data.messages_key=messages \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}" \
    data.use_dynamic_bsz=True \
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN}" \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.ignore_input_ids_mismatch=True \
    data.num_workers=0 \
    model=hf_model \
    "model.path=${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.use_remove_padding=true \
    "${optim_args[@]}" \
    engine=megatron \
    "engine.tensor_model_parallel_size=${TP}" \
    "engine.pipeline_model_parallel_size=${PP}" \
    "engine.context_parallel_size=${CP}" \
    "engine.expert_model_parallel_size=${EP}" \
    "engine.expert_tensor_parallel_size=${ETP}" \
    engine.use_mbridge=True \
    "+engine.override_transformer_config.moe_router_dtype=fp32" \
    "+engine.override_transformer_config.moe_permute_fusion=True" \
    "+engine.override_transformer_config.recompute_method=uniform" \
    "+engine.override_transformer_config.recompute_granularity=full" \
    "+engine.override_transformer_config.recompute_num_layers=1" \
    "+engine.override_transformer_config.apply_rope_fusion=True" \
    "trainer.default_local_dir=${save_path}" \
    "trainer.project_name=gsm8k-sft-30b-${optimizer}" \
    "trainer.experiment_name=${exp_name}" \
    "trainer.logger=[console,file]" \
    trainer.total_epochs=1 \
    "trainer.total_training_steps=${MAX_STEPS}" \
    trainer.test_freq=-1 \
    trainer.save_freq=-1
}

########################### A/B arms (both nodes run identical loop; torchrun syncs ranks) ###########################
# Default is always muon+adam. Subset reruns only via explicit sbatch --export=SFT_AB_ARMS_OVERRIDE=muon
# (never read MUON_SFT_ARMS / SFT_AB_ARMS from ambient env — they leaked via --export=ALL in job 14272863).
if [[ -n "${SFT_AB_ARMS_OVERRIDE:-}" ]]; then
  SFT_AB_ARMS="${SFT_AB_ARMS_OVERRIDE}"
else
  SFT_AB_ARMS="muon,adam"
fi
IFS=',' read -r -a ARM_LIST <<< "${SFT_AB_ARMS}"

for arm in "${ARM_LIST[@]}"; do
  case "${arm}" in
    muon | adam) ;;
    *)
      echo "FATAL: invalid arm '${arm}' in SFT_AB_ARMS=${SFT_AB_ARMS} (allowed: muon, adam)" >&2
      exit 6
      ;;
  esac
done

if [[ "${NODE_RANK}" == "0" ]]; then
  echo "=== A/B protocol SFT_AB_ARMS=[${SFT_AB_ARMS}] count=${#ARM_LIST[@]}; extract json['data']['train/loss'] + perf/max_memory_* ==="
fi

wait_all_nodes_arm_barrier() {
  local arm="$1"
  local marker="${RUN_ROOT}/.arm_${arm}_node${NODE_RANK}_done"
  touch "${marker}"
  for _ in $(seq 1 120); do
    local ready=1
    for ((n = 0; n < NNODES; n++)); do
      [[ -f "${RUN_ROOT}/.arm_${arm}_node${n}_done" ]] || ready=0
    done
    (( ready )) && return 0
    sleep 5
  done
  echo "FATAL: timed out waiting for arm=${arm} barrier on all ${NNODES} nodes" >&2
  exit 7
}

for ((i = 0; i < ${#ARM_LIST[@]}; i++)); do
  arm="${ARM_LIST[$i]}"
  arm_port=$((MASTER_PORT + i))
  run_sft_arm "${arm}" "${arm_port}"
  wait_all_nodes_arm_barrier "${arm}"
  if (( i + 1 < ${#ARM_LIST[@]} )); then
    # NCCL teardown + CUDA cache release before the next arm (reduces stale-handle flakes).
    echo "=== [node${NODE_RANK}] inter-arm cleanup before next arm ==="
    sync || true
    sleep 60
    rm -rf "${TRITON_CACHE_DIR}"/* 2>/dev/null || true
    python - <<'PY' || true
import gc
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except Exception:
    pass
gc.collect()
PY
  fi
done

if [[ "${NODE_RANK}" == "0" ]]; then
  echo "=== DONE job=${SLURM_JOB_ID} outputs under ${RUN_ROOT} ==="
  for arm in "${ARM_LIST[@]}"; do
    jsonl="$(find "${RUN_ROOT}/gsm8k-sft-30b-${arm}" -name "gsm8k-sft-30b-${arm}.jsonl" -print -quit 2>/dev/null || true)"
    if [[ -z "${jsonl}" || ! -f "${jsonl}" ]]; then
      echo "FATAL: missing jsonl for arm=${arm} under ${RUN_ROOT}/gsm8k-sft-30b-${arm}" >&2
      exit 8
    fi
    echo "arm=${arm} jsonl=${jsonl}"
    python - <<PY
import json
path = "${jsonl}"
rows = [json.loads(l) for l in open(path)]
losses = [(r.get("step"), r.get("data", {}).get("train/loss")) for r in rows if "train/loss" in r.get("data", {})]
mem = max((r.get("data", {}).get("perf/max_memory_allocated_gb") or 0) for r in rows)
print(f"  steps={len(losses)} first_loss={losses[0][1] if losses else None} last_loss={losses[-1][1] if losses else None} peak_alloc_gb={mem}")
PY
  done
  echo "Extract loss curve example:"
  echo "  python -c \"import json,sys;[print(json.loads(l)['step'], json.loads(l)['data'].get('train/loss'), json.loads(l)['data'].get('perf/max_memory_allocated_gb')) for l in open(sys.argv[1])]\" ${RUN_ROOT}/gsm8k-sft-30b-muon/gsm8k-sft-30b-muon/gsm8k-sft-30b-muon.jsonl"
fi
