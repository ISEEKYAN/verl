#!/usr/bin/env bash
# CW run harness (NOT part of upstream muon PR diff): 0.5B Muon vs AdamW memory diagnosis.
# Dumps torch_memory snapshots + memory_summary at optimizer-step peak (step 3).
# Proven env from job 14252936 (single-GPU Qwen2.5-0.5B SFT).
#
# Usage (on CW login):
#   cp examples/sft/gsm8k/cw_run_muon_memory_profile_05b.sbatch \
#      ${CODE_ROOT}/runtime/cw_run_muon_memory_profile_05b.sbatch
#   sbatch ${CODE_ROOT}/runtime/cw_run_muon_memory_profile_05b.sbatch
#
# Outputs:
#   ${RUN_ROOT}/job-${SLURM_JOB_ID}/mem-profile-{muon,adam}/rank0/step3/torch_memory_*.pickle
#   ${RUN_ROOT}/job-${SLURM_JOB_ID}/mem-profile-{muon,adam}/memory_summary_step3_rank0.txt

#SBATCH --job-name=muon-mem-prof-05b
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/muon-verl-sft-ab/mem-prof-05b-slurm-%j.out

set -euo pipefail

B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
CODE_ROOT=${CODE_ROOT:-${B}/code}
RUNTIME_ROOT=${RUNTIME_ROOT:-${CODE_ROOT}/runtime/muon-verl-sft-ab}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-${CODE_ROOT}/runtime/muon-mbridge-386bf7af6-r2}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/bayan/code/megatron_lite/verl.complete.sqsh}
VERL_ROOT=${VERL_ROOT:-${CODE_ROOT}/verl_muon_sft}
MLITE_ROOT=${MLITE_ROOT:-${SNAPSHOT_ROOT}/mlite}
MEGATRON_ROOT=${MEGATRON_ROOT:-${SNAPSHOT_ROOT}/megatron-d64}
EMERGING_OPT_ROOT=${EMERGING_OPT_ROOT:-${SNAPSHOT_ROOT}/emerging-optimizers}
MBRIDGE_ROOT=${MBRIDGE_ROOT:-${SNAPSHOT_ROOT}/mbridge-f5d6e2e}
NVRX_VENV=${NVRX_VENV:-${CODE_ROOT}/runtime/muon-p0p1-4d2a5b1df-mb-f5d6e2e-v2/nvrx-only-venv}
NVRX_SITE=${NVRX_SITE:-${NVRX_VENV}/lib/python3.12/site-packages}

MODEL_PATH=${MODEL_PATH:-${CODE_ROOT}/models/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-${RUNTIME_ROOT}/gsm8k_sft}
RUN_ROOT=${RUN_ROOT:-${RUNTIME_ROOT}/mem-profile-05b/job-${SLURM_JOB_ID:-local}}

MAX_STEPS=${MAX_STEPS:-5}
PROFILE_STEP=${PROFILE_STEP:-3}
LR=${LR:-1e-4}
MINLR=${MINLR:-1e-5}

for path in \
  "${CONTAINER_IMAGE}" \
  "${VERL_ROOT}" \
  "${MLITE_ROOT}/experimental/lite" \
  "${MEGATRON_ROOT}" \
  "${EMERGING_OPT_ROOT}" \
  "${MBRIDGE_ROOT}" \
  "${NVRX_VENV}/bin/activate" \
  "${MODEL_PATH}/config.json"; do
  [[ -e "${path}" ]] || {
    echo "FATAL: required path missing: ${path}" >&2
    exit 2
  }
done

mkdir -p "${RUN_ROOT}"

if [[ "${IN_MEM_PROF_CONTAINER:-0}" != "1" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || {
    echo "submit with sbatch on CW login node" >&2
    exit 2
  }
  export CODE_ROOT RUNTIME_ROOT SNAPSHOT_ROOT CONTAINER_IMAGE VERL_ROOT
  export MLITE_ROOT MEGATRON_ROOT EMERGING_OPT_ROOT MBRIDGE_ROOT NVRX_VENV NVRX_SITE
  export MODEL_PATH DATA_DIR RUN_ROOT MAX_STEPS PROFILE_STEP LR MINLR
  exec srun --export=ALL \
    --nodes=1 --ntasks=1 --gpus-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${VERL_ROOT}" \
    env IN_MEM_PROF_CONTAINER=1 bash "${CODE_ROOT}/runtime/cw_run_muon_memory_profile_05b.sbatch"
fi

export PATH=/usr/local/bin:/usr/bin:/bin
hash -r
unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV ROCR_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export VERL_SFT_LOGGING_LEVEL=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "${NVRX_VENV}/bin/activate"
export PYTHONPATH="${VERL_ROOT}:${MLITE_ROOT}/experimental/lite:${MEGATRON_ROOT}:${EMERGING_OPT_ROOT}:${MBRIDGE_ROOT}:${NVRX_SITE}:${PYTHONPATH:-}"

PREP_DONE="${DATA_DIR}/.prep_done"
if [[ ! -f "${PREP_DONE}" ]]; then
  python "${VERL_ROOT}/examples/data_preprocess/gsm8k_multiturn_sft.py" --local_dir "${DATA_DIR}"
  touch "${PREP_DONE}"
fi

run_mem_profile_arm() {
  local optimizer="$1"
  local save_path="${RUN_ROOT}/mem-profile-${optimizer}"
  mkdir -p "${save_path}"
  export VERL_FILE_LOGGER_ROOT="${save_path}"

  local -a optim_args=(
    optim=megatron
    "optim.optimizer=${optimizer}"
    "optim.lr=${LR}"
    "optim.min_lr=${MINLR}"
    optim.lr_warmup_steps=2
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
      "optim.use_layer_wise_distributed_optimizer=${MUON_LAYER_WISE:-True}"
    )
  fi

  echo "=== memory profile arm optimizer=${optimizer} save=${save_path} profile_step=${PROFILE_STEP} ==="

  torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    "data.train_files=${DATA_DIR}/train.parquet" \
    "data.val_files=${DATA_DIR}/test.parquet" \
    data.messages_key=messages \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.num_workers=0 \
    model=hf_model \
    "model.path=${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.use_remove_padding=true \
    "${optim_args[@]}" \
    engine=megatron \
    engine.tensor_model_parallel_size=1 \
    engine.pipeline_model_parallel_size=1 \
    engine.context_parallel_size=1 \
    engine.use_mbridge=True \
    "trainer.default_local_dir=${save_path}" \
    "trainer.project_name=mem-profile-05b-${optimizer}" \
    "trainer.experiment_name=mem-profile-05b-${optimizer}" \
    "trainer.logger=[console,file]" \
    trainer.total_epochs=1 \
    "trainer.total_training_steps=${MAX_STEPS}" \
    "trainer.profile_interval=[${PROFILE_STEP},${PROFILE_STEP}]" \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    profiler.enable=True \
    profiler.tool=torch_memory \
    "profiler.save_path=${save_path}/torch_mem_snapshots" \
  || return $?

  # Post-run: dump memory_summary + optimizer state byte estimate from checkpoint-less model reload
  python - <<PY
import json, os, sys, torch
save_path = "${save_path}"
jsonl = None
for root, _, files in os.walk(save_path):
    for f in files:
        if f.endswith('.jsonl'):
            jsonl = os.path.join(root, f)
            break
if jsonl:
    rows = [json.loads(l) for l in open(jsonl)]
    peak = max((r.get('data', {}).get('perf/max_memory_allocated_gb') or 0) for r in rows)
    print(f"arm=${optimizer} peak_alloc_gb={peak:.4f} steps={len(rows)}")
out_summary = os.path.join(save_path, 'memory_summary_step${PROFILE_STEP}_rank0.txt')
if torch.cuda.is_available():
    with open(out_summary, 'w') as f:
        f.write(torch.cuda.memory_summary())
    print(f"wrote {out_summary}")
PY
}

for arm in muon adam; do
  run_mem_profile_arm "${arm}"
  python - <<'PY' || true
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
PY
  sleep 30
done

echo "=== DONE job=${SLURM_JOB_ID} outputs under ${RUN_ROOT} ==="
ls -la "${RUN_ROOT}"/mem-profile-*/torch_mem_snapshots/ 2>/dev/null || true
