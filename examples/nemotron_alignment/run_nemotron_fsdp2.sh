#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

stage=${1:-}
case "$stage" in
  inference|score) ;;
  *) echo "Usage: bash $0 inference|score" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run inside a two-node Slurm allocation}"
: "${SLURM_JOB_NODELIST:?Missing allocated node list}"
: "${NEMOTRON_IMAGE:?Set the shared absolute path to the tested .sqsh}"
: "${NEMOTRON_MODEL_DIR:?Set the shared absolute path to the exact BF16 checkpoint}"
: "${NEMOTRON_RESULTS:?Set a shared absolute output directory}"
: "${NEMOTRON_MASTER_ADDR:?Set node zero IPv4 on the chosen socket interface}"
: "${NEMOTRON_SOCKET_IFNAME:?Set the cross-node socket interface name}"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} == 2 ]] || { echo "Expected two allocated nodes" >&2; exit 2; }
for path in "$NEMOTRON_IMAGE" "$NEMOTRON_MODEL_DIR" "$NEMOTRON_RESULTS"; do
  [[ "$path" == /* && "$path" != *[,:]* ]] || {
    echo "Use absolute mount paths without commas or colons: $path" >&2; exit 2;
  }
done
test -f "$NEMOTRON_IMAGE"
test -f "$NEMOTRON_MODEL_DIR/config.json"
mkdir -p "$NEMOTRON_RESULTS"
test ! -e "$NEMOTRON_RESULTS/$stage.log"
if [[ "$stage" == inference ]]; then
  for rank in {0..7}; do test ! -e "$NEMOTRON_RESULTS/inference-rank$rank.json"; done
else
  for rank in {0..7}; do test -s "$NEMOTRON_RESULTS/inference-rank$rank.json"; done
fi
export NEMOTRON_STAGE="$stage"
export NEMOTRON_DP=8 NEMOTRON_DP_LOCAL=4
export NEMOTRON_DP_MASTER="$NEMOTRON_MASTER_ADDR"
export NEMOTRON_DP_PORT=${NEMOTRON_DP_PORT:-29961}
export NEMOTRON_TRAIN_PORT=${NEMOTRON_TRAIN_PORT:-29963}
export NCCL_SOCKET_IFNAME="$NEMOTRON_SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$NEMOTRON_SOCKET_IFNAME"
export VLLM_BATCH_INVARIANT=1 NEMOTRON_SHARED_NORMS=1 NEMOTRON_EP_SLOT_DIAGNOSTIC=1
export HF_HUB_OFFLINE=1 VERL_USE_EXTERNAL_PLUGINS=none
unset NEMOTRON_EXPORT_UPDATE NEMOTRON_UPDATE_PATH NEMOTRON_ANCHOR
export NEMOTRON_TOKEN_BUDGET=128 NEMOTRON_TEMPERATURE=1
export NEMOTRON_PROMPT_LENGTHS=2045,2046,2047,2048
export NEMOTRON_RESPONSE_LENGTH=8192 NEMOTRON_MAX_MODEL_LEN=12288
export NEMOTRON_MODEL=/models/model NEMOTRON_RESULT=/results/inference.json
export NEMOTRON_REFERENCE=/results/inference-rank0.json
export NEMOTRON_EP=4 NEMOTRON_CP=2 NEMOTRON_PP=2
if [[ "$stage" == score ]]; then
  unset NEMOTRON_SHARED_NORMS NEMOTRON_EP_SLOT_DIAGNOSTIC
fi

srun --jobid="$SLURM_JOB_ID" --export=ALL -N2 -n2 --ntasks-per-node=1 --gpus-per-task=4 \
  --container-image="$NEMOTRON_IMAGE" --container-workdir=/tmp \
  --container-mounts="$NEMOTRON_MODEL_DIR:/models/model:ro,$NEMOTRON_RESULTS:/results,/dev/infiniband:/dev/infiniband,/dev/nvidia-caps-imex-channels:/dev/nvidia-caps-imex-channels" \
  bash -lc '
    set -euo pipefail
    test -n "$(find /dev/infiniband -maxdepth 1 -name "uverbs*" -print -quit)"
    test -n "$(find /dev/nvidia-caps-imex-channels -maxdepth 1 -type c -print -quit)"
    test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 4
    export NEMOTRON_NODE_RANK=$SLURM_PROCID
    if [[ "$NEMOTRON_STAGE" == inference ]]; then
      exec /opt/nemotron-verl-venv/bin/python /opt/nemotron-alignment/check_inference.py
    else
      /opt/nemotron-verl-venv/bin/python -c "import json; from pathlib import Path; rows = [r for p in Path(\"/results\").glob(\"inference-rank[0-7].json\") for r in json.loads(p.read_text())]; assert len(rows) == 32 and all(r[\"batch_equal\"] and not r[\"prefill_mismatches\"] and len(r[\"tokens\"]) == 8192 for r in rows), \"Inference gates did not pass\""
      exec /opt/nemotron-verl-venv/bin/python -m torch.distributed.run \
        --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_PROCID" \
        --master_addr="$NEMOTRON_MASTER_ADDR" --master_port="$NEMOTRON_TRAIN_PORT" \
        /opt/nemotron-alignment/check_training.py
    fi
  ' 2>&1 | tee "$NEMOTRON_RESULTS/$stage.log"
