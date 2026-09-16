# GB200 Nemotron-H alignment reproduction (FSDP2)

This reproduces **Transformers + verl FSDP2**, not Megatron. The experimental
adapter implements EP/CP/PP; no Megatron package or model adaptation is used.

## Verified configuration

| Side | Configuration |
| --- | --- |
| Hardware | Two GB200 nodes, 4 GPUs per node |
| Model | Full 52-layer BF16 Nemotron-3.5-Lightning |
| Inference | TP1 / DP8 / EP8, CUDA Graph enabled |
| Old-logprob scoring | Transformers + FSDP2, TP1 / EP4 / CP2 / PP2 |
| Requests | Four seeded prompts of 2045/2046/2047/2048 tokens; 8192 output tokens |
| Comparisons | Batch1 vs batch4; decode vs full-prefill; trainer vs rollout |

Four unique prompts are rotated across eight ranks (32 rows, not 32 unique
prompts). Trainer cases include single, packed, reversed, repeated batch32 and
dynamic32. This is not a 128-unique-prompt/n8 rollout or full RL benchmark.

## 1. Get the image and exact checkpoint

The artifact is a **Pyxis/Enroot squashfs image**, not a Docker Hub tag. It
contains the native extensions, patched vLLM, a regular installed verl wheel
and both Python check scripts under `/opt/nemotron-alignment`. No source
overlay, editable installation, compilation or Ray cluster is needed.

Image on the experiment cluster:

```text
/mnt/lustre01/users/inf-aoshen/containers/nemotron-alignment-full52-ep4cp2pp2-20260916-arm64.sqsh
SHA256: 3e39add8de1dc171b5c09d535cc5b9993517679ad3b9d7b43a2a65a0b0a6a773
Size: 7,561,379,840 bytes (about 7.04 GiB)
```

Exact checkpoint, separate from the image, about 59 GiB:

```text
/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nemotron/models/Nemotron-52L-aligned-cp2-sgd-step1
```

This derives from `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`,
source revision `434456c9a6753f29d24e23c95d622aaf17111b3b`, **after one test
SGD update**. Downloading the original Hugging Face checkpoint is not the
same weight baseline. Copy the entire checkpoint directory, including all
47 safetensor shards, index, config and tokenizer files.

These paths are **not public download URLs**. On another cluster, obtain
authorized file access or an artifact transfer from the owner. Example,
after creating your destination directories:

```bash
rsync -aP <source-host>:/mnt/lustre01/users/inf-aoshen/containers/nemotron-alignment-full52-ep4cp2pp2-20260916-arm64.sqsh /shared/repro/
rsync -aP <source-host>:/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nemotron/models/Nemotron-52L-aligned-cp2-sgd-step1/ /shared/repro/model/
printf '%s  %s\n' 3e39add8de1dc171b5c09d535cc5b9993517679ad3b9d7b43a2a65a0b0a6a773 /shared/repro/nemotron-alignment-full52-ep4cp2pp2-20260916-arm64.sqsh | sha256sum -c -
```

The checkpoint files must be readable by the recipient account. Provision
at least 80 GiB for artifacts, plus node-local Enroot extraction/JIT space.
Image, model and results paths must be shared between both nodes; results
must be writable. Weights are not embedded in the image or this repository.

Runtime: Torch 2.13.0+cu130, Transformers 5.16.1, Triton 3.7.1, FlashInfer 0.6.18.
vLLM wheel metadata is `0.28.1rc1.dev580+g385dce36b` **plus recorded Python
patches**, not a rebuilt wheel. Tested source snapshots: vLLM `f8e54b654a`,
verl `405dd5cfc3`. Core CUDA/MoE/FA2 native hashes match the original image.
Review diffs: [vLLM #55](https://github.com/aoshen02/vllm/pull/55/files),
[verl + scripts #3](https://github.com/aoshen02/verl/pull/3/files).

## 2. Get the host launcher and allocate two nodes

Get this example from the owner's fork (the clone is **not** mounted):

```bash
git clone --branch codex/nemotron-alignment https://github.com/aoshen02/verl.git verl-nemotron
cd verl-nemotron
```

Requires Slurm with Pyxis/Enroot, compatible NVIDIA drivers, working
NCCL/RDMA and host-managed IMEX channels. A Docker-only host cannot load
`.sqsh` directly. Add your site's normal account/partition flags:

```bash
salloc --nodes=2 --gpus-per-node=4 --time=01:00:00
# Continue inside the allocation shell:
scontrol show hostnames "$SLURM_JOB_NODELIST"
srun -N2 -n2 --ntasks-per-node=1 bash -lc 'hostname; nvidia-smi -L; ip -4 -br addr; ls -l /dev/infiniband /dev/nvidia-caps-imex-channels'
```

Choose the IPv4 of the **first listed node**, on the interface that works
between both nodes. The verified cluster used `enP6p9s0np0`; do not assume
that name on another cluster. IMEX channel numbers can differ by allocation.
The launcher mounts both device directories and checks actual device nodes.
No privileged-container flag is required.

## 3. Generate, then score the same generated tokens

Set paths for your cluster, then run these commands sequentially:

```bash
export NEMOTRON_IMAGE=/shared/repro/nemotron-alignment-full52-ep4cp2pp2-20260916-arm64.sqsh
export NEMOTRON_MODEL_DIR=/shared/repro/model
export NEMOTRON_RESULTS=/shared/repro/results/run-001
export NEMOTRON_MASTER_ADDR=<first-node-ipv4>
export NEMOTRON_SOCKET_IFNAME=<socket-interface>
bash examples/nemotron_alignment/run_nemotron_fsdp2.sh inference &&
bash examples/nemotron_alignment/run_nemotron_fsdp2.sh score
```

The launcher runs the check scripts **inside the image**. It mounts only the
model (read-only), results and RDMA/IMEX devices. It fixes the tested BI,
normalization, seed, prompt/output lengths and topology; it does not force
eager. Inference/scoring ports default to 29961/29963; override
`NEMOTRON_DP_PORT`/`NEMOTRON_TRAIN_PORT` if needed.

Use a fresh result directory for a rerun. Existing logs/JSONs are not
overwritten. Do not run both stages concurrently on the same GPUs.

## 4. Decide whether it passed

Both commands must exit 0. Inference must produce eight `inference-rankN.json`
files and eight `NEMOTRON7_INFERENCE_EXACT` markers. All 32 rows must have
`batch_equal: true` and empty `prefill_mismatches`.

Scoring must print eight `NEMOTRON7_REAL_ENGINE_EXACT` markers, with every
`ENGINE_OLD_LOGPROBS` mismatch count and max difference zero. The historical
`NEMOTRON7_` marker name does **not** mean seven layers were loaded.

Startup, GPU utilization or one rank's marker is not sufficient. After
checking results, exit the allocation shell to release its nodes.

## Optional checks and limits

Set `NEMOTRON_BACKWARD_SMOKE=1` before scoring to add short packed17/19
backward. This does not validate long-sequence backward. Updated-weight
export/reload has been tested through input2K/output2K separately.

The default scoring run initializes no optimizer and performs no weight
updates. Optimizer/RNG resume, arbitrary unequal EP token counts, long
backward and RL convergence are not claimed. EP transport is correctness-first,
not optimized for production throughput.

## Evidence and launcher status

The saved image passed fresh two-node full52 input2K/output8K scoring with
**no source mount**. A full8K inference rerun passed all32rows/262144response
tokens and matched cross-rank/frozen-reference persisted logprob
`float.hex()` values. Image inference packaging smoke separately passed2K/128.
Targeted vLLM tests:23 passed; visible-forward/native-VJP CPU contract:4 passed.

The host launcher consolidates those tested commands. Its shell syntax and
argument checks are verified; it has not had a separate full GPU rerun.
The image's embedded README predates this launcher; this document is the
current reproduction entrypoint. These are fork-review Draft PRs, not
upstream-approved releases; the known verl copyright-header check remains
to be resolved before upstreaming.
