# VIP / GRPO Slurm launchers

Slurm ports of the Beaker `glm52_hypers` VIP scripts. These start a Ray cluster
with `srun` (one task per node), then run `open_instruct/grpo_fast.py` on rank 0.

## Layout

| File | Beaker counterpart |
|------|--------------------|
| `qwen3_4b_glm52_hypers.sh` | `scripts/train/vip/math_train_rl/qwen3_4b_glm52_hypers.sh` |
| `dr_tulu_8b_glm52_hypers.sh` | `scripts/train/vip/dr_tulu/dr_tulu_8b_glm52_hypers.sh` |
| `ray_node_setup.sh` | Slurm analogue of `configs/beaker_configs/ray_node_setup.sh` |
| `setup_apptainer_env.sh` | Frozen `uv.lock` environment setup inside Apptainer |

## Quick start

```bash
mkdir -p logs

# Math (1 node × 8 GPUs)
sbatch scripts/slurm/vip/qwen3_4b_glm52_hypers.sh

# Dr. Tulu (2 nodes × 8 GPUs) — export API keys first
export SERPER_API_KEY=... S2_API_KEY=... JINA_API_KEY=... OPENAI_API_KEY=...
sbatch scripts/slurm/vip/dr_tulu_8b_glm52_hypers.sh
```

Override partition / account at submit time:

```bash
sbatch --partition=gpu --account=myacct scripts/slurm/vip/qwen3_4b_glm52_hypers.sh
```

## Notes

- Edit `#SBATCH` directives (partition, account, time, cpus) for your cluster.
- Training hyperparams match the Beaker glm52 scripts (group size 1, 256 prompts,
  length-adaptive GAE, skip-tool-outputs, TIS mask 0.8/3.0, value warmup 100,
  value_num_epochs 2).
- Requires `uv` + project deps installed on the compute nodes (or an env module
  that provides them).

## Hyak H200 example

Klone's host glibc is older than the binary requirement for vLLM 0.19.1. Build
or pull a modern Apptainer image containing Python 3.12 and Git, then prepare the
project environment from `uv.lock` inside that image:

```bash
export APPTAINER_IMAGE=/gscratch/h2lab/$USER/containers/vllm-openai-v0.19.1.sif
sbatch --account=h2lab --partition=gpu-h200 --gpus-per-node=4 \
  --export=ALL,APPTAINER_IMAGE \
  scripts/slurm/vip/setup_apptainer_env.sh
```

For a 4-GPU math run, use two learner actors and two vLLM engines:

```bash
sbatch --account=h2lab --partition=gpu-h200 --gpus-per-node=4 \
  --cpus-per-task=64 \
  --export=ALL,APPTAINER_IMAGE,APPTAINER_LOCAL_VENV=1,NUM_LEARNERS_PER_NODE=2,VLLM_NUM_ENGINES=2 \
  scripts/slurm/vip/qwen3_4b_glm52_hypers.sh
```

`APPTAINER_LOCAL_VENV=1` uses `uv sync --frozen --no-dev` to materialize the
locked environment under the compute node's `/scr` SSD. This avoids slow Python
imports from the many small environment files on Klone's shared filesystem.
