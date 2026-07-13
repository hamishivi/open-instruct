# VIP / GRPO Slurm launchers

Slurm ports of the Beaker `glm52_hypers` VIP scripts. These start a Ray cluster
with `srun` (one task per node), then run `open_instruct/grpo_fast.py` on rank 0.

## Layout

| File | Beaker counterpart |
|------|--------------------|
| `qwen3_4b_glm52_hypers.sh` | `scripts/train/vip/math_train_rl/qwen3_4b_glm52_hypers.sh` |
| `dr_tulu_8b_glm52_hypers.sh` | `scripts/train/vip/dr_tulu/dr_tulu_8b_glm52_hypers.sh` |
| `ray_node_setup.sh` | Slurm analogue of `configs/beaker_configs/ray_node_setup.sh` |

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
