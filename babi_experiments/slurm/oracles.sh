#!/bin/bash
## SLURM scripts for running Allennlp Distributed Training Jobs

## Job Metadata
#SBATCH --job-name=bAbI-Oracles-2G
#SBATCH --output=/checkpoint/%u/allennlp_out/bAbI-Oracles-2G-%j.out
#SBATCH --error=/checkpoint/%u/allennlp_out/bAbI-Oracles-2G-%j.err

## Partition
#SBATCH --partition=uninterrupted

## Number of Nodes (Number of Tasks to run)
#SBATCH --nodes=1

## Tasks per Node
#SBATCH --ntasks-per-node=1

## CPUS/Rollout Workers per Task
#SBATCH --cpus-per-task=16

## Time Limit - 3 Days (1min for test)
#SBATCH --time=200

## GPUS - Two per Task
#SBATCH --gres=gpu:2

# Start clean
module purge

# Load what we need
module load anaconda3

### Section 3:
### Run your job.
srun --label /private/home/siddk/.conda/envs/allennlp/bin/python3.6 /private/home/siddk/allennlp/babi_experiments/slurm/babi_runner.py --debate_mode A AA B BB AB BA