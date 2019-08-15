#!/bin/bash
## SLURM scripts for running BERT-Large DREAM Training Jobs

## Job Metadata
#SBATCH --job-name=Dream-Large-Full-Sweep
#SBATCH --output=/checkpoint/%u/debate/logs/Dream-Full-Sweep-%A-%a.out
#SBATCH --error=/checkpoint/%u/debate/logs/Dream-Full-Sweep-%A-%a.err

## Partition
#SBATCH --partition=priority
#SBATCH --comment="Running for EMNLP Abstract Deadline 5/15 and Pre-EMNLP Meeting on 5/13!"

## Number of Nodes (Number of Tasks to run)
#SBATCH --nodes=1
#SBATCH --constraint=volta32gb

## Array Job
#SBATCH --array=1-16

## Tasks per Node
#SBATCH --ntasks-per-node=1

## GPUS - One per Task
#SBATCH --gres=gpu:1

## CPUS/Rollout Workers per Task
#SBATCH --cpus-per-task=8

## Time Limit - 2 Days
#SBATCH --time=2880

# Start clean
module purge

# Load what we need
module load anaconda3

# CD To allennlp directory
cd /private/home/siddk/allennlp

### Section 3:
### Run your job.
srun --label /private/home/siddk/.conda/envs/allennlp/bin/python3.6 /private/home/siddk/allennlp/dream_experiments/slurm/dream_large_runner.py -m full