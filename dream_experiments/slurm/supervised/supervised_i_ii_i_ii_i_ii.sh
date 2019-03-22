#!/bin/bash
## SLURM scripts for running Dream Debate Jobs

## Job Metadata
#SBATCH --job-name=Supervised_I_II_I_II_I_II_Debate
#SBATCH --output=/checkpoint/%u/debate_out/Supervised-I-II-I-II-I-II-%j.out
#SBATCH --error=/checkpoint/%u/debate_out/Supervised-I-II-I-II-I-II-%j.err

## Partition
#SBATCH --partition=uninterrupted
#SBATCH --comment="Meeting later today!"

## Number of Nodes (Number of Tasks to run)
#SBATCH --nodes=10

## Tasks per Node
#SBATCH --ntasks-per-node=1

## CPUS/Rollout Workers per Task
#SBATCH --cpus-per-task=8

## Time Limit - 1 Day
#SBATCH --time=1440

## GPUS - One per Task
#SBATCH --gres=gpu:1

# Start clean
module purge

# Load what we need
module load anaconda3

### Section 3:
### Run your job.
srun --label /private/home/siddk/.conda/envs/allennlp/bin/python3.6 /private/home/siddk/allennlp/dream_experiments/slurm/dream_runner.py -m supervised -s 4