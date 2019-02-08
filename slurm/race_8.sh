#!/bin/bash
## SLURM scripts for running Allennlp Distributed Training Jobs

## Job Metadata
#SBATCH --job-name=RACE_Judge_8G
#SBATCH --output=/checkpoint/%u/allennlp_out/RACE-Judge-8-%j.out
#SBATCH --error=/checkpoint/%u/allennlp_out/RACE-Judge-8-%j.out

## Partition
#SBATCH --partition=uninterrupted

## Number of Nodes (Number of Tasks to run)
#SBATCH --nodes=1

## Tasks per Node
#SBATCH --ntasks-per-node=1

## CPUS/Rollout Workers per Task
#SBATCH --cpus-per-task=32

## Time Limit - 3 Days (1min for test)
#SBATCH --time=2160

## GPUS - Eight per Task
#SBATCH --gres=gpu:8

# Start clean
module purge

# Load what we need
module load anaconda3

### Section 3:
### Run your job.
srun /private/home/siddk/.conda/envs/allennlp/bin/python3.6 /private/home/siddk/allennlp/allennlp/run.py train /private/home/siddk/allennlp/training_config/bidaf.race.size=0.5.gpu=8.jsonnet -s /checkpoint/siddk/race_8G.f -d f