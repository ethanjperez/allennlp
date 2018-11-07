#!/usr/bin/env bash

### AllenNLP Commands
# Training J only:
allennlp train training_config/bidaf.jsonnet --serialization-dir tmp/j.rounds\=1

# Training A/B with fixed J
allennlp train training_config/bidaf.jsonnet --serialization-dir tmp/ab.rounds\=1.independent.pg -j tmp/j.rounds\=1/model.tar.gz

# Training A/B/J with initialized J (provide model.tar.gz to -j)
allennlp train training_config/bidaf.jsonnet --serialization-dir tmp/ab.rounds\=1.independent.pg.update_judge.3 -j tmp/j.rounds\=1.copy/model.tar.gz -u

# Train A/B/J from scratch (provide .jsonnet file to -j)
allennlp train training_config/bidaf.jsonnet --serialization-dir tmp/ab.rounds\=1.independent.pg.j.dropout=0.5 -j training_config/bidaf.dropout=0.5.jsonnet -u

# Train A/B/J from scratch with F1 reward
allennlp train training_config/bidaf.jsonnet --serialization-dir tmp/ab.rounds\=1.independent.pg.j.dropout=0.5.reward_method=f1 -j training_config/bidaf.dropout=0.5.jsonnet -u -m f1

# Evaluate A/B/J (add -e, no -u)
allennlp train training_config/bidaf.jsonnet --serialization-dir tmp/ab.rounds\=1.independent.pg.j.dropout=0.5 -j training_config/bidaf.dropout=0.5.jsonnet -e

### SLURM
# sbatch job
export JOB_NAME=ab.rounds=1.independent.pg.5
export SAVE_DIR=tmp/$JOB_NAME
if test -e $SAVE_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists. Make sure you wanted to load from an existing checkpoint.\n"; else mkdir -p $SAVE_DIR; fi
sbatch --job-name $JOB_NAME --mem=20000 -t 2-23:58 --gres=gpu:p40 --output $SAVE_DIR/train.log --error $SAVE_DIR/train.stderr.%j --open-mode append --requeue --wrap "\
allennlp train training_config/bidaf.jsonnet --serialization-dir tmp/ab.rounds\=1.independent.pg -j tmp/j.rounds\=1/model.tar.gz --recover"
echo -e "\n${CYAN}${SAVE_DIR}/train.log\n"

# Get a 24GB GPU
srun --pty --mem=20000 -t 2-23:58 --gres=gpu:p40 bash

# Get a dev GPU. Other GPUs: {1080ti,titanxp,titanblack,k40,k20,k20x,m2090}
srun --pty --mem=20000 --gres=gpu:titanxp:1 bash

# Live updating dashboard of your jobs:
watch 'squeue -o "%.18i %.40j %.10u %.8T %.10M %.9l %.16b %.6C %.6D %R" -u $USER'

# Copy Prince tensorboard to local:
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --exclude='*' ejp416@prince.hpc.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp

### Python
# iPDB: To run a list comprehension, use this before
globals().update(locals())
