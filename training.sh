#!/usr/bin/env bash

### AllenNLP Commands
# Training J only:
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/rr

# Training A/B with fixed J
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.pt=rr -j tmp/rr/model.tar.gz -m f1

# Training A/B/J with initialized J (provide model.tar.gz to -j)
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.pt=rr.2 -j tmp/rr/model.tar.gz -u -m f1

# Train A/B/J from scratch with F1 reward
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab -j training_config/bidaf.num_epochs=200.jsonnet -u -m f1

# Train A/R/J from scratch with F1 reward
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ar -j training_config/bidaf.num_epochs=200.jsonnet -u -m f1

# Train B/R/J from scratch with F1 reward
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode br --serialization-dir tmp/br -j training_config/bidaf.num_epochs=200.jsonnet -u -m f1

# Evaluate A/B/J (add -e -r, no -u)
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/ab.pt\=rr -j tmp/rr/model.tar.gz -m f1 -r -e

# Evaluate J with Ground Truth vs. Oracle B
allennlp train training_config/bidaf.mini.debug.jsonnet --debate_mode gB --serialization-dir tmp/debug."$(uuid)" -e
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode gB --serialization-dir tmp/gr -e -r

### SLURM
# sbatch job
export SAVE_DIR=tmp/ar
if test -e $SAVE_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists. Make sure you wanted to load from an existing checkpoint.\n"; else mkdir -p $SAVE_DIR; fi
sbatch --job-name $SAVE_DIR --mem=20000 -t 2-23:58 --gres=gpu:p40 --open-mode append --requeue --wrap "\
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ar -j training_config/bidaf.num_epochs=200.jsonnet -u -m f1
"
echo -e "\n${CYAN}${SAVE_DIR}/train.log\n"

# Get a 24GB GPU
srun --pty --mem=20000 -t 2-23:58 --gres=gpu:p40 bash

# Get a dev GPU. Other GPUs: {1080ti,titanxp,titanblack,k40,k20,k20x,m2090}
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:titanxp bash
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 bash

# Live updating dashboard of your jobs:
watch 'squeue -o "%.18i %.40j %.10u %.8T %.10M %.9l %.16b %.6C %.6D %R" -u $USER'

# Copy Prince tensorboard to local:
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --exclude='*' ejp416@prince.hpc.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp

### Python
# iPDB: To run a list comprehension, use this before
globals().update(locals())
