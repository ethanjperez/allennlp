#!/usr/bin/env bash

### AllenNLP Commands
# Training J only:
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/rr.2

# Training A/B with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=rr.3 -j tmp/rr.3/model.tar.gz
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=gr.2 -j tmp/gr.2/model.tar.gz

# Training G/B with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode bg --serialization-dir tmp/bg.3.pt=rr.3 -j tmp/rr.3/model.tar.gz

# Training A/B/J with initialized J (provide model.tar.gz to -j and use -u)
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=rr.2.u -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=gr.2.u -j tmp/gr.2/model.tar.gz -u

# Train A/B/J, A/R/J, B/R/J, or GT/B/J from scratch with F1 reward
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ar.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode br --serialization-dir tmp/br.3.j.dropout=0.4 -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode gB --serialization-dir tmp/gB.2 -u

# Evaluate A/B/J (add -e -r, no -u)
allennlp train training_config/bidaf.num_epochs=200.jsonnet --debate_mode rr --serialization-dir tmp/ab.pt\=rr -j tmp/rr.2/model.tar.gz -r -e

# Evaluate A/R/J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ab.3.pt\=rr.2.u -j training_config/bidaf.num_epochs=200.jsonnet -r -e
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ar --serialization-dir tmp/ab.3.pt\=rr.2.u -j tmp/rr.3/model.tar.gz -r -e

# Evaluate J with Ground Truth vs. Oracle B
allennlp train training_config/bidaf.mini.debug.jsonnet --serialization-dir tmp/debug."$(uuid)" -e --debate_mode gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet --serialization-dir tmp/rr.2 -e -r --debate_mode gB

# Debug
allennlp train training_config/bidaf.mini.debug.jsonnet --serialization-dir tmp/debug."$(uuid)" -j training_config/bidaf.mini.debug.jsonnet -u --debate_mode ab

### SLURM
# sbatch job
# NB: Update SERIALIZATION_DIR every run!
export SERIALIZATION_DIR=tmp/ab.3.pt=rr.3
if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists. Make sure you wanted to load from an existing checkpoint.\n"; else mkdir -p $SERIALIZATION_DIR; fi
sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 6-23:58 --gres=gpu:p40 --open-mode append --requeue --wrap "\
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet --debate_mode ab --serialization-dir tmp/ab.3.pt=rr.3 -j tmp/rr.3/model.tar.gz
"
echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"

# Get a 24GB GPU
srun --pty --mem=20000 -t 2-23:58 --gres=gpu:p40 bash

# Get a dev GPU. Other GPUs: {1080ti,titanxp,titanblack,k40,k20,k20x,m2090}
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:titanxp bash
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 bash

# Live updating dashboard of your jobs:
watch 'squeue -o "%.18i %.40j %.10u %.8T %.10M %.9l %.16b %.6C %.6D %R" -u $USER'

# Copy Prince tensorboard to local:
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --exclude='*' ejp416@prince.hpc.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --exclude='*' ejp416@access.cims.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp

### Python
# iPDB: To run a list comprehension, use this before
globals().update(locals())


