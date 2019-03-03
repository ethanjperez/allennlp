#!/usr/bin/env bash

##### AllenNLP Commands
### RACE-M
# A-only (baseline)
allennlp train training_config/bert_mc_a.race.lr=5e-6.jsonnet -s tmp/race_m.bert_mc_a.bsz=32.lr=5e-6.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=1e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=32.lr=1e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=2e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=32.lr=2e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=3e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=32.lr=3e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=5e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=32.lr=5e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=5e-6.jsonnet -s tmp/race_m.bert_mc_a.bsz=16.lr=5e-6.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=1e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=16.lr=1e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=2e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=16.lr=2e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=3e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=16.lr=3e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_a.race.lr=5e-5.jsonnet -s tmp/race_m.bert_mc_a.bsz=16.lr=5e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f

# GPT
allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=32.lr=5e-6.a=32.f -d f -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f -d f -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=32.lr=2e-5.a=32.f -d f -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=32.lr=3e-5.a=32.f -d f -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=32.lr=5e-5.a=32.f -d f -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=16.lr=5e-6.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=16.lr=1e-5.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=16.lr=2e-5.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=16.lr=3e-5.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.bsz=1.jsonnet -s tmp/race_m.bert_mc_gpt.bsz=16.lr=5e-5.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f

# DCMN
allennlp train training_config/bert_mc.race.lr=5e-6.jsonnet -s tmp/race.bert_mc.bsz=32.lr=5e-6.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=1e-5.jsonnet -s tmp/race.bert_mc.bsz=32.lr=1e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=2e-5.jsonnet -s tmp/race.bert_mc.bsz=32.lr=2e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=3e-5.jsonnet -s tmp/race.bert_mc.bsz=32.lr=3e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=5e-5.jsonnet -s tmp/race.bert_mc.bsz=32.lr=5e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=5e-6.jsonnet -s tmp/race.bert_mc.bsz=16.lr=5e-6.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=1e-5.jsonnet -s tmp/race.bert_mc.bsz=16.lr=1e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=2e-5.jsonnet -s tmp/race.bert_mc.bsz=16.lr=2e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=3e-5.jsonnet -s tmp/race.bert_mc.bsz=16.lr=3e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc.race.lr=5e-5.jsonnet -s tmp/race.bert_mc.bsz=16.lr=5e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f

# PQ2A
allennlp train training_config/bert_mc_pq2a.race.lr=5e-6.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=32.lr=5e-6.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=1e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=32.lr=1e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=32.lr=2e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=32.lr=3e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=32.lr=5e-5.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=5e-6.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=16.lr=5e-6.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=1e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=16.lr=1e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=16.lr=2e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=16.lr=3e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.bsz=4.jsonnet -s tmp/race_m.bert_mc_pq2a.bsz=16.lr=5e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f

# Q2A (baseline)
allennlp train training_config/bert_mc_q2a.race.lr=5e-6.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=32.lr=5e-6.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=1e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=32.lr=1e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=2e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=32.lr=2e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=3e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=32.lr=3e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=5e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=32.lr=5e-5.f -d f -a 4 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=5e-6.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=16.lr=5e-6.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=1e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=16.lr=1e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=2e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=16.lr=2e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=3e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=16.lr=3e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f
allennlp train training_config/bert_mc_q2a.race.lr=5e-5.jsonnet -s tmp/race_m.bert_mc_q2a.bsz=16.lr=5e-5.f -d f -a 2 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f

# TODO: Add arch=bert_mc_pqa_sa
