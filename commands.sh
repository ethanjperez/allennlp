#!/usr/bin/env bash

### AllenNLP Commands

# NB: Span-based debates: Do NOT use span_end_encoder in debater config (only SQUAD judge config)
# RACE current best model: tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy/model.tar.gz

# BERT RACE Q2A (Cassio)
allennlp train training_config/bert_mc_q2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=32.lr=2e-5.f -d f -a 4 -f #29.0
allennlp train training_config/bert_mc_q2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=32.lr=3e-5.f -d f -a 4 -f #26.8
allennlp train training_config/bert_mc_q2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=32.lr=5e-5.f -d f -a 4 -f #24.8
allennlp train training_config/bert_mc_q2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=16.lr=2e-5.f -d f -a 2 -f #27.3
allennlp train training_config/bert_mc_q2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=16.lr=3e-5.f -d f -a 2 -f #30.4
allennlp train training_config/bert_mc_q2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=16.lr=5e-5.f -d f -a 2 -f #29.6

# BERT RACE A-only (Cassio): 44.6
allennlp train training_config/bert_mc_a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_a.bsz=32.lr=2e-5.f -d f -a 4 -f # 44.6
allennlp train training_config/bert_mc_a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_a.bsz=32.lr=3e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_a.bsz=32.lr=5e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_a.bsz=16.lr=2e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_a.bsz=16.lr=3e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_a.bsz=16.lr=5e-5.f -d f -a 2 -f #

# BERT RACE PQ2A: 42.7
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=2e-5.f -d f -a 4 -f #42.7
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=3e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=5e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=2e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=3e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=5e-5.f -d f -a 2 -f #
# BERT RACE PQ2A: Smaller forward pass
allennlp train training_config/bert_mc_pq2a.race.lr=1e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=1e-5.a=8.f -d f -a 8 -f #  Prince
allennlp train training_config/bert_mc_pq2a.race.lr=1e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=1e-5.a=8.f.2 -d f -a 8 -f #  Cassio
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=2e-5.a=8.f -d f -a 8 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=3e-5.a=8.f -d f -a 8 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=5e-5.a=8.f -d f -a 8 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=2e-5.a=4.f -d f -a 4 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=3e-5.a=4.f -d f -a 4 -f #  Prince
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=5e-5.a=4.f -d f -a 4 -f #  Prince

# BERT RACE GPT-style
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.f.p40 -d f -a 16 -f # Prince: p40
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=2e-5.f -d f -a 16 -f #
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=3e-5.f -d f -a 16 -f #
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=5e-5.f -d f -a 16 -f #
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=2e-5.f -d f -a 8 -f #
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=3e-5.f -d f -a 8 -f #
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=5e-5.f -d f -a 8 -f #
# BERT RACE GPT-style: Smaller forward pass
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f -d f -a 32 -f #  Prince
allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=5e-6.a=32.f -d f -a 32 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2 -d f -a 32 -f #  Cassio: 61.0 @ Epoch 1
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=2e-5.a=32.f -d f -a 32 -f #  Cassio NB: Training stats may be different. Before gradient accumulation change.
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=3e-5.a=32.f -d f -a 32 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=5e-5.a=32.f -d f -a 32 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=1e-5.a=16.f -d f -a 16 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=2e-5.a=16.f -d f -a 16 -f #  Cassio NB: Training stats may be different. Before gradient accumulation change.
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=3e-5.a=16.f -d f -a 16 -f #  Prince
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=5e-5.a=16.f -d f -a 16 -f #  Prince

# BERT RACE with Answer-masking + BertAdam, lr={1e-5, 2e-5, 3e-5}, bsz={32, 64}
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=32.lr=3e-5.f.3 -d f -a 4 -f #50.3
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=32.lr=2e-5.f.3 -d f -a 4 -f #42.1
allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.f.3 -d f -a 4 -f #55.4
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=64.lr=3e-5.f.3 -d f -a 8 -f #39
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=64.lr=2e-5.f.3 -d f -a 8 -f #42
allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=64.lr=1e-5.f.3 -d f -a 8 -f #39
# BERT RACE: Oracle eval of best method
allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.f.3 -e -r -m ssp -d A > tmp/race.bert.bsz=32.lr=1e-5.f.3/eval-A.txt
allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.f.3 -e -r -m ssp -d B > tmp/race.bert.bsz=32.lr=1e-5.f.3/eval-B.txt

train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy -e -r -d B -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}"

# BERT RACE: top_layer_only=false, lr={1e-5, 2e-5, 3e-5}, bsz={32}. 41.6 @ 2 Epochs
allennlp train training_config/bert.race.lr=3e-5.top_layer_only=false.jsonnet -s tmp/race.bert.bsz=32.lr=3e-5.top_layer_only=false.f -d f -a 4 -f #
allennlp train training_config/bert.race.lr=2e-5.top_layer_only=false.jsonnet -s tmp/race.bert.bsz=32.lr=2e-5.top_layer_only=false.f -d f -a 4 -f #
allennlp train training_config/bert.race.lr=1e-5.top_layer_only=false.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.top_layer_only=false.f -d f -a 4 -f #

# RACE MC: Full passage
allennlp train training_config/bert.race_mc.lr=2e-5.jsonnet -s tmp/race_mc.bert.bsz=32.lr=2e-5.f -d f -a 4 -f # 25
allennlp train training_config/bert.race_mc.lr=3e-5.jsonnet -s tmp/race_mc.bert.bsz=32.lr=3e-5.f -d f -a 4 -f # 45.0
allennlp train training_config/bert.race_mc.lr=5e-5.jsonnet -s tmp/race_mc.bert.bsz=32.lr=5e-5.f -d f -a 4 -f # 25
allennlp train training_config/bert.race_mc.lr=2e-5.jsonnet -s tmp/race_mc.bert.bsz=16.lr=2e-5.f -d f -a 2 -f # 25
allennlp train training_config/bert.race_mc.lr=3e-5.jsonnet -s tmp/race_mc.bert.bsz=16.lr=3e-5.f -d f -a 2 -f # 25
allennlp train training_config/bert.race_mc.lr=5e-5.jsonnet -s tmp/race_mc.bert.bsz=16.lr=5e-5.f -d f -a 2 -f # OOM

# BERT RACE
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=32.lr=2e-5.f.2 -d f -a 4 -f # 56.1
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=32.lr=3e-5.f.2 -d f -a 4 -f # 50.7
allennlp train training_config/bert.race.lr=5e-5.jsonnet -s tmp/race.bert.bsz=32.lr=5e-5.f.2 -d f -a 4 -f # 28.1
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=16.lr=2e-5.f.2 -d f -a 2 -f # 53.6
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=16.lr=3e-5.f.2 -d f -a 2 -f # 50.0
allennlp train training_config/bert.race.lr=5e-5.jsonnet -s tmp/race.bert.bsz=16.lr=5e-5.f.2 -d f -a 2 -f # 36.6

# BERT RACE Augmented
allennlp train training_config/bert.race_augmented.lr=2e-5.jsonnet -s tmp/race_augmented.bert.bsz=32.lr=2e-5.f -d f -a 4 -f # 53.4
allennlp train training_config/bert.race_augmented.lr=3e-5.jsonnet -s tmp/race_augmented.bert.bsz=32.lr=3e-5.f -d f -a 4 -f # 32.3
allennlp train training_config/bert.race_augmented.lr=5e-5.jsonnet -s tmp/race_augmented.bert.bsz=32.lr=5e-5.f -d f -a 4 -f # 1.0
allennlp train training_config/bert.race_augmented.lr=2e-5.jsonnet -s tmp/race_augmented.bert.bsz=16.lr=2e-5.f -d f -a 2 -f # 39.8
allennlp train training_config/bert.race_augmented.lr=3e-5.jsonnet -s tmp/race_augmented.bert.bsz=16.lr=3e-5.f -d f -a 2 -f #
allennlp train training_config/bert.race_augmented.lr=5e-5.jsonnet -s tmp/race_augmented.bert.bsz=16.lr=5e-5.f -d f -a 2 -f # 21.1

# BERT RACE (Bidaf on top)
allennlp train training_config/bidaf_bert.race.lr=3e-5.jsonnet -s tmp/race.bert.f -d f  # xl: ~46.% TODO: Rename dir after
allennlp train training_config/bidaf_bert.race.jsonnet -s tmp/race.bert.a=2.f -d f -a 2  # EM=~3%
allennlp train training_config/bidaf_bert.race.jsonnet -s tmp/race.bert.a=4.f -d f -a 4  # EM=~3%
allennlp train training_config/bidaf_bert.race.lr=3e-5.jsonnet -s tmp/race.bert.a=4.lr=3e-5.f -d f -a 4  # race.a  53.4
allennlp train training_config/bidaf_bert.race.lr=3e-5.jsonnet -s tmp/race.bert.a=2.lr=3e-5.f -d f -a 2  # tmp.2  28.0
allennlp train training_config/bidaf_bert.race.lr=2e-5.jsonnet -s tmp/race.bert.a=4.lr=2e-5.f -d f -a 4  # race.b  54.6
allennlp train training_config/bidaf_bert.race.lr=2e-5.jsonnet -s tmp/race.bert.a=2.lr=2e-5.f -d f -a 2  # bg  54.6

# Older runs with RACE logits bug
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=32.lr=3e-5.f -d f -a 4 -f # 1: 34.6% train
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=16.lr=3e-5.f -d f -a 2 -f # 1: 31.7% train
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=32.lr=2e-5.f -d f -a 4 -f # 1: 38.5/48.3 train/val. 2: 53.5/53.3

# ***BERT SQUAD***
allennlp train training_config/bert.lr=5e-5.jsonnet -s tmp/bert.bsz=16.lr=5e-5.f -d f -a 2 -f
allennlp train training_config/bert.lr=5e-5.jsonnet -s tmp/bert.bsz=32.lr=5e-5.f -d f -a 4 -f
allennlp train training_config/bert.lr=3e-5.jsonnet -s tmp/bert.bsz=16.lr=3e-5.f -d f -a 2 -f
allennlp train training_config/bert.lr=3e-5.jsonnet -s tmp/bert.bsz=32.lr=3e-5.f -d f -a 4 -f
allennlp train training_config/bert.lr=2e-5.jsonnet -s tmp/bert.bsz=16.lr=2e-5.f -d f -a 2 -f
allennlp train training_config/bert.lr=2e-5.jsonnet -s tmp/bert.bsz=32.lr=2e-5.f -d f -a 4 -f

# SQuAD+BERT: Training J on full passage (normal supervised training)
allennlp train training_config/bidaf_bert.batch_size=8.lr=3e-5.jsonnet -s tmp/bert.f -d f  # 85.5 F1 TODO: Rename dir after
allennlp train training_config/bidaf_bert.batch_size=8.jsonnet -s tmp/bert.a=2.f -d f -a 2  # 84.9 F1
allennlp train training_config/bidaf_bert.batch_size=8.jsonnet -s tmp/bert.a=4.f -d f -a 4  # 86.1 F1  # REDO: bg

# SQUAD XL Memory sweep
allennlp train training_config/bidaf_bert.squad_xl.max_instances=None.jsonnet -s tmp/squad_xl.a=4.mi=None.f -d f -a 4  # eval.4  Loss ~1K Train F1 10.7
allennlp train training_config/bidaf_bert.squad_xl.max_instances=675.jsonnet -s tmp/squad_xl.a=4.mi=675 -d f.f -a 4  # eval.5  Loss ~100K Train F1 ~0
allennlp train training_config/bidaf_bert.squad_xl.max_instances=1250.jsonnet -s tmp/squad_xl.a=4.mi=1250.f -d f -a 4  # eval.2  Loss ~100K Train F1 9.7



# RACE: Print debates
allennlp train training_config/bidaf.race.jsonnet -s tmp/debug -f -j tmp/race.f/model.tar.gz -d B -e -m ssp  # eval.1
allennlp train training_config/bidaf.race.jsonnet -s tmp/debug -f -j tmp/race.f/model.tar.gz -d A -e -m ssp  # eval.6

# RACE: Training J on full passage (normal supervised training)
allennlp train training_config/bidaf.race.size=0.5.jsonnet -s tmp/race.f -d f

# RACE: Debate baselines
allennlp train training_config/bidaf.race.size=0.5.jsonnet -s tmp/race.f -e -r -d BAA

# RACE: SL baselines
allennlp train training_config/bidaf.race.size=0.5.patience=None.dropout=0.0.jsonnet -s tmp/race.a.m=sl-ssp.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d a
allennlp train training_config/bidaf.race.size=0.5.patience=None.dropout=0.0.jsonnet -s tmp/race.b.m=sl-ssp.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d b
allennlp train training_config/bidaf.race.patience=None.jsonnet -s tmp/race.a.m=sl-ssp.size=full.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d a
allennlp train training_config/bidaf.race.patience=None.jsonnet -s tmp/race.b.m=sl-ssp.size=full.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d b
allennlp train training_config/bidaf.race.patience=None.dropout=0.0.jsonnet -s tmp/race.a.m=sl-ssp.size=full.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d a
allennlp train training_config/bidaf.race.patience=None.dropout=0.0.jsonnet -s tmp/race.b.m=sl-ssp.size=full.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl-ssp -d b

# RACE: RL: EM Reward (OOM on Titan XP 43% through 1 epoch with br)
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.a.m=em.pt=race.f -j tmp/race.f/model.tar.gz -m em -d a
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.b.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d b
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ab.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d ab
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ar.m=em.pt=race.f -j tmp/race.f/model.tar.gz -m em -d ar
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.br.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d br

# RACE: RL: SSP Reward
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.a.m=ssp.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d a
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.b.m=ssp.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d b
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ab.m=ssp.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d ab
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ar.m=ssp.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d ar
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.br.m=ssp.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m ssp -d br

# Training J only:
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/rr.2 -d rr
allennlp train training_config/bidaf.squad_xl.num_epochs=200.batch_size=20.jsonnet -s tmp/squad_xl.f -d f
allennlp train training_config/bidaf.squad_xl.num_epochs=200.size=0.5.jsonnet -s tmp/squad_xl.size=0.5.f -d f
allennlp train training_config/bidaf.squad_xl.num_epochs=200.size=0.25.jsonnet -s tmp/squad_xl.size=0.25.f -d f
allennlp train training_config/bidaf.squad_xl.num_epochs=200.jsonnet -s tmp/squad_xl.rr -d rr

# Training ab with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ab.3.pt=rr.3 -d ab -j tmp/rr.3/model.tar.gz
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ab.3.pt=gr.2 -d ab -j tmp/gr.2/model.tar.gz

# Training bg with fixed J
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/bg.3.rb=1-ra.pt=rr.3 -d bg -j tmp/rr.3/model.tar.gz
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/bg.3.m=ssp.rb=1-ra.pt=rr.3 -d bg -j tmp/rr.3/model.tar.gz -m ssp

# Training abj with initialized J (provide model.tar.gz to -j and use -u)
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ab.3.pt=rr.2.u -d ab -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ab.3.pt=gr.2.u -d ab -j tmp/gr.2/model.tar.gz -u

# Train abj\arj\brj from scratch with F1 reward
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ab.3.j.dropout=0.4 -d ab -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ar.3.j.dropout=0.4 -d ar -j training_config/bidaf.dropout=0.4.jsonnet -u
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/br.3.j.dropout=0.4 -d br -j training_config/bidaf.dropout=0.4.jsonnet -u

# Train j on gB
allennlp train training_config/bidaf.jsonnet -d gB -s tmp/gB.2 -u
allennlp train training_config/bidaf.dropout=0.4.jsonnet -d gB -s tmp/gB.j.dropout=0.4.u -u
# Train j on gB with pre-trained initialization (and maybe frozen layers)
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/gB.j.pt=rr.2.u -d gB -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cw.jsonnet -s tmp/gB.j.no_grad=cw.u -d gB -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwp.jsonnet -s tmp/gB.j.no_grad=cwp.u -d gB -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwpa.jsonnet -s tmp/gB.j.no_grad=cwpa.u -d gB -j tmp/rr.2/model.tar.gz -u
allennlp train training_config/bidaf.num_epochs=200.j.no_grad=cwpam.jsonnet -s tmp/gB.j.no_grad=cwpam.u -d gB -j tmp/rr.2/model.tar.gz -u

# Train b on gb with SL on oracle
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet -s tmp/debug -j training_config/bidaf.cpu.mini.debug.jsonnet -u -d gb -m sl
allennlp train training_config/bidaf.patience=None.num_epochs=200.dropout=0.0.jsonnet -s tmp/gb.m=sl-ssp.dropout=0.0 -j tmp/rr.3/model.tar.gz -d gb -m sl-ssp
allennlp train training_config/bidaf.patience=None.num_epochs=200.size=2.dropout=0.0.jsonnet -s tmp/gb.m=slr-ssp.size=2.dropout=0.0 -j tmp/rr.3/model.tar.gz -d gb -m sl-ssp
allennlp train training_config/bidaf.patience=None.num_epochs=200.size=2.jsonnet -s tmp/gb.m=sl-ssp.size=2 -j tmp/rr.3/model.tar.gz -d gb -m sl-ssp

# Evaluate abj (add -e -r no -u)
allennlp train training_config/bidaf.num_epochs=200.jsonnet -d rr -s tmp/ab.pt\=rr -j tmp/rr.2/model.tar.gz -r -e

# Evaluate arj
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ab.3.pt\=rr.2.u -d ar -j training_config/bidaf.num_epochs=200.jsonnet -r -e
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/ab.3.pt\=rr.2.u -d ar -j tmp/rr.3/model.tar.gz -r -e

# Evaluate j with gB
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet -s tmp/debug -e -d gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/rr.2 -e -r -d gB
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/rr.2 -e -r -d gB -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-one-sent.json'}"
allennlp train training_config/bidaf.num_epochs=200.jsonnet -s tmp/rr.2 -e -r -d gB -o "{'test_data_path': 'datasets/squad/squad-adversarial-add-sent.json'}"

# Evaluate j with gb (SL-trained)
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/gb.m=sl.dropout=0.2.backup -j tmp/rr.3/model.tar.gz -d gb -m sl -e -r

# Debug
allennlp train training_config/bidaf.cpu.mini.debug.jsonnet -s tmp/debug -j training_config/bidaf.cpu.mini.debug.jsonnet -u -d ab

# To manually serialize a model via Python from root dir
from allennlp.models.archival import archive_model
serialization_dir = 'tmp/...'
archive_model(serialization_dir, 'best.th', {})

# Copy Prince tensorboard to local:
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --include '*.json' --exclude='*' ejp416@prince.hpc.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp
rsync -rav -e ssh --include '*/' --include 'events.out.tfevents.*' --include '*.json' --exclude='*' ejp416@access.cims.nyu.edu:~/research/allennlp/tmp/ ~/research/allennlp/tmp

### SLURM
# Live updating dashboard of your jobs:
watch 'squeue -o "%.18i %.40j %.10u %.8T %.10M %.9l %.16b %.6C %.6D %R" -u $USER'

# Cassio GPUs: {1080ti,titanxp,titanblack,k40,k20,k20x,m2090}
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:titanxp bash
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 bash

# Prince GPUS: {p40,v100,p100,k80}
srun --pty --mem=20000 -t 6-23:58 --gres=gpu:p40 bash
srun --pty --mem=20000 -t 6-23:58 --gres=gpu:k80 bash

# SBATCH
export COMMAND="allennlp train training_config/bert_mc_gpt.race.lr=1e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.f.p40 -d f -a 16 -f # Prince: p40"
export COMMAND_ARRAY=($COMMAND)
export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:p40 --open-mode append --requeue --wrap "$COMMAND"
echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
