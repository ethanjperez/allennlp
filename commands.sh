#!/usr/bin/env bash

##### AllenNLP Commands

# NB: Span-based debates: Do NOT use span_end_encoder in debater config (only SQUAD judge config)

# TODO: Check 6090-6095. And others right before / that may also have updated

### RACE
# Concat Oracle L
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d L -c concat -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=L.c=concat.dev.txt

# Concat SL Multi-step: -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.all.pkl TODO


# Concat Oracle Multi-step
~0.7250601995184038 (train)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.0.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.0'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.0.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.1.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.1'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.1.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.2'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.3.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.3'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.3.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.4.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.4'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.4.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.5.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.5'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.5.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.6.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.6'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.6.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.7.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.7'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.7.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.8.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.8'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.8.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.train.9.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.9'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.train.9.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.dev.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.dev.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B A B A B A B A -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A.test.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}" 2>&1 | tee tmp/race.best.f/d=B_A_B_A_B_A_B_A.c=concat.test.txt #done(cassio)

~0.8454527581170656 (train)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.0.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.0'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.0.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.1.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.1'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.1.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.2'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.2.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.3.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.3'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.3.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.4.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.4'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.4.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.5.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.5'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.5.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.6.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.6'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.6.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.7.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.7'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.7.txt #done(cassio)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.8.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.8'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.8.txt #done(prince)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.train.9.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.9'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.train.9.txt #done(prince)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.dev.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.dev.txt #done(?)(prince)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A B A B A B A B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.test.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}" 2>&1 | tee tmp/race.best.f/d=A_B_A_B_A_B_A_B.c=concat.test.txt #done(prince)

# RL A Eval with Wrong Answer Conditioning
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.rl.e -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -r -e -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}"
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.rl.e -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -r -e -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train'}"

allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat

# B Pretrained
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.init=a.rl.m=prob.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -r
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.init=a.rl.m=prob.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -r

tb race.b.init=a.rl.m=prob.bsz=32.lr=5e-6.c=concat,race.l.init=a.rl.m=prob.bsz=32.lr=5e-6.c=concat:race.l.init=a.rl.m=prob.bsz=32.lr=5e-6.c=concat --port 6090

# A/B Influence (Vary LR)
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.b.i.bsz=32.lr=2e-6.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.i.bsz=32.lr=1e-5.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.b.i.bsz=32.lr=2e-5.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -i -f

tb race.b.i.bsz=32.lr=2e-6.m=prob.c=concat:race.b.i.bsz=32.lr=2e-6.m=prob.c=concat,race.b.i.bsz=32.lr=1e-5.m=prob.c=concat:race.b.i.bsz=32.lr=1e-5.m=prob.c=concat,race.b.i.bsz=32.lr=2e-5.m=prob.c=concat:race.b.i.bsz=32.lr=2e-5.m=prob.c=concat,race.b.m=prob.bsz=32.lr=5e-6.c=concat.i:race.b.m=prob.bsz=32.lr=5e-6.c=concat.i --port 6091

allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.a.i.bsz=32.lr=2e-6.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.i.bsz=32.lr=1e-5.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.a.i.bsz=32.lr=2e-5.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -i -f

tb race.a.i.bsz=32.lr=2e-6.m=prob.c=concat:race.a.i.bsz=32.lr=2e-6.m=prob.c=concat,race.a.i.bsz=32.lr=1e-5.m=prob.c=concat:race.a.i.bsz=32.lr=1e-5.m=prob.c=concat,race.a.i.bsz=32.lr=2e-5.m=prob.c=concat:race.a.i.bsz=32.lr=2e-5.m=prob.c=concat,race.a.m=prob.bsz=32.lr=5e-6.c=concat.i:race.a.m=prob.bsz=32.lr=5e-6.c=concat.i --port 6092

# A/B+Theory-of-Mind
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=5e-6.c=concat.t -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -t -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=5e-6.c=concat.t -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -c concat -t -f

tb race.b.m=prob.bsz=32.lr=5e-6.c=concat.t:race.b.m=prob.bsz=32.lr=5e-6.c=concat.t,race.b.m=prob.bsz=64.lr=5e-6.c=concat.t:race.b.m=prob.bsz=64.lr=5e-6.c=concat.t --port 6093

allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=5e-6.c=concat.t -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -t -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=5e-6.c=concat.t -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -c concat -t -f

tb race.a.m=prob.bsz=32.lr=5e-6.c=concat.t:race.a.m=prob.bsz=32.lr=5e-6.c=concat.t,race.a.m=prob.bsz=64.lr=5e-6.c=concat.t:race.a.m=prob.bsz=64.lr=5e-6.c=concat.t --port 6094

# L+QA_loss+Theory-of-Mind
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.t.q=2.bsz=32.lr=5e-6.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 2 -t -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.t.q=1.bsz=32.lr=5e-6.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 1 -t -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.l.t.q=2.bsz=32.lr=1e-5.m=prob.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 2 -t -f

tb race.l.t.q=2.bsz=32.lr=5e-6.m=prob.c=concat:race.l.t.q=2.bsz=32.lr=5e-6.m=prob.c=concat,race.l.t.q=1.bsz=32.lr=5e-6.m=prob.c=concat:race.l.t.q=1.bsz=32.lr=5e-6.m=prob.c=concat,race.l.t.q=2.bsz=32.lr=1e-5.m=prob.c=concat:race.l.t.q=2.bsz=32.lr=1e-5.m=prob.c=concat --port 6095

# L+Influence
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=2e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=1e-5.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -i -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -i -f

tb race.l.m=prob.bsz=32.lr=5e-6.c=concat.i:race.l.m=prob.bsz=32.lr=5e-6.c=concat.i,race.l.m=prob.bsz=32.lr=2e-6.c=concat.i:race.l.m=prob.bsz=32.lr=2e-6.c=concat.i,race.l.m=prob.bsz=64.lr=1e-5.c=concat.i:race.l.m=prob.bsz=64.lr=1e-5.c=concat.i,race.l.m=prob.bsz=64.lr=5e-6.c=concat.i:race.l.m=prob.bsz=64.lr=5e-6.c=concat.i --port 6061

# L+Influence+QA Loss
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=2.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 2 -i -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 1 -i -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=5e-1.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q .5 -i -f

allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=2.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -q 2 -i -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -q 1 -i -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=5e-1.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -q .5 -i -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 1 -i -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=5e-1.i -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q .5 -i -f

tb race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=2.i:race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=2.i,race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1.i:race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1.i,race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=5e-1.i:race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=5e-1.i,race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=2.i:race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=2.i,race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1.i:race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1.i,race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=5e-1.i:race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=5e-1.i,race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1.i:race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1.i,race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=5e-1.i:race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=5e-1.i --port 6060

# RL+Influence
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=5e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=2e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=1e-5.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -c concat -i -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=5e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -c concat -i -f

tb race.b.m=prob.bsz=32.lr=5e-6.c=concat.i:race.b.m=prob.bsz=32.lr=5e-6.c=concat.i,race.b.m=prob.bsz=32.lr=2e-6.c=concat.i:race.b.m=prob.bsz=32.lr=2e-6.c=concat.i,race.b.m=prob.bsz=64.lr=1e-5.c=concat.i:race.b.m=prob.bsz=64.lr=1e-5.c=concat.i,race.b.m=prob.bsz=64.lr=5e-6.c=concat.i:race.b.m=prob.bsz=64.lr=5e-6.c=concat.i --port 6070
tb race.b.m=prob.bsz=32.lr=5e-6.c=concat:race.b.m=prob.bsz=32.lr=5e-6.c=concat,race.b.m=prob.bsz=32.lr=2e-6.c=concat:race.b.m=prob.bsz=32.lr=2e-6.c=concat,race.b.m=prob.bsz=64.lr=1e-5.c=concat:race.b.m=prob.bsz=64.lr=1e-5.c=concat,race.b.m=prob.bsz=64.lr=5e-6.c=concat:race.b.m=prob.bsz=64.lr=5e-6.c=concat --port 6080

allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=5e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=2e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -i -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=1e-5.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -c concat -i -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=5e-6.c=concat.i -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -c concat -i -f

tb race.a.m=prob.bsz=32.lr=5e-6.c=concat.i:race.a.m=prob.bsz=32.lr=5e-6.c=concat.i,race.a.m=prob.bsz=32.lr=2e-6.c=concat.i:race.a.m=prob.bsz=32.lr=2e-6.c=concat.i,race.a.m=prob.bsz=64.lr=1e-5.c=concat.i:race.a.m=prob.bsz=64.lr=1e-5.c=concat.i,race.a.m=prob.bsz=64.lr=5e-6.c=concat.i:race.a.m=prob.bsz=64.lr=5e-6.c=concat.i --port 6071
tb race.a.m=prob.bsz=32.lr=5e-6.c=concat:race.a.m=prob.bsz=32.lr=5e-6.c=concat,race.a.m=prob.bsz=32.lr=2e-6.c=concat:race.a.m=prob.bsz=32.lr=2e-6.c=concat,race.a.m=prob.bsz=64.lr=1e-5.c=concat:race.a.m=prob.bsz=64.lr=1e-5.c=concat,race.a.m=prob.bsz=64.lr=5e-6.c=concat:race.a.m=prob.bsz=64.lr=5e-6.c=concat --port 6081

# RL+QA Loss Concat
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1e-2 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q .01 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1e-1 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q .1 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 1 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=2 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 2 -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1e-2 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q .01 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1e-1 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q .1 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -q 1 -f

allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1e-1 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -q .1 -f #p
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -q 1 -f #p
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=2 -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -q 2 -f #p

tb race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1e-2:race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1e-2,race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1e-1:race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1e-1,race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1:race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=1,race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=2:race.l.m=prob.bsz=32.lr=5e-6.c=concat.q=2,race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1e-2:race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1e-2,race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1e-1:race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1e-1,race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1:race.l.m=prob.bsz=32.lr=1e-5.c=concat.q=1,race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1e-1:race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1e-1,race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1:race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=1,race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=2:race.l.m=prob.bsz=64.lr=5e-6.c=concat.q=2 --port 6050

# SL+QA Loss Concat (Broken: A/B given answer as input)
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl.bsz=32.lr=1e-5.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -q 1 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl.bsz=32.lr=1e-5.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -q 1 -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents.bsz=32.lr=1e-5.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -q 1 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents.bsz=32.lr=1e-5.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -q 1 -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -q 1 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat.q=1 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -q 1 -f

tb race.a.m=sl.bsz=32.lr=1e-5.c=concat.q=1:race.a.m=sl.bsz=32.lr=1e-5.c=concat.q=1,race.a.m=sl-sents.bsz=32.lr=1e-5.c=concat.q=1:race.a.m=sl-sents.bsz=32.lr=1e-5.c=concat.q=1,race.a.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat.q=1:race.a.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat.q=1 --port 6042
tb race.b.m=sl.bsz=32.lr=1e-5.c=concat.q=1:race.b.m=sl.bsz=32.lr=1e-5.c=concat.q=1,race.b.m=sl-sents.bsz=32.lr=1e-5.c=concat.q=1:race.b.m=sl-sents.bsz=32.lr=1e-5.c=concat.q=1,race.b.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat.q=1:race.b.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat.q=1 --port 6043

# SL Concat
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl.bsz=32.lr=1e-5.2.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=sl.bsz=32.lr=5e-6.2.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl.bsz=32.lr=1e-5.2.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=sl.bsz=32.lr=5e-6.2.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
# SL-Sents
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents.bsz=32.lr=1e-5.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=sl-sents.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents.bsz=32.lr=1e-5.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=sl-sents.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
# SL-Sents-Delta
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.c=concat.all.pkl -a 32 -c concat -f

tb race.a.m=sl.bsz=32.lr=1e-5.2.c=concat:race.a.m=sl.bsz=32.lr=1e-5.2.c=concat,race.a.m=sl.bsz=32.lr=5e-6.2.c=concat:race.a.m=sl.bsz=32.lr=5e-6.2.c=concat,race.a.m=sl-sents.bsz=32.lr=1e-5.c=concat:race.a.m=sl-sents.bsz=32.lr=1e-5.c=concat,race.a.m=sl-sents.bsz=32.lr=5e-6.c=concat:race.a.m=sl-sents.bsz=32.lr=5e-6.c=concat,race.a.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat:race.a.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat,race.a.m=sl-sents-delta.bsz=32.lr=5e-6.c=concat:race.a.m=sl-sents-delta.bsz=32.lr=5e-6.c=concat --port 7010
tb race.b.m=sl.bsz=32.lr=1e-5.2.c=concat:race.b.m=sl.bsz=32.lr=1e-5.2.c=concat,race.b.m=sl.bsz=32.lr=5e-6.2.c=concat:race.b.m=sl.bsz=32.lr=5e-6.2.c=concat,race.b.m=sl-sents.bsz=32.lr=1e-5.c=concat:race.b.m=sl-sents.bsz=32.lr=1e-5.c=concat,race.b.m=sl-sents.bsz=32.lr=5e-6.c=concat:race.b.m=sl-sents.bsz=32.lr=5e-6.c=concat,race.b.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat:race.b.m=sl-sents-delta.bsz=32.lr=1e-5.c=concat,race.b.m=sl-sents-delta.bsz=32.lr=5e-6.c=concat:race.b.m=sl-sents-delta.bsz=32.lr=5e-6.c=concat --port 7011

# RL Concat
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat # Best A RL
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=2e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=1e-5.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=128.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 128 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=256.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 256 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=512.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 512 -c concat -f

allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=2e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=1e-5.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=128.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 128 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=256.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 256 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=512.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 512 -c concat -f

allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.l.m=prob.bsz=32.lr=2e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 32 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=64.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 64 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=128.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 128 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=256.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 256 -c concat -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.l.m=prob.bsz=512.lr=5e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d l -m prob -a 512 -c concat -f

tb race.l.m=prob.bsz=32.lr=5e-6.c=concat:race.l.m=prob.bsz=32.lr=5e-6.c=concat,race.l.m=prob.bsz=32.lr=2e-6.c=concat:race.l.m=prob.bsz=32.lr=2e-6.c=concat,race.l.m=prob.bsz=64.lr=5e-6.c=concat:race.l.m=prob.bsz=64.lr=5e-6.c=concat,race.l.m=prob.bsz=128.lr=5e-6.c=concat:race.l.m=prob.bsz=128.lr=5e-6.c=concat,race.l.m=prob.bsz=256.lr=5e-6.c=concat:race.l.m=prob.bsz=256.lr=5e-6.c=concat,race.l.m=prob.bsz=512.lr=5e-6.c=concat:race.l.m=prob.bsz=512.lr=5e-6.c=concat --port 7000

## SL
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.a.m=sl.bsz=32.lr=2e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl.bsz=64.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 64 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=sl.bsz=32.lr=5e-6.2 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.b.m=sl.bsz=32.lr=2e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl.bsz=64.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 64 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=sl.bsz=32.lr=5e-6.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
# SL-Sents
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents.bsz=32.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.a.m=sl-sents.bsz=32.lr=2e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents.bsz=64.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 64 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=sl-sents.bsz=32.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents.bsz=32.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.b.m=sl-sents.bsz=32.lr=2e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents.bsz=64.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 64 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=sl-sents.bsz=32.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
# SL-Sents-Delta
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=32.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=32.lr=2e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f #p
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=64.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 64 -f #p
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=32.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f #p

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=32.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=32.lr=2e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f #p
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=64.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 64 -f #p
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=32.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f #p

# SL-Sents(-Delta): Equal loss per-sentence (not sample)
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=sl-sents-delta.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=sl-sents-delta.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m sl-sents-delta -p tmp/race.best.f/oracle_outputs.all.pkl -a 32 -f

# RL
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.a.m=prob.bsz=32.lr=2e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.a.m=prob.bsz=64.lr=2e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 64 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=prob.bsz=128.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 128 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=128.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 128 -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.a.m=prob.bsz=128.lr=2e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 128 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.a.m=prob.bsz=256.lr=1e-5 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 256 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.a.m=prob.bsz=256.lr=5e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 256 -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.a.m=prob.bsz=256.lr=2e-6 -j tmp/race.best.f/model.tar.gz -b 1 -d a -m prob -a 256 -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=5e-6.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.b.m=prob.bsz=32.lr=2e-6.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -f
allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=5e-6.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -f
allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.b.m=prob.bsz=64.lr=2e-6.2 -j tmp/race.best.f/model.tar.gz -b 1 -d b -m prob -a 64 -f

allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.ab.m=prob.bsz=32.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d ab -m prob -a 32 -f
allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.ab.m=prob.bsz=64.lr=1e-5.2 -j tmp/race.best.f/model.tar.gz -b 1 -d ab -m prob -a 64 -f

# Q2A (Cassio)
allennlp train training_config/bert_mc_q2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=32.lr=2e-5.f -d f -a 4 -f #29.0
allennlp train training_config/bert_mc_q2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=32.lr=3e-5.f -d f -a 4 -f #26.8
allennlp train training_config/bert_mc_q2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=32.lr=5e-5.f -d f -a 4 -f #24.8
allennlp train training_config/bert_mc_q2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=16.lr=2e-5.f -d f -a 2 -f #27.3
allennlp train training_config/bert_mc_q2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=16.lr=3e-5.f -d f -a 2 -f #30.4
allennlp train training_config/bert_mc_q2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_q2a.bsz=16.lr=5e-5.f -d f -a 2 -f #29.6

# A-only (Cassio): 44.6
allennlp train training_config/bert_mc_a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_a.bsz=32.lr=2e-5.f -d f -a 4 -f # 44.6
allennlp train training_config/bert_mc_a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_a.bsz=32.lr=3e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_a.bsz=32.lr=5e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_a.bsz=16.lr=2e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_a.bsz=16.lr=3e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_a.bsz=16.lr=5e-5.f -d f -a 2 -f #

# PQ2A: 42.7
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=2e-5.f -d f -a 4 -f #42.7
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=3e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=5e-5.f -d f -a 4 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=2e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=3e-5.f -d f -a 2 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=5e-5.f -d f -a 2 -f #
# PQ2A: Smaller forward pass
allennlp train training_config/bert_mc_pq2a.race.lr=1e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=1e-5.a=8.f -d f -a 8 -f #  Prince
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=2e-5.a=8.f -d f -a 8 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=3e-5.a=8.f -d f -a 8 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=5e-5.a=8.f -d f -a 8 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=2e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=2e-5.a=4.f -d f -a 4 -f #
allennlp train training_config/bert_mc_pq2a.race.lr=3e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=3e-5.a=4.f -d f -a 4 -f #  Prince
allennlp train training_config/bert_mc_pq2a.race.lr=5e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=5e-5.a=4.f -d f -a 4 -f #  Prince

# GPT-style
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=2e-5.f -d f -a 16 -f #
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=3e-5.f -d f -a 16 -f #
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=5e-5.f -d f -a 16 -f #
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=2e-5.f -d f -a 8 -f #
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=3e-5.f -d f -a 8 -f #
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=5e-5.f -d f -a 8 -f #
# GPT-style: Smaller forward pass
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f -d f -a 32 -f #  Prince
allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=5e-6.a=32.f -d f -a 32 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2 -d f -a 32 -f #  Cassio: 66.32 @ Epoch 5, 61.0 @ Epoch 1
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=2e-5.a=32.f -d f -a 32 -f #  Cassio NB: Training stats may be different. Before gradient accumulation change.
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=3e-5.a=32.f -d f -a 32 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=5e-5.a=32.f -d f -a 32 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=1e-5.a=16.f -d f -a 16 -f #  Cassio
allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=2e-5.a=16.f -d f -a 16 -f #  Cassio NB: Training stats may be different. Before gradient accumulation change.
allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=3e-5.a=16.f -d f -a 16 -f #  Prince
allennlp train training_config/bert_mc_gpt.race.lr=5e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=16.lr=5e-5.a=16.f -d f -a 16 -f #  Prince

# With Answer-masking + BertAdam, lr={1e-5, 2e-5, 3e-5}, bsz={32, 64}
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=32.lr=3e-5.f.3 -d f -a 4 -f #50.3
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=32.lr=2e-5.f.3 -d f -a 4 -f #42.1
allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.f.3 -d f -a 4 -f #55.4
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=64.lr=3e-5.f.3 -d f -a 8 -f #39
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=64.lr=2e-5.f.3 -d f -a 8 -f #42
allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=64.lr=1e-5.f.3 -d f -a 8 -f #39

## Oracle eval of best method
# Concat Oracle (Dev set only)
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ABABABAB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.ABABABAB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=ABABABAB.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BABABABA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BABABABA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BABABABA.c=concat.dev.2.txt

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ABABABA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.ABABABA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=ABABABA.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BABABAB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BABABAB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BABABAB.c=concat.dev.2.txt

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ABABAB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.ABABAB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=ABABAB.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BABABA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BABABA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BABABA.c=concat.dev.2.txt

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ABABA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.ABABA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=ABABA.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BABAB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BABAB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BABAB.c=concat.dev.2.txt

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ABAB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.ABAB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=ABAB.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BABA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BABA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BABA.c=concat.dev.2.txt

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d AAA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.AAA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=AAA.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d AAB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.AAB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=AAB.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ABA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.ABA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=ABA.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d ABB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.ABB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=ABB.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BAA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BAA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BAA.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BAB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BAB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BAB.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BBA -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BBA.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BBA.c=concat.dev.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BBB -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.BBB.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=BBB.c=concat.dev.2.txt

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.dev.d=A.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-A-concat.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -p tmp/race.best.f/oracle_outputs.dev.d=B.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-B-concat.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d r -p tmp/race.best.f/oracle_outputs.dev.d=r.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-r-concat.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d rr -p tmp/race.best.f/oracle_outputs.dev.d=rr.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-rr-concat.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d AB -p tmp/race.best.f/oracle_outputs.dev.d=AB.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-AB-concat.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BA -p tmp/race.best.f/oracle_outputs.dev.d=BA.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-BA-concat.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d AA -p tmp/race.best.f/oracle_outputs.dev.d=AA.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-AA-concat.txt #p
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d BB -p tmp/race.best.f/oracle_outputs.dev.d=BB.pkl -c concat -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/eval-BB-concat.txt #p

# Concat Oracle on full dataset
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.0.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.0'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.0.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.1.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.1'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.1.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.2'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.3.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.3'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.3.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.4.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.4'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.4.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.5.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.5'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.5.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.6.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.6'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.6.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.7.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.7'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.7.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.8.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.8'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.8.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.train.9.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.9'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.train.9.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.dev.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.dev.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -c concat -p tmp/race.best.f/oracle_outputs.c=concat.test.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}" 2>&1 | tee tmp/race.best.f/d=B.c=concat.test.txt

allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.f.3 -e -r -m prob -d A > tmp/race.bert.bsz=32.lr=1e-5.f.3/eval-A.txt
allennlp train training_config/bert.race.lr=1e-5.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.f.3 -e -r -m prob -d B > tmp/race.bert.bsz=32.lr=1e-5.f.3/eval-B.txt

allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy -e -r -d A -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy/eval-A-model-epoch-1.txt.2
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy -e -r -d B -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy/eval-B-model-epoch-1.txt.2
allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy -e -r -d r -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f.2.copy/eval-r-model-epoch-1.txt.2

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=A.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d B -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=B.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d r -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/dev'}" 2>&1 | tee tmp/race.best.f/d=r.txt

allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.0.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.0'}" 2>&1 | tee tmp/race.best.f/d=A.train.0.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.1.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.1'}" 2>&1 | tee tmp/race.best.f/d=A.train.1.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.2.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.2'}" 2>&1 | tee tmp/race.best.f/d=A.train.2.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.3.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.3'}" 2>&1 | tee tmp/race.best.f/d=A.train.3.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.4.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.4'}" 2>&1 | tee tmp/race.best.f/d=A.train.4.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.5.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.5'}" 2>&1 | tee tmp/race.best.f/d=A.train.5.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.6.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.6'}" 2>&1 | tee tmp/race.best.f/d=A.train.6.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.7.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.7'}" 2>&1 | tee tmp/race.best.f/d=A.train.7.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.8.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.8'}" 2>&1 | tee tmp/race.best.f/d=A.train.8.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.train.9.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/train.9'}" 2>&1 | tee tmp/race.best.f/d=A.train.9.txt
allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d A -p tmp/race.best.f/oracle_outputs.test.pkl -o "{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw/test'}" 2>&1 | tee tmp/race.best.f/d=A.test.txt

# top_layer_only=false, lr={1e-5, 2e-5, 3e-5}, bsz={32}. 41.6 @ 2 Epochs
allennlp train training_config/bert.race.lr=3e-5.top_layer_only=false.jsonnet -s tmp/race.bert.bsz=32.lr=3e-5.top_layer_only=false.f -d f -a 4 -f #
allennlp train training_config/bert.race.lr=2e-5.top_layer_only=false.jsonnet -s tmp/race.bert.bsz=32.lr=2e-5.top_layer_only=false.f -d f -a 4 -f #
allennlp train training_config/bert.race.lr=1e-5.top_layer_only=false.jsonnet -s tmp/race.bert.bsz=32.lr=1e-5.top_layer_only=false.f -d f -a 4 -f #

# MC: Full passage
allennlp train training_config/bert.race_mc.lr=2e-5.jsonnet -s tmp/race_mc.bert.bsz=32.lr=2e-5.f -d f -a 4 -f # 25
allennlp train training_config/bert.race_mc.lr=3e-5.jsonnet -s tmp/race_mc.bert.bsz=32.lr=3e-5.f -d f -a 4 -f # 45.0
allennlp train training_config/bert.race_mc.lr=5e-5.jsonnet -s tmp/race_mc.bert.bsz=32.lr=5e-5.f -d f -a 4 -f # 25
allennlp train training_config/bert.race_mc.lr=2e-5.jsonnet -s tmp/race_mc.bert.bsz=16.lr=2e-5.f -d f -a 2 -f # 25
allennlp train training_config/bert.race_mc.lr=3e-5.jsonnet -s tmp/race_mc.bert.bsz=16.lr=3e-5.f -d f -a 2 -f # 25
allennlp train training_config/bert.race_mc.lr=5e-5.jsonnet -s tmp/race_mc.bert.bsz=16.lr=5e-5.f -d f -a 2 -f # OOM

# Vanilla
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=32.lr=2e-5.f.2 -d f -a 4 -f # 56.1
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=32.lr=3e-5.f.2 -d f -a 4 -f # 50.7
allennlp train training_config/bert.race.lr=5e-5.jsonnet -s tmp/race.bert.bsz=32.lr=5e-5.f.2 -d f -a 4 -f # 28.1
allennlp train training_config/bert.race.lr=2e-5.jsonnet -s tmp/race.bert.bsz=16.lr=2e-5.f.2 -d f -a 2 -f # 53.6
allennlp train training_config/bert.race.lr=3e-5.jsonnet -s tmp/race.bert.bsz=16.lr=3e-5.f.2 -d f -a 2 -f # 50.0
allennlp train training_config/bert.race.lr=5e-5.jsonnet -s tmp/race.bert.bsz=16.lr=5e-5.f.2 -d f -a 2 -f # 36.6

# Augmented data
allennlp train training_config/bert.race_augmented.lr=2e-5.jsonnet -s tmp/race_augmented.bert.bsz=32.lr=2e-5.f -d f -a 4 -f # 53.4
allennlp train training_config/bert.race_augmented.lr=3e-5.jsonnet -s tmp/race_augmented.bert.bsz=32.lr=3e-5.f -d f -a 4 -f # 32.3
allennlp train training_config/bert.race_augmented.lr=5e-5.jsonnet -s tmp/race_augmented.bert.bsz=32.lr=5e-5.f -d f -a 4 -f # 1.0
allennlp train training_config/bert.race_augmented.lr=2e-5.jsonnet -s tmp/race_augmented.bert.bsz=16.lr=2e-5.f -d f -a 2 -f # 39.8
allennlp train training_config/bert.race_augmented.lr=3e-5.jsonnet -s tmp/race_augmented.bert.bsz=16.lr=3e-5.f -d f -a 2 -f #
allennlp train training_config/bert.race_augmented.lr=5e-5.jsonnet -s tmp/race_augmented.bert.bsz=16.lr=5e-5.f -d f -a 2 -f # 21.1

# BIDAF on top
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

# BiDAF: Print debates
allennlp train training_config/bidaf.race.jsonnet -s tmp/debug -f -j tmp/race.f/model.tar.gz -d B -e -m prob  # eval.1
allennlp train training_config/bidaf.race.jsonnet -s tmp/debug -f -j tmp/race.f/model.tar.gz -d A -e -m prob  # eval.6

# BiDAF: Training J on full passage (normal supervised training)
allennlp train training_config/bidaf.race.size=0.5.jsonnet -s tmp/race.f -d f

# BiDAF: Debate baselines
allennlp train training_config/bidaf.race.size=0.5.jsonnet -s tmp/race.f -e -r -d BAA

# BiDAF: SL baselines
allennlp train training_config/bidaf.race.size=0.5.patience=None.dropout=0.0.jsonnet -s tmp/race.a.m=sl.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d a
allennlp train training_config/bidaf.race.size=0.5.patience=None.dropout=0.0.jsonnet -s tmp/race.b.m=sl.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d b
allennlp train training_config/bidaf.race.patience=None.jsonnet -s tmp/race.a.m=sl.size=full.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d a
allennlp train training_config/bidaf.race.patience=None.jsonnet -s tmp/race.b.m=sl.size=full.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d b
allennlp train training_config/bidaf.race.patience=None.dropout=0.0.jsonnet -s tmp/race.a.m=sl.size=full.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d a
allennlp train training_config/bidaf.race.patience=None.dropout=0.0.jsonnet -s tmp/race.b.m=sl.size=full.dropout=0.0.pt=race.f -j tmp/race.f/model.tar.gz -m sl -d b

# BiDAF: RL: EM Reward (OOM on Titan XP 43% through 1 epoch with br)
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.a.m=em.pt=race.f -j tmp/race.f/model.tar.gz -m em -d a
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.b.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d b
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ab.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d ab
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ar.m=em.pt=race.f -j tmp/race.f/model.tar.gz -m em -d ar
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.br.m=em.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m em -d br

# BiDAF: RL: SSP Reward
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.a.m=prob.pt=race.f -j tmp/race.f/model.tar.gz -m prob -d a
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.b.m=prob.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m prob -d b
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ab.m=prob.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m prob -d ab
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.ar.m=prob.pt=race.f -j tmp/race.f/model.tar.gz -m prob -d ar
allennlp train training_config/bidaf.race.size=0.5.patience=None.jsonnet -s tmp/race.br.m=prob.rb=1-ra.pt=race.f -j tmp/race.f/model.tar.gz -m prob -d br

### SQuAD+BERT
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

### SQuAD+BiDAF
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
allennlp train training_config/bidaf.patience=None.num_epochs=200.jsonnet -s tmp/bg.3.m=prob.rb=1-ra.pt=rr.3 -d bg -j tmp/rr.3/model.tar.gz -m prob

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
allennlp train training_config/bidaf.patience=None.num_epochs=200.dropout=0.0.jsonnet -s tmp/gb.m=sl.dropout=0.0 -j tmp/rr.3/model.tar.gz -d gb -m sl
allennlp train training_config/bidaf.patience=None.num_epochs=200.size=2.dropout=0.0.jsonnet -s tmp/gb.m=slr-prob.size=2.dropout=0.0 -j tmp/rr.3/model.tar.gz -d gb -m sl
allennlp train training_config/bidaf.patience=None.num_epochs=200.size=2.jsonnet -s tmp/gb.m=sl.size=2 -j tmp/rr.3/model.tar.gz -d gb -m sl

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

rsync -rav -e ssh --include '*/' ejp416@access.cims.nyu.edu:~/allennlp/datasets/race_raw .
rsync -rav -e ssh --include '*/' ejp416@access.cims.nyu.edu:~/research/allennlp/tmp/race.best.f/ ~/research/allennlp/tmp/race.best.f

### SLURM
# Live updating dashboard of your jobs:
watch 'squeue -o "%.18i %.40j %.10u %.8T %.10M %.9l %.16b %.6C %.6D %R" -u $USER'

# Cassio GPUs: {titanxp,1080ti,titanx,titanblack,k40,k20,k20x,m2090} {1,.85,.64}
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:titanxp bash
srun --pty --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 bash

# Prince GPUs: {p40,v100,p100,k80}  {.55,1,.65}
srun --pty --mem=20000 -t 6-23:58 --gres=gpu:p40 bash
srun --pty --mem=20000 -t 6-23:58 --gres=gpu:k80 bash

# SBATCH: NB: Cut memory usage based on plots
export COMMAND="allennlp train training_config/race.best.debate.lr=2e-6.jsonnet -s tmp/race.aa.m=prob.bsz=32.lr=2e-6.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d aa -m prob -a 32 -c concat"
export COMMAND_ARRAY=($COMMAND)
export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
