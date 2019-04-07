#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d ⅠⅡⅢⅣ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ.race_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_12/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=ⅠⅡⅢⅣ.c=concat.race_large.dev.txt"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d ⅠⅡⅢⅣ ⅠⅡⅢⅣ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.race_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_12/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.c=concat.race_large.dev.txt"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d ⅠⅡⅢⅣ ⅠⅡⅢⅣ ⅠⅡⅢⅣ -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.race_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_12/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.c=concat.race_large.dev.txt"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d rrrr -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=rrrr.race_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_12/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=rrrr.c=concat.race_large.dev.txt"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d rrrrrrrr -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=rrrrrrrr.race_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_12/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=rrrrrrrr.c=concat.race_large.dev.txt"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.best.f -e -r -d rrrrrrrrrrrr -c concat -p tmp/race.num_sents_leq_12.best.f/oracle_outputs.c=concat.d=rrrrrrrrrrrr.race_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_12/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=rrrrrrrrrrrr.c=concat.race_large.dev.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=10000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
