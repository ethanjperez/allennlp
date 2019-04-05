#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_16.bert_mc_gpt.bsz=12.lr=1e-5.a=12.f -e -r -d f -c concat -p tmp/race.num_sents_leq_16.bert_mc_gpt.bsz=12.lr=1e-5.a=12.f/oracle_outputs.c=concat.d=f.race_very_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_26/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_16.bert_mc_gpt.bsz=12.lr=1e-5.a=12.f/d=f.c=concat.race_very_large.dev.txt"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_16.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f -e -r -d f -c concat -p tmp/race.num_sents_leq_16.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f/oracle_outputs.c=concat.d=f.race_very_large.dev.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw.num_sents_gt_26/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_16.bert_mc_gpt.bsz=32.lr=1e-5.a=32.f/d=f.c=concat.race_very_large.dev.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:titanblack --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
