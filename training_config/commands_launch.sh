#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=32.lr=2e-5.a=32.f -d f -a 32 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=32.lr=3e-5.a=32.f -d f -a 32 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=16.lr=5e-6.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=16.lr=1e-5.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=16.lr=2e-5.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=16.lr=3e-5.a=16.f -d f -a 16 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=12.lr=5e-6.a=12.f -d f -a 12 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=12.lr=1e-5.a=12.f -d f -a 12 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=12.lr=2e-5.a=12.f -d f -a 12 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=12.lr=3e-5.a=12.f -d f -a 12 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=8.lr=5e-6.a=8.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=8.lr=1e-5.a=8.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=8.lr=2e-5.a=8.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race.num_sents_leq_12.bert_mc_gpt.bsz=8.lr=3e-5.a=8.f -d f -a 8 -o \"{'train_data_path': 'datasets/race_raw.num_sents_leq_12/train', 'validation_data_path': 'datasets/race_raw.num_sents_leq_12/dev'}\" -f"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=30000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
