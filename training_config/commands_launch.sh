#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/bert_mc_pq2a.race.lr=5e-6.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=32.lr=5e-6.a=8.f -d f -a 8 -f"
"allennlp train training_config/bert_mc_pq2a.race.lr=5e-6.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=5e-6.a=4.f -d f -a 4 -f"
"allennlp train training_config/bert_mc_pq2a.race.lr=1e-5.bsz=4.jsonnet -s tmp/race.bert_mc_pq2a.bsz=16.lr=1e-5.a=4.f -d f -a 4 -f"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
#    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
