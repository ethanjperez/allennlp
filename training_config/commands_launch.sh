#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race.best.f -e -r -d f -p tmp/race.best.f/dream.oracle_outputs.d=f.test.num_sents_gt_26.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/dream/train.json', 'validation_data_path': 'datasets/dream/test.num_sents_gt_26.json', 'dataset_reader': {'type': 'dream-mc'}}\" 2>&1 | tee tmp/race.best.f/dream.d=f.test.num_sents_gt_26.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
