#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f.qtype=A -e -r -d f -c concat -p tmp/race.num_sents_leq_12.best.f.qtype=A/oracle_outputs.c=concat.d=f.qtype=A.num_sents_gt_26.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/A.num_sents_gt_26'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f.qtype=A/d=f.c=concat.qtype=A.num_sents_gt_26.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f.qtype=B -e -r -d f -c concat -p tmp/race.num_sents_leq_12.best.f.qtype=B/oracle_outputs.c=concat.d=f.qtype=B.num_sents_gt_26.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/B.num_sents_gt_26'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f.qtype=B/d=f.c=concat.qtype=B.num_sents_gt_26.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f.qtype=C -e -r -d f -c concat -p tmp/race.num_sents_leq_12.best.f.qtype=C/oracle_outputs.c=concat.d=f.qtype=C.num_sents_gt_26.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/C.num_sents_gt_26'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f.qtype=C/d=f.c=concat.qtype=C.num_sents_gt_26.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f.qtype=D -e -r -d f -c concat -p tmp/race.num_sents_leq_12.best.f.qtype=D/oracle_outputs.c=concat.d=f.qtype=D.num_sents_gt_26.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/D.num_sents_gt_26'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f.qtype=D/d=f.c=concat.qtype=D.num_sents_gt_26.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f.qtype=E -e -r -d f -c concat -p tmp/race.num_sents_leq_12.best.f.qtype=E/oracle_outputs.c=concat.d=f.qtype=E.num_sents_gt_26.pkl -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/E.num_sents_gt_26'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f.qtype=E/d=f.c=concat.qtype=E.num_sents_gt_26.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done

"{'trainer': {'num_epochs': 20, 'validation_metric': '-loss', 'patience': 20, 'cuda_device': 0, 'learning_rate_scheduler': {'type': 'reduce_on_plateau', 'factor': 0.67, 'mode': 'max', 'patience': 1}, 'optimizer': {'lr': 0.000005, 'type': 'bert_adam'}}}"
