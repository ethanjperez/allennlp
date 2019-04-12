#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=12.f.eval.influence -d r -e -j tmp/race_m.best.bsz=12.f/model.tar.gz -m sl -c concat -i -f -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=12.f/d=r.c=concat.i.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race_m.best.bsz=32.f.eval.influence -d r -e -j tmp/race_m.best.bsz=32.f/model.tar.gz -m sl -c concat -i -f -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race_m.best.bsz=32.f/d=r.c=concat.i.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_16.best.bsz=12.f.eval.influence -d r -e -j tmp/race.num_sents_leq_16.best.bsz=12.f/model.tar.gz -m sl -c concat -i -f -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_16.best.bsz=12.f/d=r.c=concat.i.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_16.best.bsz=32.f.eval.influence -d r -e -j tmp/race.num_sents_leq_16.best.bsz=32.f/model.tar.gz -m sl -c concat -i -f -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_16.best.bsz=32.f/d=r.c=concat.i.race_h.dev.txt"
"allennlp train training_config/race.best.jsonnet -s tmp/race.num_sents_leq_12.best.f.eval.influence -d r -e -j tmp/race.num_sents_leq_12.best.f/model.tar.gz -m sl -c concat -i -f -o \"{'train_data_path': 'allennlp/tests/fixtures/data/race_raw/train', 'validation_data_path': 'datasets/race_raw_high/dev'}\" 2>&1 | tee tmp/race.num_sents_leq_12.best.f/d=r.c=concat.i.race_h.dev.txt"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done

----William Blake",
Spring is coming , spring is coming , All around is _ , Shimmer and quiver on the river , Joy is everywhere .
Spring is coming , spring is coming , Birdies , build your nest ; Weave together straw and feather , Doing each your best .
Spring is coming , spring is coming , Flowers are coming too : Pansies , lilies , daffodillies Now are coming through .
