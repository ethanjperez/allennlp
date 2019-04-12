#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race_m.bsz=16.lr=1e-5.rrrrrrrr -d rrrrrrrr -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race_m.bsz=16.lr=2e-5.rrrrrrrr -d rrrrrrrr -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race_m.bsz=16.lr=3e-5.rrrrrrrr -d rrrrrrrr -a 16 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=5e-6.bsz=1.jsonnet -s tmp/race_m.bsz=32.lr=5e-6.rrrr -d rrrr -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=1e-5.bsz=1.jsonnet -s tmp/race_m.bsz=32.lr=1e-5.rrrr -d rrrr -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=2e-5.bsz=1.jsonnet -s tmp/race_m.bsz=32.lr=2e-5.rrrr -d rrrr -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f"
"allennlp train training_config/bert_mc_gpt.race.lr=3e-5.bsz=1.jsonnet -s tmp/race_m.bsz=32.lr=3e-5.rrrr -d rrrr -a 32 -o \"{'train_data_path': 'datasets/race_raw_middle/train', 'validation_data_path': 'datasets/race_raw_middle/dev'}\" -f"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=20000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done







mv debate_logs.d=AW_AW_AW_AW.json race_h.debate_logs.d=AW_AW_AW_AW.json
mv debate_logs.d=f.json race_h.debate_logs.d=f.json
mv debate_logs.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.json race_h.debate_logs.d=ⅡⅣ_ⅡⅣ_ⅡⅣ_ⅡⅣ.json
mv debate_logs.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.json race_h.debate_logs.d=ⅡⅢ_ⅡⅢ_ⅡⅢ_ⅡⅢ.json
mv debate_logs.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.json race_h.debate_logs.d=ⅢⅣ_ⅢⅣ_ⅢⅣ_ⅢⅣ.json
mv debate_logs.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.json race_h.debate_logs.d=ⅠⅡ_ⅠⅡ_ⅠⅡ_ⅠⅡ.json
mv debate_logs.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.json race_h.debate_logs.d=ⅠⅣ_ⅠⅣ_ⅠⅣ_ⅠⅣ.json
mv debate_logs.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.json race_h.debate_logs.d=ⅠⅢ_ⅠⅢ_ⅠⅢ_ⅠⅢ.json
mv debate_logs.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.json race_h.debate_logs.d=ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ_ⅠⅡⅢⅣ.json
mv debate_logs.d=rrrr_rrrr_rrrr_rrrr.json race_h.debate_logs.d=rrrr_rrrr_rrrr_rrrr.json
