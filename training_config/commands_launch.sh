#!/usr/bin/env bash

COMMANDS=(
"allennlp train training_config/race.best.debate.lr=5e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=5e-5.bsz=8.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 8 -c concat -f"
"allennlp train training_config/race.best.debate.lr=3e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=3e-5.bsz=8.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 8 -c concat -f"
"allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=2e-5.bsz=8.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 8 -c concat -f"
"allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=1e-5.bsz=8.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 8 -c concat -f"
"allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=5e-6.bsz=8.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 8 -c concat -f"
"allennlp train training_config/race.best.debate.lr=5e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=5e-5.bsz=12.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 12 -c concat -f"
"allennlp train training_config/race.best.debate.lr=3e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=3e-5.bsz=12.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 12 -c concat -f"
"allennlp train training_config/race.best.debate.lr=2e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=2e-5.bsz=12.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 12 -c concat -f"
"allennlp train training_config/race.best.debate.lr=1e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=1e-5.bsz=12.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 12 -c concat -f"
"allennlp train training_config/race.best.debate.lr=5e-6.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=5e-6.bsz=12.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 12 -c concat -f"
"allennlp train training_config/race.best.debate.lr=5e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=5e-5.bsz=16.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 16 -c concat -f"
"allennlp train training_config/race.best.debate.lr=3e-5.jsonnet -s tmp/race.ⅱ.m=sl.n=1.x=0.5.lr=3e-5.bsz=16.c=concat -j tmp/race.best.f/model.tar.gz -b 1 -d ⅱ -m sl -p tmp/race.best.f/oracle_outputs.c=concat.d=1_ⅠⅡ_turns.all.pkl -a 16 -c concat -f"
)

for COMMAND in "${COMMANDS[@]}"; do
    export COMMAND_ARRAY=($COMMAND)
    export SERIALIZATION_DIR="${COMMAND_ARRAY[4]}"
    if test -e $SERIALIZATION_DIR; then echo -e "\n${PURPLE}NOTICE: Directory already exists.\n"; else mkdir -p $SERIALIZATION_DIR; fi
    sbatch --job-name $SERIALIZATION_DIR --mem=40000 -t 1-23:58 --gres=gpu:1080ti:1 --open-mode append --requeue --wrap "$COMMAND"
    echo -e "\n${CYAN}${SERIALIZATION_DIR}/train.log\n"
done
