"""
dream_runner.py

SLURM Helper Function for Quickly Running various DREAM BERT MC Sweeps.
"""
import argparse
import os

PYTHON_PATH = '/private/home/siddk/.conda/envs/allennlp/bin/python3.6'

PROGRAM_PATH = '/private/home/siddk/allennlp/allennlp/run.py'

TRAIN_CONFIG = "{'trainer': { 'num_epochs': 20, 'patience': 10, 'validation_metric': '+start_acc', 'cuda_device': 0, " \
               "'learning_rate_scheduler': {'type': 'reduce_on_plateau', 'factor': 0.67, 'mode': 'max', " \
               "'patience': 1}, 'optimizer': {'lr': %.7f, 'type': 'bert_adam'}}}"

LR = [1e-5, 2e-5, 3e-5, 5e-5, 5e-6]

BASE_BSZ = 8

ACCUMULATION_STEPS = [2, 4]

GPT_ACCUMULATION_STEPS = [16, 32]

MODE_CONFIGS = {m: '/private/home/siddk/allennlp/training_config/dream/bert_mc_%s.dream.bsz=8.lr=FILL.jsonnet' % m
                for m in ['pq2a', 'a', 'q2a']}

MODE_CONFIGS['gpt'] = '/private/home/siddk/allennlp/training_config/dream/bert_mc_gpt.dream.bsz=1.lr=FILL.jsonnet'
MODE_CONFIGS['gpt-scratch'] = '/private/home/siddk/allennlp/training_config/dream/bert_mc_gpt_scratch.dream.bsz=1.lr=FILL.jsonnet'

CKPT_PATH = "/checkpoint/siddk/debate/dream/dream.bert_mc_%s.bsz=%d.lr=%.1e.f"

BEST_CONFIG = '/private/home/siddk/allennlp/training_config/dream/dream.best.jsonnet'

BEST_TRAIN_CONFIG = '/private/home/siddk/allennlp/training_config/dream/dream.best.train.jsonnet'

BEST_CKPT_PATH = '/checkpoint/siddk/debate/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f'

# DEBATE_MODES = ["ⅰ", "ⅱ", "ⅰ ⅱ", "ⅰ ⅱ ⅰ ⅱ", "ⅰ ⅱ ⅰ ⅱ ⅰ ⅱ"]

DEBATE_MODES = ["a", "b", "a b", "a b a b", "a b a b a b"]

# DEBATE_MODES = ["A B A B A B", "B A B A B A"]


def parse_args():
    p = argparse.ArgumentParser("Dream SLURM Runner")
    p.add_argument("-m", "--mode", default='pq2a', help='Default BERT Mode - choices <pq2a | a | q2a | gpt | oracle>')
    p.add_argument("-o", "--oracle_mode", default='eval', help='Oracle Mode for dumping - choices <eval | train>')
    p.add_argument("-s", "--supervised", default=0, type=int, help='Debate mode to run search over for super training')

    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Lift SLURM Node ID
    s_id = int(os.environ['SLURM_NODEID'])

    # Set Up Run Command
    if args.mode in ['pq2a', 'a', 'q2a']:
        run_command = '%s %s train %s -s %s -d f -a %d -o "%s"' % (
            PYTHON_PATH,
            PROGRAM_PATH,
            MODE_CONFIGS[args.mode],
            CKPT_PATH % (args.mode, BASE_BSZ * ACCUMULATION_STEPS[s_id % len(ACCUMULATION_STEPS)], LR[s_id % len(LR)]),
            ACCUMULATION_STEPS[s_id % len(ACCUMULATION_STEPS)],
            TRAIN_CONFIG % LR[s_id % len(LR)]
        )

    elif args.mode in ['gpt']:
        run_command = '%s %s train %s -s %s -d f -a 32 -o "%s"' % (
            PYTHON_PATH,
            PROGRAM_PATH,
            MODE_CONFIGS[args.mode],
            CKPT_PATH % (args.mode, 32, LR[s_id % len(LR)]),
            TRAIN_CONFIG % LR[s_id % len(LR)]
        )

    elif args.mode in ['gpt-scratch']:
        run_command = '%s %s train %s -s %s -d f -a 32 -o "%s"' % (
            PYTHON_PATH,
            PROGRAM_PATH,
            MODE_CONFIGS[args.mode],
            CKPT_PATH % (args.mode, 32, LR[s_id % len(LR)]),
            TRAIN_CONFIG % LR[s_id % len(LR)]
        )

    elif args.mode in ['oracle']:
        if args.oracle_mode == 'eval':
            run_command = '%s %s train %s -s %s -e -r -d %s -c concat -p %s' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                BEST_CONFIG,
                BEST_CKPT_PATH,
                DEBATE_MODES[s_id],
                os.path.join(BEST_CKPT_PATH, 'oracle_outputs.d=' + ("".join(DEBATE_MODES[s_id].split()) + '.dev.pkl'))
            )
        elif args.oracle_mode == 'train':
            run_command = '%s %s train %s -s %s -e -r -d %s -c concat -p %s' % (
                PYTHON_PATH,
                PROGRAM_PATH,
                BEST_TRAIN_CONFIG,
                BEST_CKPT_PATH,
                DEBATE_MODES[s_id],
                os.path.join(BEST_CKPT_PATH, 'oracle_outputs.d=' + ("".join(DEBATE_MODES[s_id].split()) + '.train.pkl'))
            )

    elif args.mode in ['supervised']:
        debate_mode = DEBATE_MODES[args.supervised]
        atom_lr = LR[s_id % len(LR)]
        atom_bsz = GPT_ACCUMULATION_STEPS[s_id % len(GPT_ACCUMULATION_STEPS)]

        ckpt_path = '/checkpoint/siddk/debate/dream/dream.%s.m=sl.n=1.x=0.5.lr=%.1e.bsz=%d.c=concat' % \
                    ("".join(debate_mode.split()), atom_lr, atom_bsz)
        judge_path = '/checkpoint/siddk/debate/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f/model.tar.gz'
        oracle_path = '/checkpoint/siddk/debate/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f/oracle_outputs.d=6_AB_turns.all.pkl'

        run_command = '%s %s train %s -s %s -j %s -b 1 -d %s -m sl -p %s -a %d -c concat' % (
            PYTHON_PATH,
            PROGRAM_PATH,
            MODE_CONFIGS['gpt'],
            ckpt_path,
            judge_path,
            debate_mode,
            oracle_path,
            atom_bsz
        )

        if len(debate_mode.split()) > 1:
            run_command += ' -n 1 -x 0.5'

        run_command += ' -o "%s"' % (TRAIN_CONFIG % atom_lr)

    print("Running %s!" % run_command)
    os.system(run_command)
