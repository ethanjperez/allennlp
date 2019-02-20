"""
babi_runner.py

Runs experiments for all three bAbI Tasks in parallel. Assumes each node has 2 GPUs.
"""
import argparse
import os

PYTHON_PATH = '/private/home/siddk/.conda/envs/allennlp/bin/python3.6'

PROGRAM_PATH = '/private/home/siddk/allennlp/allennlp/run.py'

TASKS = ['single', 'double', 'triple']

CONFIGS = ['/private/home/siddk/allennlp/babi_experiments/config/single_babi.jsonnet',
           '/private/home/siddk/allennlp/babi_experiments/config/double_babi.jsonnet',
           '/private/home/siddk/allennlp/babi_experiments/config/triple_babi.jsonnet']

CKPT_ROOT = '/checkpoint/siddk/debate/%s/babi-%s.%s'
JUDGE_ROOT = '/checkpoint/siddk/debate/%s/babi-%s.full/model.tar.gz'


def parse_args():
    p = argparse.ArgumentParser(description="bAbI Debate Experiment Runner")
    p.add_argument("-d", "--debate_mode", nargs='+', help='Debate mode to run - default: (f)ull')

    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Lift SLURM NODEID
    s_id = int(os.environ['SLURM_NODEID']) + 2

    # Launch Jobs
    for mode in args.debate_mode:
        if mode == 'f':      # Full Debate Training
            run_command = "%s %s train %s -s %s -d f -g" % (
                PYTHON_PATH,
                PROGRAM_PATH,
                CONFIGS[s_id],
                CKPT_ROOT % (TASKS[s_id], TASKS[s_id], 'full')
            )

            print("Running following command: %s" % run_command)
            os.system(run_command)

        elif mode.isupper():  # Oracle Debate Training
            run_command = "%s %s train %s -s %s -j %s -e -m f1 -d %s -g" % (
                PYTHON_PATH,
                PROGRAM_PATH,
                CONFIGS[s_id],
                CKPT_ROOT % (TASKS[s_id], TASKS[s_id], mode),
                JUDGE_ROOT % (TASKS[s_id], TASKS[s_id]),
                mode
            )

            print("Running following command: %s" % run_command)
            os.system(run_command)
