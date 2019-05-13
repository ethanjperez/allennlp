"""
dream_large_runner.py

SLURM Helper for quickly running DREAM BERT Large Training jobs
"""
import argparse
import os

PYTHON_PATH = '/private/home/siddk/.conda/envs/allennlp/bin/python3.6'
PROGRAM_PATH = '/private/home/siddk/allennlp/allennlp/run.py'

# DREAM FULL Large Training
full_commands = [
    """train training_config/bert_mc_gpt.large.dream.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=32.lr=5e-6.f -d f -a 32"""
    """train training_config/bert_mc_gpt.large.dream.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=32.lr=1e-5.f -d f -a 32""",
    """train training_config/bert_mc_gpt.large.dream.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=32.lr=2e-5.f -d f -a 32""",
    """train training_config/bert_mc_gpt.large.dream.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=32.lr=3e-5.f -d f -a 32""",
    """train training_config/bert_mc_gpt.large.dream.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=16.lr=5e-6.f -d f -a 16""",
    """train training_config/bert_mc_gpt.large.dream.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=16.lr=1e-5.f -d f -a 16""",
    """train training_config/bert_mc_gpt.large.dream.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=16.lr=2e-5.f -d f -a 16""",
    """train training_config/bert_mc_gpt.large.dream.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=16.lr=3e-5.f -d f -a 16""",
    """train training_config/bert_mc_gpt.large.dream.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=12.lr=5e-6.f -d f -a 12""",
    """train training_config/bert_mc_gpt.large.dream.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=12.lr=1e-5.f -d f -a 12""",
    """train training_config/bert_mc_gpt.large.dream.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=12.lr=2e-5.f -d f -a 12""",
    """train training_config/bert_mc_gpt.large.dream.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=12.lr=3e-5.f -d f -a 12""",
    """train training_config/bert_mc_gpt.large.dream.lr=5e-6.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=8.lr=5e-6.f -d f -a 8""",
    """train training_config/bert_mc_gpt.large.dream.lr=1e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=8.lr=1e-5.f -d f -a 8""",
    """train training_config/bert_mc_gpt.large.dream.lr=2e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=8.lr=2e-5.f -d f -a 8""",
    """train training_config/bert_mc_gpt.large.dream.lr=3e-5.jsonnet -s /checkpoint/siddk/debate/runs/dream/dream.f.bert_mc_gpt.large.bsz=8.lr=3e-5.f -d f -a 8"""
]


def parse_args():
    p = argparse.ArgumentParser("DREAM SLURM Runner")
    p.add_argument("-m", "--mode", default='full', help='Default run mode in < mh (middle-high), pl (passage-len)>')

    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    if args.mode == 'full':
        n_id = int(os.environ['SLURM_ARRAY_TASK_ID']) % 16

        run_command = "%s %s %s" % (PYTHON_PATH, PROGRAM_PATH, full_commands[n_id])
        print('Running %s!' % run_command)
        os.system(run_command)
