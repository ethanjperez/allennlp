"""
logs2allen.py

Translates Debate Logs to the Allennlp-Expected RACE Format.
"""
import argparse
import json


DUMP_DIR = 'datasets/race_raw/test_%s'


def parse_args():
    p = argparse.ArgumentParser(description='Debate Logs -> Allennlp Translator')
    p.add_argument("-m", "--mode", required=True, help='String ID for Debater Mode (tfidf, fasttext, bert, etc.)')

    p.add_argument("-v", "--val", nargs='+', required=True, help='Path to debate logs for mode agent')
    return p.parse_args()


def translate(d, log):
    with open(log, 'rb') as f:
        data = json.load(f)

    for key in data:
        # Split Key Up
        dtype, lvl, text_id, q_num = key.split('/')
        assert (dtype == 'test')

        import IPython
        IPython.embed()



if __name__ == "__main__":
    # Parse Arguments
    args = parse_args()

    # Create Dump Dir
    dump_dir = DUMP_DIR % args.mode

    # Iterate through Debate Logs and Dump
    for val in args.val:
        translate(dump_dir, val)