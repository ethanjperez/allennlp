"""
race_results.py

Fetch race results
"""
import argparse
import json
import os


ROOT_DIR = "/checkpoint/siddk/debate/runs/race"

BSZ = [8, 12, 16, 32]
LR = ["1e-5", "2e-5", "3e-5", "5e-6"]


def parse_args():
    p = argparse.ArgumentParser("Race Result Fetcher")
    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Iterate through Full Sweep and Collect Results
    mode_dict = {}
    for b in BSZ:
        for l in LR:
            # Open Metrics
            ckpt = os.path.join(ROOT_DIR, 'race_m.bert_mc_gpt.large.bsz=%d.lr=%s.f' % (b, l))
            if 'model.tar.gz' not in os.listdir(ckpt):
                print("Oops: %s" % ckpt)
                continue

            metric_file = sorted([metrics for metrics in os.listdir(ckpt) if 'metrics_epoch_' in metrics])[-1]
            with open(os.path.join(ckpt, metric_file), 'r') as f:
                metric_data = json.load(f)

            best_epoch, best_rew = metric_data['best_epoch'], metric_data['best_validation_start_acc']

            # Add to mode_dict
            mode_dict[(l, b)] = (best_epoch, best_rew)

    # Write Report
    with open('graphs/report.md', 'w') as f:
        for lr, bsz in mode_dict:
            f.write("\tLR: %s\tBSZ: %d\tBest Epoch: %d\tBest Validation: %.5f\n" % (lr, bsz,
                                                                                    mode_dict[(lr, bsz)][0],
                                                                                    mode_dict[(lr, bsz)][1]))