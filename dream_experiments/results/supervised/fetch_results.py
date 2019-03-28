"""
fetch_results.py

Obtain supervised learning results (best validation accuracy) and aggregate over sweeps to find best HP settings.
"""
import argparse
import json
import os


ROOT_DIR = "/checkpoint/siddk/debate/dream"

MODES = ['a', 'b', 'ab', 'abab', 'ababab']
LR = [1e-5, 2e-5, 3e-5, 5e-5, 5e-6]
BSZ = [16, 32]


def parse_args():
    p = argparse.ArgumentParser("Dream Result Fetcher")
    return p.parse_args()


if __name__ == "__main__":
    # Parse Args
    args = parse_args()

    # Iterate through Modes and Collect Results
    mode_dict, result_dirs = {m: {b: {} for b in BSZ} for m in MODES}, os.listdir(ROOT_DIR)
    for m in MODES:
        # Iterate through Mode-Specific Directories, and Pull Out Learning Rate
        mode_dirs = [os.path.join(ROOT_DIR, mdir) for mdir in result_dirs if ('.%s.' % m) in mdir]
        for mdir in mode_dirs:
            # Pull Out Learning Rate, Batch Size
            lr = mdir.split('=')[-3][:-4]
            bsz = int(mdir.split('=')[-2][:-2])

            # Iterate through Directory + Pull Out Last-Epoch Metrics
            metric_file = os.path.join(mdir, 'metrics.json')
            with open(os.path.join(mdir, metric_file), 'r') as f:
                metric_data = json.load(f)
            best_epoch, best_val = metric_data['best_epoch'], metric_data['best_validation_start_acc']

            # Add to mode_dict
            mode_dict[m][bsz][lr] = (best_epoch, best_val)

    # Write Report
    with open('graphs/report.md', 'w') as f:
        for m in MODES:
            f.write("Mode - %s\n" % m)
            f.write("-" * 30 + "\n")
            for b in BSZ:
                for lr in mode_dict[m][b]:
                    f.write("\tBSZ: %d\tLR: %s\tBest Epoch: %d\tBest Validation: %.3f\n" % (b, lr,
                                                                                            mode_dict[m][b][lr][0],
                                                                                            mode_dict[m][b][lr][1]))
                f.write("\n")
            f.write("\n")
