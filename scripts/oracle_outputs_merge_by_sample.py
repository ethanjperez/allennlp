import argparse
import os
import pickle
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--prefix",
                    default='tmp/race.best.f/oracle_outputs.c=concat.d=Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ',
                    type=str,
                    help="The prefix for files to load.")
args = parser.parse_args()

postfixes = ['dev.pkl', 'test.pkl', 'train.0.pkl', 'train.1.pkl', 'train.2.pkl', 'train.3.pkl', 'train.4.pkl', 'train.5.pkl', 'train.6.pkl', 'train.7.pkl', 'train.8.pkl', 'train.9.pkl']

files = [args.prefix + '.' + postfix for postfix in postfixes]
save_file = args.prefix + '.all.pkl'
assert not os.path.exists(save_file), 'Save file already exists! Not overriding: ' + save_file


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def fix_sample_id(sample_id: str) -> str:
    file_parts = sample_id.split('/')[2:]
    for split in ['train', 'dev', 'test']:
        if split in file_parts[0]:
            file_parts[0] = split
            break
    return '/'.join(file_parts)


oracle_outputs = []
for file in files:
    print('Reading', file, '...')
    with open(file, 'rb') as f:
        oracle_outputs.append(pickle.load(f))

print('Merging dictionaries...')
all_oracle_outputs = merge_dicts(*oracle_outputs)

print('Correcting sample_ids and moving to CPU...')
fixed_all_oracle_outputs = {}
for sample_id, sample_oracle_outputs in all_oracle_outputs.items():
    if 'datasets' in sample_id:
        sample_id = fix_sample_id(sample_id)
    fixed_all_oracle_outputs[sample_id] = {}
    for prev_turns_str, oracle_dict in sample_oracle_outputs.items():
        fixed_all_oracle_outputs[sample_id][prev_turns_str] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                                               for k, v in oracle_dict.items()}

print('Saving to file:', save_file, '...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

example_key = list(fixed_all_oracle_outputs.keys())[-1]
print('Example key:', example_key)
example_prev_turns_strs = list(fixed_all_oracle_outputs[example_key].keys())
print('Example prev_turns_strs:', example_prev_turns_strs)
print('Example oracle_output dict:', fixed_all_oracle_outputs[example_key][example_prev_turns_strs[0]])
print('Done!')
