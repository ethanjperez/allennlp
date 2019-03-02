import os
import pickle
import torch

prefix = 'tmp/race.best.f/oracle_outputs.c=concat.d=B_A_B_A_B_A_B_A'
postfixes = ['dev.pkl', 'test.pkl', 'train.0.pkl', 'train.1.pkl', 'train.2.pkl', 'train.3.pkl', 'train.4.pkl', 'train.5.pkl', 'train.6.pkl', 'train.7.pkl', 'train.8.pkl', 'train.9.pkl']

files = [prefix + '.' + postfix for postfix in postfixes]
save_file = prefix + '.' + 'all.pkl'
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
    for cum_turn_str, oracle_dict in sample_oracle_outputs.items():
        for k, v in oracle_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            fixed_all_oracle_outputs[sample_id][cum_turn_str] = {k: v}

print('Saving to file...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

example_key = list(fixed_all_oracle_outputs.keys())[0]
print('Example key:', example_key)
print('Example cum_turn_strs:', fixed_all_oracle_outputs[example_key].keys())
example_cum_turn_str = list(fixed_all_oracle_outputs[example_key].keys())[0]
print('Example oracle_output dict:', fixed_all_oracle_outputs[example_key][example_cum_turn_str])

print('Done!')
