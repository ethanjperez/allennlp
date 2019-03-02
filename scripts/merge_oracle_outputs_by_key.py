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


oracle_outputs = []
for file in files:
    print('Reading', file, '...')
    with open(file, 'rb') as f:
        oracle_outputs.append(pickle.load(f))

print('Merging dictionaries...')
all_oracle_outputs = merge_dicts(*oracle_outputs)

print('Correcting dictionary keys...')
fixed_all_oracle_outputs = {}
for k, v in all_oracle_outputs.items():
    if 'train' in k:
        for i in range(10):
            bad_str = 'train.' + str(i)
            if bad_str in k:
                fixed_k = k.replace(bad_str, 'train')
                fixed_all_oracle_outputs[fixed_k] = v
                break
    else:
        fixed_all_oracle_outputs[k] = v

print('Moving tensors to cpu...')
for sample_id, sample_oracle_outputs in fixed_all_oracle_outputs.items():
    for cum_turn_str, oracle_dict in sample_oracle_outputs.items():
        for k, v in oracle_dict.items():
            fixed_all_oracle_outputs[sample_id][cum_turn_str][k] = v.cpu() if isinstance(v, torch.Tensor) else v

print('Saving to file...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

print('Done!')
