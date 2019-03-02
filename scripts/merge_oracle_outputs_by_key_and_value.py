import os
import pickle
import torch

prefix = 'tmp/race.best.f/oracle_outputs.c=concat.d='
postfixes = ['B_A_B_A_B_A_B_A.all.pkl', 'A_B_A_B_A_B_A_B.all.pkl']
max_turns = None

files = [prefix + postfix for postfix in postfixes]
save_file = prefix + '1_AB_turns.all.pkl'
assert not os.path.exists(save_file), 'Save file already exists! Not overriding: ' + save_file


def merge_dicts_by_key_and_value(*dict_args, max_turns=None):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = dict_args[0]
    for dictionary in dict_args[1:]:
        for sample_id, oracle_outputs_per_turn in dictionary.items():
            result[sample_id].update(oracle_outputs_per_turn)
    if max_turns is not None:
        for sample_id, oracle_outputs_per_turn in result.items():
            filtered_results = {}
            for turn, oracle_outputs in result[sample_id].items():
                if len(turn) <= max_turns:
                    filtered_results[turn] = oracle_outputs
            result[sample_id] = filtered_results
    return result


oracle_outputs = []
for file in files:
    print('Reading', file, '...')
    with open(file, 'rb') as f:
        oracle_outputs.append(pickle.load(f))

print('Merging dictionaries...')
all_oracle_outputs = merge_dicts_by_key_and_value(*oracle_outputs, max_turns=max_turns)

print('Correcting dictionary keys and moving tensors to CPU...')
fixed_all_oracle_outputs = {}
for sample_id, sample_oracle_outputs in all_oracle_outputs.items():
    for cum_turn_str, oracle_dict in sample_oracle_outputs.items():
        for k, v in oracle_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            fixed_k = k
            if 'train' in k:
                for i in range(10):
                    bad_str = 'train.' + str(i)
                    if bad_str in k:
                        fixed_k = k.replace(bad_str, 'train')
                        break
            fixed_all_oracle_outputs[sample_id][cum_turn_str][fixed_k] = v

print('Saving to file...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

example_key = list(fixed_all_oracle_outputs.keys())[0]
print('Example key:', example_key)
print('Example cum_turn_strs:', fixed_all_oracle_outputs[example_key].keys())
example_cum_turn_str = list(fixed_all_oracle_outputs[example_key].keys())[0]
print('Example oracle_output dict:', fixed_all_oracle_outputs[example_key][example_cum_turn_str])

print('Done!')
