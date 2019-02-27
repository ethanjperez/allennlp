import pickle
import os

prefix = 'tmp/race.best.f/oracle_outputs.c=concat.d='
postfixes = ['B_A_B_A_B_A_B_A.all.pkl', 'A_B_A_B_A_B_A_B.all.pkl']

files = [prefix + postfix for postfix in postfixes]
save_file = prefix + '8AB_turns.all.pkl'
assert not os.path.exists(save_file), 'Save file already exists! Not overriding: ' + save_file


def merge_dicts_by_key_and_value(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = dict_args[0]
    for dictionary in dict_args[1:]:
        for sample_id, oracle_outputs_per_turn in dictionary.items():
            result[sample_id].update(oracle_outputs_per_turn)
    return result


oracle_outputs = []
for file in files:
    print('Reading', file, '...')
    with open(file, 'rb') as f:
        oracle_outputs.append(pickle.load(f))

print('Merging dictionaries...')
all_oracle_outputs = merge_dicts_by_key_and_value(*oracle_outputs)

print('Correcting dictionary...')
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

print('Saving to file...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

print('Done!')
