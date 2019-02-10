import pickle
import os

folder = 'tmp/race.best.f/'
files = ['oracle_outputs.dev.pkl', 'oracle_outputs.test.pkl', 'oracle_outputs.train.0.pkl', 'oracle_outputs.train.1.pkl', 'oracle_outputs.train.2.pkl', 'oracle_outputs.train.3.pkl', 'oracle_outputs.train.4.pkl', 'oracle_outputs.train.5.pkl', 'oracle_outputs.train.6.pkl', 'oracle_outputs.train.7.pkl', 'oracle_outputs.train.8.pkl', 'oracle_outputs.train.9.pkl']

save_file = os.path.join(folder, 'oracle_outputs.all.corrected.pkl')
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
    with open(os.path.join(folder, file), 'rb') as f:
        oracle_outputs.append(pickle.load(f))

print('Merging dictionaries...')
all_oracle_outputs = merge_dicts(*oracle_outputs)

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
import ipdb; ipdb.set_trace()

print('Saving to file...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

print('Done!')
