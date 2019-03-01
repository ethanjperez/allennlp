import os
import pickle
import torch

file = 'tmp/race.best.f/oracle_outputs.c=concat.d=8_AB_turns.all.pkl'
save_file = file[:-3] + 'cpu.pkl'
assert os.path.exists(file), 'Read file does not exist! Please correct file path ' + save_file
assert not os.path.exists(save_file), 'Save file already exists! Not overriding: ' + save_file

print('Reading', file, '...')
with open(file, 'rb') as f:
    oracle_output = pickle.load(f)

print('Moving tensors to cpu...')
for sample_id, sample_oracle_outputs in oracle_output.items():
    for cum_turn_str, oracle_dict in sample_oracle_outputs.items():
        for k, v in oracle_dict.items():
            oracle_output[sample_id][cum_turn_str][k] = v.cpu() if isinstance(v, torch.Tensor) else v

print('Saving to file...')
with open(save_file, 'wb') as f:
    pickle.dump(oracle_output, f, pickle.HIGHEST_PROTOCOL)

print('Saved to', save_file, ' !')
