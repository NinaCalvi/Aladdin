import os
import pickle

from libkge import KgDataset

dataset = utils.load_pse_dataset('pse')

ent_mappings = dataset.ent_mappings
rel_mappings = dataset.rel_mappings

with open('pse_entity2idx.pkl', 'wb') as f:
    picke.dump(ent_mappings, f)
with open('pse_rel2idx.pkl', 'wb') as f:
    pickle.dump(rel_mappings, f)
