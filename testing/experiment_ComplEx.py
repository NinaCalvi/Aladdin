#a lot of this is taken from sameh's code

import os
import itertools
import gzip
import numpy as np
from tqdm import tqdm

import argparse
import utils

from ..embeddings import *
from libkge import KgDataset


def main():
    parser = argparse.ArgumentParser('BioLinkPred', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', action='store', required=True, type=str)

    #model params
    parser.add_argument('--model', '-m', action='store', type=str, default='complex',
                        choices=['distmult', 'complex', 'transe', 'cp'])
    parser.add_argument('--mcl', action='store', type=bool, default=False)

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100)

    data = parser.data
    emb_size = parser.embedding_size
    batch_size = parser.batch_size

    if data == 'pse':
        dataset = utils.load_pse_dataset()

    train_data = dataset.data["bench_train"]
    valid_data = dataset.data["bench_valid"]
    test_data = dataset.data["bench_test"]

    nb_ents = dataset.get_ents_count()
    nb_rels = dataset.get_rels_count()
    #NUMBER OF ENTITIES IS GIVEN SAME AS BETWEEN OBJECTS AND SUBJECTS
    #THIS MAY BE ALSO BE DUE TO THE SYMMETRIC/AUGMENTATION SITUATION

    bench_idx_data = np.concatenate([train_data, valid_data, test_data])

    #need to pass size of embeddings
    if parser.model == 'complex':
        if parser.mcl:
            model = ComplEx_MC((nb_ents, nb_rels, nb_ents), emb_size)
        else:
            model = ComplEx((nb_nets, nb_rels_ nb_ents), emb_size)
