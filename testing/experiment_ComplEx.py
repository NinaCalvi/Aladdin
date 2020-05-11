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


def main(argv):
    parser = argparse.ArgumentParser('BioLinkPred', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', action='store', required=True, type=str)

    #model params
    parser.add_argument('--model', '-m', action='store', type=str, default='complex',
                        choices=['distmult', 'complex', 'transe', 'cp'])
    parser.add_argument('--mcl', action='store', type=bool, default=False)

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100)

    parser.add_argument('--eval-batch-size', '-B', action='store', type=int, default=None)

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--loss', '-l', action='store', type=str, default='pw_logistic')
    parser.add_argument('--regulariser', '-r', action='store', type=str, default='n3')
    parser.add_argument('--reg-weight', action='store', type=float, required=True)

    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--quiet', '-q', action='store_true', default=False)


    data = parser.data
    emb_size = parser.embedding_size
    batch_size = parser.batch_size

    nb_epochs = parser.epochs
    lr = parser.learning_rate
    optimizer_name = parser.optimizer
    loss = parser.loss
    regulariser = parser.regulariser
    reg_weight = parser.reg_weight

    seed = praser.seed
    quiet = parser.quiet

    args = parser.parse_args(argv)

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

    optimizer_factory = {
        'adagrad': lambda: optim.Adagrad(parameter_lst, lr=learning_rate),
        'adam': lambda: optim.Adam(parameter_lst, lr=learning_rate),
        'sgd': lambda: optim.SGD(parameter_lst, lr=learning_rate)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name]()

    #need to pass size of embeddings
    if parser.model == 'complex':
        if parser.mcl:
            model = ComplEx_MC((nb_ents, nb_rels, nb_ents), emb_size)
        else:
            model = ComplEx((nb_nets, nb_rels_ nb_ents), emb_size)
