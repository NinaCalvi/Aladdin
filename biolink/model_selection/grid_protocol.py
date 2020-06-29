import os
import itertools
import gzip
import numpy as np
from tqdm import tqdm

import argparse
from biolink.utility import utils, train

from biolink.embeddings import *
from biolink.eval import evaluate
# from biolink.utility train import *
from libkge import KgDataset
from torch import optim
import torch


import logging
import sys

import constants as cn
from sklearn.model_selection import ParameterGrid

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


def choose(dictionary, parameter_key, parameter_constant):
    if parameter_key in dictionary:
        return dictionary[parameter_key]
    else:
        return parameter_constant

def select_best_model(model, model_params_options, X_train, X_val, X_test, nb_ents, nb_rels, filter=True):
    bench_idx_data = torch.cat((train_data, valid_data, test_data), 0)
    model_params_grid = list(ParameterGrid(model_params_options))

    for model_params in model_params_grid:

        logger.info(f'model parameters \t{model_params}')

        emb_size = choose(model_params, 'embedding_size', cn.EMB_SIZE)
        batch_size = choose(model_params, 'batch_size', cn.BATCH_SIZE)
        nb_epochs = choose(model_params, 'nb_epochs', cn.NB_EPOCHS)
        lr = choose(model_params, 'learning_rate', cn.LR)

        optimizer = choose(model_params, 'optimizer', cn.OPTIMIZER)
        loss = choose(model_params, 'loss', cn.LOSS)
        regulariser = choose(model_params, 'regulariser', cn.REG)
        reg_weight = choose(model_params, 'reg_weight', cn.REG_WEIGHT)

        seed = choose(model_params, 'seed', cn.SEED)

        transe_norm = choose(model_params, 'transe_norm', cn.TRANSE_NORM)

        quiet = True

        mcl = choose(model_params, 'mcl', cn.MCL)
        valid = True


        # set the seeds
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device: {device}')

        if mcl:
            model_dict = {'complex': lambda: ComplEx_MC((nb_ents, nb_rels, nb_ents), emb_size),
            'transe': lambda: TransE_MC((nb_ents, nb_rels, nb_ents), emb_size, norm_ = transe_norm)
        else:
            model_dict = {'complex': lambda: ComplEx((nb_ents, nb_rels, nb_ents), emb_size, loss, device, args),
            'transe': lambda: TransE((nb_ents, nb_rels, nb_ents), emb_size, loss, device, args, norm_=transe_norm)

        model = model_dict[args.model]()
        model.to(device)

        optimizer_factory = {
            'adagrad': lambda: optim.Adagrad(model.parameters(), lr=lr),
            'adam': lambda: optim.Adam(model.parameters(), lr=lr),
            'sgd': lambda: optim.SGD(model.parameters(), lr=lr)
        }

        assert optimizer_name in optimizer_factory
        optimizer = optimizer_factory[optimizer_name]()


        train.train(model, regulariser, optimizer, train_data, valid_data, bench_idx_data, args)

        # if args.mcl:
        #     train.train_mc(model, regulariser, optimizer, train_data, args)
        # else:
        #     train.train_not_mc(model, regulariser, optimizer, train_data, args)

        logger.info(f'Done training')

        for dataset_name, data in dataset.data.items():
            if dataset_name == 'test':
                logger.info(f'in evalute for dataset: \t{dataset_name}')
                metrics = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device)
                logger.info(f'Error \t{dataset_name} results\t{metrics_to_str(metrics)}')
