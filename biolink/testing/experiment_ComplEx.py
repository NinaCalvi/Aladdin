#a lot of this is taken from sameh's code
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


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tMRR_pasquale {metrics["mrr_pasquale"]:.6f}\tAU-ROC_raw {metrics["AU-ROC_raw"]:.6f}\tAU-ROC_fil {metrics["AU-ROC_fil"]:.6f}'



def main(argv):
    parser = argparse.ArgumentParser('BioLinkPred', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', action='store', required=True, type=str, choices=['pse', 'fb15'])

    #model params
    parser.add_argument('--model', '-m', action='store', type=str, default='complex',
                        choices=['distmult', 'complex', 'transe', 'cp'])
    parser.add_argument('--mcl', action='store', type=bool, default=False)

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100)

    parser.add_argument('--eval-batch-size', '-B', action='store', type=int, default=None)

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning-rate', '-lr', action='store', type=float, default=0.01)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--loss', '-l', action='store', type=str, default='pw_logistic')
    parser.add_argument('--regulariser', '-r', action='store', type=str, default='n3')
    parser.add_argument('--reg-weight', action='store', type=float, required=True)

    parser.add_argument('--nb-negs', action='store', default=6)

    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--quiet', '-q', action='store_true', default=False)



    args = parser.parse_args(argv)


    data = args.data
    emb_size = args.embedding_size
    batch_size = args.batch_size

    nb_epochs = args.epochs
    lr = args.learning_rate
    optimizer_name = args.optimizer
    loss = args.loss
    regulariser = args.regulariser
    reg_weight = args.reg_weight

    seed = args.seed
    quiet = args.quiet



    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    dataset = utils.load_pse_dataset(data)


    train_data = torch.tensor(dataset.data["train"])
    print('train data type', train_data)
    valid_data = torch.tensor(dataset.data["valid"])
    test_data = torch.tensor(dataset.data["test"])

    nb_ents = dataset.get_ents_count()
    nb_rels = dataset.get_rels_count()
    #NUMBER OF ENTITIES IS GIVEN SAME AS BETWEEN OBJECTS AND SUBJECTS
    #THIS MAY BE ALSO BE DUE TO THE SYMMETRIC/AUGMENTATION SITUATION

    bench_idx_data = torch.cat((train_data, valid_data, test_data), 0)
    #need to pass size of embeddings
    # if parser.model == 'complex':
    if args.mcl:
        model_dict = {'complex': lambda: ComplEx_MC((nb_ents, nb_rels, nb_ents), emb_size)}
    else:
        model_dict = {'complex': lambda: ComplEx((nb_nets, nb_rels_, nb_ents), emb_size)}

    model = model_dict[args.model]()
    model.to(device)

    optimizer_factory = {
        'adagrad': lambda: optim.Adagrad(model.parameters(), lr=lr),
        'adam': lambda: optim.Adam(model.parameters(), lr=lr),
        'sgd': lambda: optim.SGD(model.parameters(), lr=lr)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name]()

    if args.mcl:
        train.train_mc(model, regulariser, optimizer, train_data, args)
    else:
        train.train_not_mc(model, regulariser, optimizer, train_data, args)

    logger.info(f'Done training')

    for dataset_name, data in dataset.data.items():
        if dataset_name == 'test':
            logger.info(f'in evalute for dataset: \t{dataset_name}')
            metrics = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device)
            logger.info(f'Error \t{dataset_name} results\t{metrics_to_str(metrics)}')

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
