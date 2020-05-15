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


import logging
import sys


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tAU-ROC_raw {metrics["AU-ROC_raw"]:.6f}\tAU-ROC_fil {metrics["AU-ROC_fil"]:.6f}'



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
    parser.add_argument('--learning-rate', '-lr', action='store', type=float, default=0.01)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--loss', '-l', action='store', type=str, default='pw_logistic')
    parser.add_argument('--regulariser', '-r', action='store', type=str, default='n3')
    parser.add_argument('--reg-weight', action='store', type=float, required=True)

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
            model = ComplEx((nb_nets, nb_rels_, nb_ents), emb_size)

    if parser.mcl:
        train.train_mc(model, regulariser, optimizer, train_data, args)
    else:
        train.train_not_mc(model, regulariser, optimizer, train_data, args)

    for dataset_name, data in dataset.data:
        metrics = evaluate(model, data, bench_idx_data, batch_size, device)
        logger.info(f'Error \t{dataset_name} results\t{metrics_to_str(metrics)}')

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
