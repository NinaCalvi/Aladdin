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
from libkge.io import load_kg_file

from torch import optim
import torch
from torch.optim.lr_scheduler import ExponentialLR

import json
import pandas as pd

import logging
import sys


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["H@1"]:.6f}\tH@3 {metrics["H@3"]:.6f}\tH@10 {metrics["H@10"]:.6f}\tAU-ROC_raw {metrics["AU-ROC_raw"]:.6f}\tAU-ROC_fil {metrics["AU-ROC_fil"]:.6f}'

def metrics_str_auc(metrics):
    return f'AUC-ROC {metrics["AUC-ROC"]:.6f}\tAP {metrics["AP"]:.6f}\tP@50 {metrics["P@50"]:.6f}\tAUC_PR {metrics["AUC_PR"]:.6f}'


def main(argv, bayesian=False):
    parser = argparse.ArgumentParser('BioLinkPred', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', action='store', required=True, type=str, choices=['pse', 'fb15', 'fb15k2', 'wn18rr', 'wn18', 'covid'])

    #model params
    parser.add_argument('--model', '-m', action='store', type=str, default='complex',
                        choices=['distmult', 'complex', 'transe', 'cp', 'trivec', 'tucker', 'rotate'])
    parser.add_argument('--mcl', action='store', type=bool, default=False)

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--rel-emb-size', action='store', type=int, default=100)

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
    parser.add_argument('--tucker-reg-weight', action='store', type=float, default=-1.0)

    parser.add_argument('--nb-negs', action='store', type=int, default=6)
    parser.add_argument('--transe-norm', action='store', type=str, default='l1')

    parser.add_argument('--loss-margin', action='store', type=float, default=1.0)

    parser.add_argument('--seed', action='store', type=int, default=5)
    parser.add_argument('--valid', action='store_true', default=False)
    parser.add_argument('--auc', action='store_true', default=False)
    parser.add_argument('--harder', action='store_true', default=False) #harder auc - i.e. 10:1 and 5:1 neg-to-pos
    parser.add_argument('--rare', action='store_true', default=False) #test mrr etc on rare entties
    parser.add_argument('--valid-stp', action='store', type=int, default=50)
    parser.add_argument('--label-smoothing', action='store', type=float, default=0.0)

    parser.add_argument('--input-dropout', action='store', type=float, default=0.3)
    parser.add_argument('--hidden-dropout1', action='store', type=float, default=0.4)
    parser.add_argument('--hidden-dropout2', action='store', type=float, default=0.5)
    parser.add_argument('--lr_decay', action='store', type=float, default=0.0)

    parser.add_argument('--quiet', '-q', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--pret', action='store_true', default=False) #use pretrained embeddings
    parser.add_argument('--save_model_name', action='store', type=str, default='Empty')
    parser.add_argument('--testrel', action='store', type=str, default='Empty')
    parser.add_argument('--rel_type', action='store', type=str, default='Empty', help='path of dictionary with relation-to-typ')
    parser.add_argument('--ent_type', action='store', type=str, default='Empty', help='path of csv with entity-to-type')
    parser.add_argument('--neg_by_type', action='store_true', default=False, help='if want to use entity types for AUC. AUC, ent_type and rel_type must be submitted')
    
    parser.add_argument('--transe_pret', action='store', type=str, default=None)
    



    args = parser.parse_args(argv)

    data = args.data

    SAVE_PATH = os.path.join(os.getcwd(),f'best_models/{data}/' )

    emb_size = args.embedding_size
    batch_size = args.batch_size

    nb_epochs = args.epochs
    lr = args.learning_rate
    lr_decay = args.lr_decay

    optimizer_name = args.optimizer
    loss = args.loss
    regulariser = args.regulariser
    reg_weight = args.reg_weight

    seed = args.seed
    quiet = args.quiet


    print('MCL', args.mcl)


    logger.info(f'Valid: {args.valid}')



    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(4)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    dataset = utils.load_pse_dataset(data)


    train_data = torch.tensor(dataset.data["train"])
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
        model_dict = {
            'complex': lambda: ComplEx_MC((nb_ents, nb_rels, nb_ents), emb_size, optimizer_name, args.pret),
            'transe': lambda: TransE_MC((nb_ents, nb_rels, nb_ents), emb_size, optimizer_name, norm_ = args.transe_norm),
            'distmult': lambda: DistMult_MC((nb_ents, nb_rels, nb_ents), emb_size, optimizer_name),
            'trivec': lambda: TriVec_MC((nb_ents, nb_rels, nb_ents), emb_size, optimizer_name),
            'tucker': lambda: TuckEr_MC((nb_ents, nb_rels, nb_ents), emb_size, args.rel_emb_size, optimizer_name, input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, hidden_dropout2=args.hidden_dropout2),
            'rotate': lambda: RotatE_MC((nb_ents, nb_rels, nb_ents), emb_size,optimizer_name)
        }
    else:
        model_dict = {
            'complex': lambda: ComplEx((nb_ents, nb_rels, nb_ents), emb_size, loss, device, optimizer_name, args),
            'transe': lambda: TransE((nb_ents, nb_rels, nb_ents), emb_size, loss, device, optimizer_name, args, norm_=args.transe_norm),
            'distmult': lambda: DistMult((nb_ents, nb_rels, nb_ents), emb_size, loss, device, optimizer_name, args),
            'trivec': lambda: TriVec((nb_ents, nb_rels, nb_ents), emb_size, loss, device, optimizer_name, args),
            'tucker': lambda: TuckEr((nb_ents, nb_rels, nb_ents), emb_size, args.rel_emb_size, loss, device, optimizer_name, args, input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, hidden_dropout2=args.hidden_dropout2),
            'rotate': lambda: RotatE((nb_ents, nb_rels, nb_ents), emb_size, loss, device, optimizer_name, args)
        }


    model = model_dict[args.model]()
    if args.load:
        model.load_state_dict(torch.load(SAVE_PATH + args.save_model_name + '.pt'))
    else:
        model.init()

    model.to(device)

    ###if AUC then we are just testing to see this performance##
    if args.load:
        logger.info('auc and lod happening')
        for dataset_name, data in dataset.data.items():
            logger.info(f'dataset name \t{dataset_name}')
            if dataset_name == 'test':
                logger.info(f'in evalute for dataset: \t{dataset_name}')
                if args.data == 'pse':
                    batch_size = 5000
                test_batch_size = 2048
                metrics_test = evaluate(model, torch.tensor(data), bench_idx_data, test_batch_size, device, auc = False)
                logger.info(f'TEST RESULTS')
                logger.info(f'Error \t{dataset_name} results \t{metrics_to_str(metrics_test)}')
                logger.info(f'===========')
                
                if args.rare:
                    
                    very_rare_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_rare_head.txt.gz'))
                    very_rare_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_rare_tail.txt.gz'))

                    dataset.load_triples(very_rare_head, tag='rare_head')
                    dataset.load_triples(very_rare_tail, tag='rare_tail')

                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'VERY RARE RESULTS')
                    logger.info(f'Error VERYRARE HEAD results \t{metrics_to_str(metrics_head)}')
                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error VERYRARE TAIL results \t{metrics_to_str(metrics_tail)}')


                    rare_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/rare_head_2.txt.gz'))
                    rare_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/rare_tail_2.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(rare_head, tag='rare_head')
                    dataset.load_triples(rare_tail, tag='rare_tail')

                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'RARE RESULTS')
                    logger.info(f'Error RARE HEAD results \t{metrics_to_str(metrics_head)}')
                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error RARE TAIL results \t{metrics_to_str(metrics_tail)}')


                    common_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/common_head.txt.gz'))
                    common_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/common_tail.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(common_head, tag='rare_head')
                    dataset.load_triples(common_tail, tag='rare_tail')
                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'COMMON RESULTS')
                    logger.info(f'Error COMMON- HEAD results \t{metrics_to_str(metrics_head)}')

                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error COMMON- TAIL results \t{metrics_to_str(metrics_tail)}')

                    very_common_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_common_head.txt.gz'))
                    very_common_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_common_tail.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(very_common_head, tag='rare_head')
                    dataset.load_triples(very_common_tail, tag='rare_tail')
                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'VERYCOMMON RESULTS')
                    logger.info(f'Error VERY-COMMON HEAD results \t{metrics_to_str(metrics_head)}')

                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error VERY-COMMON TAIL results \t{metrics_to_str(metrics_tail)}')
                    
                    very_very_common_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_very_common_head.txt.gz'))
                    very_very_common_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_very_common_tail.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(very_very_common_head, tag='rare_head')
                    dataset.load_triples(very_very_common_tail, tag='rare_tail')
                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'VERYVERYCOMMON RESULTS')
                    logger.info(f'Error VERYVERY-COMMON HEAD results \t{metrics_to_str(metrics_head)}')

                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error VERYVERY-COMMON TAIL results \t{metrics_to_str(metrics_tail)}')
                
                if args.testrel != 'Empty':
                    metrics_rel = evaluate(model, torch.tensor(data), bench_idx_data, test_batch_size, device, auc = False, rel_file = args.testrel)
                
                
                
                
                if args.auc: 
                    
                    if args.neg_by_type:
                        if args.rel_type != 'Empty' and args.ent_type != 'Empty':
                            logger.info('APPLYING TYPE INFORMATION TO NEG SAMPLING')
                            with open(args.rel_type, 'r') as f:
                                rel_type = json.load(f)
                            ent_type = pd.read_csv(args.ent_type)
                            neg_by_type = utils.process_relation_entities(rel_type, ent_type)
                        else:
                            logger.info('NEED TO SUPPLY REL TYPE AND ENT TYPE FILE')
                            return
                            
                    else:
                        neg_by_type = None
                        rel_type = None
                    if args.harder:        
                        metrics_five, metrics_ten = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device, auc = args.auc, harder=args.harder, rel_type = rel_type, neg_by_type = neg_by_type, dataset_dict=dataset)
                        logger.info('CLASSIFICATION METRICS FIVE')
                        logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics_five)}')
                        logger.info('CLASSIFICATION METRICS TEN')
                        logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics_ten)}')
#                     return
                
               
                    metrics = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device, auc = args.auc, rel_type = rel_type, neg_by_type = neg_by_type, dataset_dict=dataset)
                    logger.info('CLASSIFICATION METRICS')
                    logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics)}')
                return



    optimizer_factory = {
        'adagrad': lambda: optim.Adagrad(model.parameters(), lr=lr),
        'adam': lambda: optim.Adam(model.parameters(), lr=lr),
        'sgd': lambda: optim.SGD(model.parameters(), lr=lr)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name]()

    if lr_decay > 0:
        scheduler = ExponentialLR(optimizer, lr_decay)
    else:
        scheduler = None


    model.train()
    train.train(model, regulariser, optimizer, train_data, valid_data, bench_idx_data, args, scheduler=scheduler)
    logger.info(f'is bad performing {train.BAD_PERFORMING}')

    if args.save_model_name != 'Empty':
        torch.save(model.state_dict(), SAVE_PATH + args.save_model_name + '.pt')
        logger.info(f'Save model in {SAVE_PATH + args.save_model_name}')
        for dataset_name, data in dataset.data.items():
            logger.info(f'dataset name \t{dataset_name}')
            if dataset_name == 'test':
                logger.info(f'in evalute for dataset: \t{dataset_name}')
                if args.data == 'pse':
                    batch_size = 5000
                test_batch_size = 2048
                metrics_test = evaluate(model, torch.tensor(data), bench_idx_data, test_batch_size, device, auc = False)
                logger.info(f'TEST RESULTS')
                logger.info(f'Error \t{dataset_name} results \t{metrics_to_str(metrics_test)}')
                logger.info(f'===========')
                # if args.harder:
                #     metrics_five, metrics_ten = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device, auc = args.auc, harder=args.harder)
                #     logger.info('CLASSIFICATION METRICS FIVE')
                #     logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics_five)}')
                #     logger.info('CLASSIFICATION METRICS TEN')
                #     logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics_ten)}')
                #     return
#                 if args.auc:
#                     metrics = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device, auc = args.auc)
#                     logger.info('CLASSIFICATION METRICS')
#                     logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics)}')
                if args.auc and args.harder:
                    if args.neg_by_type:
                        if args.rel_type != 'Empty' and args.ent_type != 'Empty':
                            logger.info('APPLYING TYPE INFORMATION TO NEG SAMPLING')
                            with open(args.rel_type, 'r') as f:
                                rel_type = json.load(f)
                            ent_type = pd.read_csv(args.ent_type)
                            neg_by_type = utils.process_relation_entities(rel_type, ent_type)
                        else:
                            logger.info('NEED TO SUPPLY REL TYPE AND ENT TYPE FILE')
                            return
                            
                    else:
                        neg_by_type = None
                        rel_type = None
                            
                    metrics_five, metrics_ten = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device, auc = args.auc, harder=args.harder, rel_type = rel_type, neg_by_type = neg_by_type)
                    logger.info('CLASSIFICATION METRICS FIVE')
                    logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics_five)}')
                    logger.info('CLASSIFICATION METRICS TEN')
                    logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics_ten)}')
#                     return
                
                if args.auc:
                    metrics = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device, auc = args.auc, rel_type = rel_type, neg_by_type = neg_by_type)
                    logger.info('CLASSIFICATION METRICS')
                    logger.info(f'Error \t{dataset_name} results\t{metrics_str_auc(metrics)}')
                if args.rare:

                    
                    very_rare_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_rare_head.txt.gz'))
                    very_rare_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_rare_tail.txt.gz'))

                    dataset.load_triples(very_rare_head, tag='rare_head')
                    dataset.load_triples(very_rare_tail, tag='rare_tail')

                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'VERY RARE RESULTS')
                    logger.info(f'Error VERYRARE HEAD results \t{metrics_to_str(metrics_head)}')
                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error VERYRARE TAIL results \t{metrics_to_str(metrics_tail)}')


                    rare_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/rare_head_2.txt.gz'))
                    rare_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/rare_tail_2.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(rare_head, tag='rare_head')
                    dataset.load_triples(rare_tail, tag='rare_tail')

                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'RARE RESULTS')
                    logger.info(f'Error RARE HEAD results \t{metrics_to_str(metrics_head)}')
                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error RARE TAIL results \t{metrics_to_str(metrics_tail)}')


                    common_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/common_head.txt.gz'))
                    common_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/common_tail.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(common_head, tag='rare_head')
                    dataset.load_triples(common_tail, tag='rare_tail')
                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'COMMON RESULTS')
                    logger.info(f'Error COMMON- HEAD results \t{metrics_to_str(metrics_head)}')

                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error COMMON- TAIL results \t{metrics_to_str(metrics_tail)}')

                    very_common_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_common_head.txt.gz'))
                    very_common_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_common_tail.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(very_common_head, tag='rare_head')
                    dataset.load_triples(very_common_tail, tag='rare_tail')
                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'VERYCOMMON RESULTS')
                    logger.info(f'Error VERY-COMMON HEAD results \t{metrics_to_str(metrics_head)}')

                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error VERY-COMMON TAIL results \t{metrics_to_str(metrics_tail)}')
                    
                    very_very_common_head = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_very_common_head.txt.gz'))
                    very_very_common_tail = load_kg_file(os.path.join(os.getcwd(), f'testing/data/{args.data}/very_very_common_tail.txt.gz'))

                    del dataset.data['rare_head']
                    del dataset.data['rare_tail']

                    dataset.load_triples(very_very_common_head, tag='rare_head')
                    dataset.load_triples(very_very_common_tail, tag='rare_tail')
                    rare_head = torch.tensor(dataset.data['rare_head'])
                    rare_tail = torch.tensor(dataset.data['rare_tail'])

                    metrics_head = evaluate(model, rare_head, bench_idx_data, test_batch_size, device, mode='head')
                    logger.info(f'VERYVERYCOMMON RESULTS')
                    logger.info(f'Error VERYVERY-COMMON HEAD results \t{metrics_to_str(metrics_head)}')

                    metrics_tail = evaluate(model, rare_tail, bench_idx_data, test_batch_size, device, mode='tail')
                    logger.info(f'Error VERYVERY-COMMON TAIL results \t{metrics_to_str(metrics_tail)}')
                
                if args.testrel != 'Empty':
                    metrics_rel = evaluate(model, torch.tensor(data), bench_idx_data, test_batch_size, device, auc = False, rel_file = args.testrel)



                return

        return

    # if args.mcl:
    #     train.train_mc(model, regulariser, optimizer, train_data, args)
    # else:
    #     train.train_not_mc(model, regulariser, optimizer, train_data, args)

    logger.info(f'Done training')
    if not train.BAD_PERFORMING:
        for dataset_name, data in dataset.data.items():
            if (dataset_name == 'test') or (dataset_name == 'valid'):
                logger.info(f'in evalute for dataset: \t{dataset_name}')
                if args.data == 'pse':
                    batch_size = 2048
                if args.data == 'covid':
                    batch_size = 2048
                if args.model == 'rotate':
                    batch_size = 200
                metrics = evaluate(model, torch.tensor(data), bench_idx_data, batch_size, device, auc = args.auc)
                logger.info(f'Error \t{dataset_name} results\t{metrics_to_str(metrics)}')

                if dataset_name == 'valid' or (dataset_name == 'test' and not args.valid):


                    if bayesian:
                        return metrics
    else:
        logger.info(f'Not checking test, since interrupted')
        if bayesian:
            return {'MRR': train.bad_mrr, 'H@1': -1, 'H@3': -1, 'H@10': -1}

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
