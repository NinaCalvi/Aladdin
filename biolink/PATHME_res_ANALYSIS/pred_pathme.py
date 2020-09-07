import json
import pandas as pd
import numpy as np
import itertools
from biolink.utility import utils
import torch

from biolink.embeddings import *
from biolink.eval import evaluate
# from biolink.utility train import *
from libkge import KgDataset
from libkge.io import load_kg_file

import logging
import os
import sys


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def main():
    '''
    Evaluation method immediately returns the metrics wanted
    Parameters:
    ---------------
    all_triples: all triples in train/test/dev for calculaing filtered options
    rel_type: dictionary {relation: type}
    neg_by_type: dictionary {rel_tpe: {head_ents: [pool of accepted head entities], tail_ents: [pool of accepted tail entities]}}
    '''

    #store the labels for subject_predicate and predicte_object that were seen in training
    #some sp and po situations may have more than one label assigned to it (i.e. may link to different entitites)
    #when calculating the score we would like to filter them out


    dataset = utils.load_pse_dataset('pathme')
    train_data = torch.tensor(dataset.data["train"])
    valid_data = torch.tensor(dataset.data["valid"])
    test_data = torch.tensor(dataset.data["test"])


    all_triples = torch.cat((train_data, valid_data, test_data), 0)

    predicate_indeces = list(set(all_triples[:, 1].numpy()))
    logger.info(f'length predicate_instanes \t{len(predicate_indeces)}')
    # print(predicate_indeces)
    # print(type(predicate_indeces[0]))
    se_facts_full_dict = {se: set() for se in predicate_indeces}

    logger.info('predicate instances done')


    for instance in all_triples:
        s, p, o = instance.numpy()
        se_facts_full_dict[p].add((s, p, o))



    emb_size = 200
    batch_size = 2048

    nb_epochs = 100
    lr = 0.136

    optimizer_name = 'adagrad'
    regulariser = 'n3'
    reg_weight = 0.062
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    nb_ents = dataset.get_ents_count()
    nb_rels = dataset.get_rels_count()
    model = ComplEx_MC((nb_ents, nb_rels, nb_ents), emb_size, optimizer_name, pret = False)
    model.load_state_dict(torch.load('/home/acalvi/Dissertation/Aladdin/biolink/best_models/pathme/cplex_batch2048_es_200_reg_n3_rw_062_lr_136_pathme.pt'))
    model.to(device)


    #logger.info(f'all tiples \t{all_triples.size()}')
    #test_triples = test_triples.numpy()
    #logger.info('make test triples numpy')


    rel_type = '/home/acalvi/Dissertation/Aladdin/biolink/testing/data/pathme/re_type_pathme.json'
    ent_type = '/home/acalvi/Dissertation/Aladdin/biolink/testing/data/pathme/ent_type_pathme.csv'
    with open(rel_type, 'r') as f:
        rel_type = json.load(f)
    ent_type = pd.read_csv(ent_type)
    neg_by_type = utils.process_relation_entities(rel_type, ent_type)


    if neg_by_type is not None and rel_type is not None:
        logger.info('calculating ents combinations')
        ents_combinations = dict()
        for r, rt in rel_type.items():
#             head_ents = neg_by_type[rt]['head']
#             tail_ents = neg_by_type[rt]['tail']
#             ent_combinations[r] = np.array([[d1, d2] for d1, d2 in list(itertools.product(head_ents, tail_ents)) if d1 != d2])
            ents_combinations[dataset.rel_mappings[r]] = neg_by_type[rt]

    # softmax = nn.Softmax(dim=1)


    logger.info('about to start predicting')

    #for drug_increases_protein
    logger.info('RELATION drug_increases_protein')
    pred = dataset.rel_mappings['drug_association_protein']
    predicate_all_facts_set = se_facts_full_dict[pred]
    head = []
    tail = []
    for h in ents_combinations[pred]['head']:
        head.append(dataset.ent_mappings[h])
    for t in ents_combinations[pred]['tail']:
        tail.append(dataset.ent_mappings[t])
    logger.info(f'tails {tail[:100]}')
    logger.info(len(tail))
    ents_combinations = list(itertools.product(head, tail))
    logger.info(f'length ents combs {len(ents_combinations)}')
    se_test_facts_all_dict = dict()
    for d1, d2 in ents_combinations:
        if (d1, pred, d2) not in predicate_all_facts_set:
            if d1 in se_test_facts_all_dict:
                se_test_facts_all_dict[d1] = np.concatenate([se_test_facts_all_dict[d1],[[d1, pred, d2]]], axis=0)
            else:
                se_test_facts_all_dict[d1] = np.array([[d1, pred, d2]])

    # se_test_facts_all = np.array([[d1, pred, d2] for d1, d2 in ents_combinations
    #                               if (d1, pred, d2) not in predicate_all_facts_set])

    #logger.info(f'all test facts {len(se_test_facts_all)}')
    #logger.info(f'{len(list(set(se_test_facts_all[:, 2])))}')

    model.eval()
    se_test_facts_scores_final = None
    se_test_facts_all_final = None
    with torch.no_grad():
        for k, se_test_facts_all in se_test_facts_all_dict.items():
            if se_test_facts_all.shape[0] > 40000:
                batch_start = 0
                batch_size = 40000
                se_test_facts_scores = None
                while batch_start < se_test_facts_all.shape[0]:
                    batch_end = min(batch_start + batch_size, se_test_facts_all.shape[0])
                    se_test_facts_all_inp = torch.from_numpy(se_test_facts_all[batch_start:batch_end]).to(device)
                    se_test_facts_scores_tmp = model.score(se_test_facts_all_inp)
                    if type(se_test_facts_scores_tmp) is tuple:
                        se_test_facts_scores_tmp =  se_test_facts_scores_tmp[0].cpu().numpy()
                    else:
                        se_test_facts_scores_tmp =  se_test_facts_scores_tmp.cpu().numpy()
                    if se_test_facts_scores is None:
                        se_test_facts_scores = se_test_facts_scores_tmp
                    else:
                        se_test_facts_scores = np.concatenate([se_test_facts_scores, se_test_facts_scores_tmp], axis=0)
                    batch_start += batch_size
            else:
                se_test_facts_all = torch.from_numpy(se_test_facts_all).to(device)
                se_test_facts_scores = model.score(se_test_facts_all)
                if type(se_test_facts_scores) is tuple:
                    se_test_facts_scores = se_test_facts_scores[0].cpu().numpy()
                else:
                    se_test_facts_scores = se_test_facts_scores.cpu().numpy()
            se_test_facts_scores = softmax(se_test_facts_scores.squeeze())
            # logger.info(f'se test facts scores shape {se_test_facts_scores.shape}')
            if se_test_facts_scores_final is None:
                se_test_facts_scores_final = se_test_facts_scores
                se_test_facts_all_final = se_test_facts_all.cpu().numpy()
            else:
                se_test_facts_scores_final = np.concatenate([se_test_facts_scores_final, se_test_facts_scores])
                se_test_facts_all_final = np.concatenate([se_test_facts_all_final, se_test_facts_all.cpu().numpy()])


    logger.info(f'test facts scores shape {se_test_facts_scores_final.shape}')
    best_pred = np.argsort(se_test_facts_scores_final, axis=0)[-50:]
    vals = se_test_facts_all_final[best_pred.squeeze()]
    for d1, rel, p1 in vals:
        logger.info(f'{dataset.ent_mappings.inverse[d1]} increases protein {dataset.ent_mappings.inverse[p1]}')

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
