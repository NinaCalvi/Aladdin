#contain useful functions for different needs
import torch
import os
import numpy as np
import gzip
from libkge import KgDataset
from libkge import datasets
import logging
import sys


logger = logging.getLogger(os.path.basename(sys.argv[0]))

def load_pse_dataset(data_name):
    if data_name == "pse":
        print('curr directory' ,os.getcwd())
        #THIS SHOULD BE FIXED WITH CORRECT PATH
        kg_dp_path = os.path.join(os.getcwd(),'testing/data/pse/' )

        #getting entity mappings
        se_map_raw = [l.strip().split("\t") for l in open(os.path.join(kg_dp_path, "se_maps.txt")).readlines()]
        se_mapping = {k: v for k, v in se_map_raw}

        print("Importing dataset files ... ")
        benchmark_train_fd = gzip.open(os.path.join(kg_dp_path, "ploypharmacy_facts_train.txt.gz"), "rt")
        benchmark_valid_fd = gzip.open(os.path.join(kg_dp_path, "ploypharmacy_facts_valid.txt.gz"), "rt")
        benchmark_test_fd = gzip.open(os.path.join(kg_dp_path, "ploypharmacy_facts_test.txt.gz"), "rt")

        benchmark_train = np.array([l.strip().split() for l in benchmark_train_fd.readlines()])
        benchmark_valid = np.array([l.strip().split() for l in benchmark_valid_fd.readlines()])
        benchmark_test = np.array([l.strip().split() for l in benchmark_test_fd.readlines()])

        dataset = KgDataset(name=data_name)
        dataset.load_triples(benchmark_train, tag="train")
        dataset.load_triples(benchmark_valid, tag="valid")
        dataset.load_triples(benchmark_test, tag="test")


    else:
        kg_dp_path = os.path.join(os.getcwd(), 'testing/data/{}/'.format(data_name))
        dataset = datasets.load_dataset_from_dir(kg_dp_path)


    # pse_drugs = list(set(list(np.concatenate([benchmark_triples[:, 0], benchmark_triples[:, 2]]))))
    # pse_list = set(list(benchmark_triples[:, 1]))




    # del benchmark_train
    # del benchmark_valid
    # del benchmark_test
    # del benchmark_triples
    #
    # nb_entities = dataset.get_ents_count()
    # nb_relations = dataset.get_rels_count()
    # pse_indices = dataset.get_rel_indices(list(pse_list))
    #
    #
    # drug_combinations = np.array([[d1, d2] for d1, d2 in list(itertools.product(pse_drugs, pse_drugs)) if d1 != d2])
    #
    # d1 = np.array(dataset.get_ent_indices(list(drug_combinations[:, 0]))).reshape([-1, 1])
    # d2 = np.array(dataset.get_ent_indices(list(drug_combinations[:, 1]))).reshape([-1, 1])
    # drug_combinations = np.concatenate([d1, d2], axis=1)
    # del d1
    # del d2
    #
    #
    # # grouping side effect information by the side effect type
    # train_data = dataset.data["bench_train"]
    # valid_data = dataset.data["bench_valid"]
    # test_data = dataset.data["bench_test"]
    #
    # bench_idx_data = np.concatenate([train_data, valid_data, test_data])
    #
    # se_facts_full_dict = {se: set() for se in pse_indices}
    #
    # for s, p, o in bench_idx_data:
    #     se_facts_full_dict[p].add((s, p, o))

    return dataset


def generate_neg_instances(triples: torch.Tensor, nb_corrs: int, nb_ents: int, seed: int, *args, **kwargs):
    '''
    Generate random negatives for some positive triples
    This is needed when we are using pairwise/pointwise losses
    i.e. NOT needed for mcl models

    Parameters:
    -----------
    triples: positive triples with size [?, 3]
    nb_corss: number of corruptions to generate per triple
    nb_ents: total number of entities
    seed: random seed

    Returns:
    ---------
    torch.Tensor
        torch tensor for negative triples of size [?, 3]

    Note
    ---------
    The passed `nb_corrs` is evenly distributed between head and tail corruptions.
    i.e. create equal number of corruption on the subject and on the object

    Warning
    ---------
    This corruption heuristic might generate original true triples as corruptions.
    '''

    #create split
    # logger.info(f'Number of corruptions {nb_corrs}')
    # logger.info(type(nb_corrs))
    nb_corrs /= 2 #number of corruptions
    obj_corruptions = triples.repeat(int(np.ceil(nb_corrs)), 1)
    subj_corruptions = triples.repeat(int(np.floor(nb_corrs)), 1)

    #split tensors to isolate subject or object for corruption purposes
    sub_pred, obj = torch.split(obj_corruptions, [2, 1], dim=1)
    sub, obj_pred = torch.split(subj_corruptions, [1,2], dim=1)

    #corrupt
    obj_corr = torch.randint(0, nb_ents, obj.shape)
    sub_corr = torch.randint(0, nb_ents, sub.shape)

    return torch.cat((torch.cat((sub_pred, obj_corr), axis=1), torch.cat((sub_corr ,obj_pred), axis=1)), axis=0)



def df_to_dict(df, key_label, value_label):
    dict_ = dict()
    for i,r in df.iterrows():
        dict_[r[key_label]] = r[value_label]
    return dict_


def process_relation_entities(rel_type, ent_type):
    ent_type = ent_type.groupby('type')['Entity'].unique().reset_index()
    ent_type = df_to_dict(ent_type, 'type', 'Entity')
    neg_by_type = dict()
    for rel, t in rel_type.items():
        #relations are of form head_tail
        #head and tails could have + if multiple entities involved
        #or 'Any' if all types invovled
        head, tail = t.split('_')
        head = head.split('+')
        tail = tail.split('+')
        head_vals = []
        tail_vals = []
        
        for h in head:
            if h == 'Any':
                for t_ent, ents in ent_type.items():
                    head_vals += ents.tolist()
            else:
                head_vals += ent_type[h].tolist()
        
        
        for tl in tail:
            if tl == 'Any':
                for t_ent, ents in ent_type.items():
                    tail_vals += ents.tolist()
            else:
                tail_vals += ent_type[tl].tolist()
        if t not in neg_by_type:
            neg_by_type[t] = {'head': head_vals, 'tail': tail_vals}
        
       
    return neg_by_type
         
        
