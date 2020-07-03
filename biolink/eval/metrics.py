#contains possible different metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
from torch import nn

from biolink.embeddings import KBCModel, KBCModelMCL, TransE
import logging
import os
import sys


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

def rank(y_pred: np.array, true_idx: np.array):

    '''
    y_pred: np.array 2-dim of predictions - num instances x num labels
    true_idx: np array of idx of true labels - 1d, (num_labels, )
    '''

    #sorts from smallest to biggest
    #we want biggest first (i.e. highest score)

    #applying argsort again will undo the sort done before
    #multiplying by -1 because we want the most likely thing to be sorted earlier
    #and since argsort puts -ve value before the rest, this makes sense


    #an assign to each element in the list its rank within the sorted stuff
    scored_true = y_pred[np.arange(len(y_pred)), true_idx]
    rank = 1 + np.sum(y_pred >= scored_true.reshape(-1,1), axis=1)
    # order_rank  = np.argsort(np.argsort(-y_pred))
    # rank  = order_rank[np.arange(len(y_pred)), true_idx] + 1
    return rank

def mean_rank(y_pred: np.array, true_idx: np.array):
    '''
    Compute the mean ranks

    y_pred: np.array 2-dim of predictions - num instances x num labels
    true_idx: np array of idx of true labels - 1d, (num_labels, )
    '''
    ranks = rank(y_pred, true_idx)
    return np.mean(ranks)

def mrr(y_pred: np.array, true_idx: np.array):
    '''
    Compute the mean reciprocal rank

    y_pred: np.array 2-dim of predictions - num instances x num labels
    true_idx: np array of idx of true labels - 1d, (num_instances, )
    '''
    reciprocal = 1/rank(y_pred, true_idx)
    return np.mean(reciprocal)


def hits_rate(ranked_values: np.array, hits: dict, hist_at: list):
    '''
    ranked_values = ranked predictions
    hits = dictionary which counts the hits
    hits_at = list of which hits to consider
    '''

    for n in hist_at:
        tot_ranks = np.sum(ranked_values <= n)
        hits[n] += tot_ranks

def auc_roc(y_pred: np.array, true_idx: np.array):
    '''
    Compute the area under the ROC curve

    y_pred: np.array 2-dim of predictions - num instances x num labels
    true_idx: np array of idx of true labels - 1d, (num_labels, )
    '''
    #first is to translate the ture label idx in matrix of num_instance x num_labels
    #this matrix will be binary, 1 at the label index and 0 everywhere else.

    logger.info('auc_roc')
    print(y_pred)
    # logger.info(f'auc_roc shape ypred \t{y_pred.shape}')
    # logger.info(f'predicted values \t{y_pred[0]}, \t{y_pred[10]}')
    # logger.in
    # logger.info(f'auc_roc shape ytrue \t{true_idx.shape}')

    labels = np.zeros_like(y_pred)

    # logger.info(f'labels shape \t{labels.shape}')
    labels[np.arange(len(labels)), true_idx] = 1

    # logger.info(f'\t{labels[0]}')

    return roc_auc_score(labels, y_pred, average='micro')

def auc_pr(y_pred: np.array, true_idx: np.array):
    '''
    Compute the area under the precision-recall curve. The outcome summarizes a precision-recall curve as the
    weighted mean of precisions achieved at each threshold

    y_pred: np.array 2-dim of predictions - num instances x num labels
    true_idx: np array of idx of true labels - 1d, (num_labels, )
    '''

    labels = np.zeros_like(y_pred)
    labels[np.arange(len(labels)), true_idx] = 1
    return average_precision_score(labels, y_pred)



def evaluate(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor, batch_size: int, device: torch.device, validate: bool = False):
    if isinstance(model, KBCModelMCL):
        return evaluate_mc(model, test_triples, all_triples, batch_size, device)
    elif isinstance(model, KBCModel):
        return evaluate_non_mc(model, test_triples, all_triples, batch_size, device, validate)
    else:
        raise ValueError("Incorrect model instance given (%s)" %type(model))


def evaluate_non_mc(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor, batch_size: int, device: torch.device, validate: bool):
    '''
    Evaluation method immediately returns the metrics wanted for non mc
    Parameters:
    ---------------
    all_triples: all triples in train/test/dev for calculaing filtered options
    '''
    metrics = {}
    sp_to_o = {}
    po_to_s = {}

    if isinstance(model, TransE):
        batch_size = 90

    for training_instance in all_triples:
        s_idx, p_idx, o_idx = training_instance.numpy()
        sp_key = (s_idx, p_idx)
        po_key = (p_idx, o_idx)

        if sp_key not in sp_to_o:
            sp_to_o[sp_key] = [o_idx]
        else:
            sp_to_o[sp_key].append(o_idx)

        if po_key not in po_to_s:
            po_to_s[po_key] = [s_idx]
        else:
            po_to_s[po_key].append(s_idx)


    hits = dict()
    hits_at = [1, 3, 10]
    # hits_at = [1, 3, 5, 10, 50, 100]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0


    batch_start = 0
    mrr_val = 0.0
    counter = 0
    counter_hits = 0

    prediction_object = None
    prediction_object_filtered = None

    prediction_subject = None
    prediction_subject_filtered = None


    eps = 1-10
    softmax = nn.Softmax(dim=1)


    while batch_start < test_triples.shape[0]:
        counter += 2
        batch_end = min(batch_start + batch_size, test_triples.shape[0])
        counter_hits += 2*min(batch_size, batch_end - batch_start)
        batch_input = test_triples[batch_start:batch_end]

        #need to create negative instances
        with torch.no_grad():
            batch_tensor = batch_input.to(device)

            #CHANGED THE FOLLOWING LINE
            scores_sp, scores_po, factors = model.forward(batch_tensor)
            #slightly confused as to whether I should be attempting to score


            # logger.info(f'socre_sp shape \t{scores_sp.shape}, score_po shape \t{scores_po.shape}')

            #remove them from device
            #need to have probability scores for auc calculations
            if not validate:
                prob_scores_sp = softmax(scores_sp.cpu()).numpy()
                prob_scores_po = softmax(scores_po.cpu()).numpy()

            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

        # logger.info(f'in evaluate:')
        if not validate:
            if prediction_subject is not None:
                prediction_subject = np.vstack((prediction_subject, prob_scores_po))
                prediction_object = np.vstack((prediction_object, prob_scores_sp))
            else:
                prediction_subject = prob_scores_po
                prediction_object = prob_scores_sp



        #remove scores given to filtered labels
        for i, el in enumerate(batch_input):
            s_idx, p_idx, o_idx = el.numpy()
            sp_key = (s_idx, p_idx)
            po_key = (p_idx, o_idx)

            o_to_remove = sp_to_o[sp_key]
            s_to_remove = po_to_s[po_key]

            for tmp_o_idx in o_to_remove:
                if tmp_o_idx != o_idx:
                    scores_sp[i, tmp_o_idx] = - np.infty
                    if not validate:
                        prob_scores_sp[i, tmp_o_idx] = 0
                    # clipped_sp[i, tmp_o_idx] = - np.infty

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[i, tmp_s_idx] = - np.infty
                    if not validate:
                        prob_scores_po[i, tmp_s_idx] = 0

         # logger.info(f'gone through batch input')

        if not validate:
            if prediction_subject_filtered is not None:
                prediction_subject_filtered = np.vstack((prediction_subject_filtered, prob_scores_po))
                prediction_object_filtered = np.vstack((prediction_object_filtered, prob_scores_sp))
            else:
                prediction_subject_filtered = prob_scores_po
                prediction_object_filtered = prob_scores_sp

        #calculate the two mrr
        rank_object = rank(scores_sp, batch_input[:, 2])
        mrr_object = np.mean(1/rank_object)
        # mrr_object = mrr(scores_sp, batch_input[:, 2])
        # rank_object = rank(scores_sp, batch_input[:, 2]) #redundancy in that i am doing this also inside mrr
        hits_rate(rank_object, hits, hits_at)

        rank_subject = rank(scores_po, batch_input[:, 0])
        mrr_subject = np.mean(1/rank_subject)
        # mrr_subject = mrr(scores_po, batch_input[:, 0])
        hits_rate(rank_subject, hits, hits_at)

        mrr_val += mrr_object
        mrr_val += mrr_subject


        batch_start += batch_size
        if (batch_start % 10000) == 0:
            logger.info(f'batch start \t{batch_start}')



    mrr_val /= counter
    for n in hits_at:
        hits[n] /= counter_hits
    metrics['MRR'] = mrr_val
    metrics['H@1'] = hits[1]
    metrics['H@3'] = hits[3]
    metrics['H@10'] = hits[10]

    logger.info('done')

    metrics['AU-ROC_raw'] = -1
    metrics['AU-ROC_fil'] = -1

    # if not validate:
    #     auc_roc_raw_subj = auc_roc(prediction_subject, test_triples[:, 0])
    #     auc_roc_raw_obj = auc_roc(prediction_object, test_triples[:, 2])
    #
    #     logger.info('done not filtered aucroc')
    #
    #     auc_roc_filt_subj = auc_roc(prediction_subject_filtered, test_triples[:, 0])
    #     auc_roc_filt_obj = auc_roc(prediction_object_filtered, test_triples[:, 2])
    #
    #     metrics['AU-ROC_raw'] = (auc_roc_raw_obj + auc_roc_raw_subj)/2
    #     metrics['AU-ROC_fil'] = (auc_roc_filt_obj + auc_roc_filt_subj)/2
    logger.info('metrics done')

    return metrics



#   NOTE: NEED TO MAKE SURE THAT TRAIN TRIPLES ARE INDEED NP ARRAY AND NOT A TENSRO?
def evaluate_mc(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor,
            batch_size: int, device: torch.device):
    '''
    Evaluation method immediately returns the metrics wanted
    Parameters:
    ---------------
    all_triples: all triples in train/test/dev for calculaing filtered options
    '''

    #store the labels for subject_predicate and predicte_object that were seen in training
    #some sp and po situations may have more than one label assigned to it (i.e. may link to different entitites)
    #when calculating the score we would like to filter them out

    sp_to_o = {}
    po_to_s = {}
    metrics = {}

    logger.info(f'all tiples \t{all_triples.size()}')


    for training_instance in all_triples:
        s_idx, p_idx, o_idx = training_instance.numpy()
        sp_key = (s_idx, p_idx)
        # print('sp_key', sp_key)
        po_key = (p_idx, o_idx)

        if sp_key not in sp_to_o:
            sp_to_o[sp_key] = [o_idx]
        else:
            sp_to_o[sp_key].append(o_idx)

        if po_key not in po_to_s:
            po_to_s[po_key] = [s_idx]
        else:
            po_to_s[po_key].append(s_idx)


    hits = dict()
    hits_at = [1, 3, 10]
    # hits_at = [1, 3, 5, 10, 50, 100]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0


    batch_start = 0
    mrr_val = 0.0
    counter = 0
    counter_hits = 0

    logger.info(f'test triples type \t{type(test_triples)}')
    logger.info(f'test triples shape \t{test_triples.shape}')


    prediction_subject = None
    prediction_object = None

    prediction_subject_filtered = None
    prediction_object_filtered = None

    eps = 1-10

    softmax = nn.Softmax(dim=1)

    #get whole rhs
    # whole_rhs = model.embeddings[0].weight.data.transpose(0,1)
    # logger.info(f'whole_rh shape \t{whole_rhs.shape}')

    while batch_start < test_triples.shape[0]:
        counter += 2
        batch_end = min(batch_start + batch_size, test_triples.shape[0])
        counter_hits += 2*min(batch_size, batch_end - batch_start)
        batch_input = test_triples[batch_start:batch_end]
        with torch.no_grad():
            batch_tensor = batch_input.to(device)

            #score triples
            # queries_sp = model.get_queries(batch_tensor)
            # queries_po = model.get_queries(torch.index_select(batch_tensor, 1, torch.LongTensor([2,1,0]).to(device)))
            #
            # scores_sp = queries_sp @ whole_rhs
            # scores_po = queries_po @ whole_rhs

            #CHANGED THE FOLLOWING LINE
            scores_sp, scores_po, factors = model.forward(batch_tensor)


            logger.info(f'socre_sp shape \t{scores_sp.shape}, score_po shape \t{scores_po.shape}')

            #remove them from device
            #need to have probability scores for auc calculations
            prob_scores_sp = softmax(scores_sp.cpu()).numpy()
            prob_scores_po = softmax(scores_po.cpu()).numpy()

            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

        # logger.info(f'in evaluate:')
        if prediction_subject is not None:
            prediction_subject = np.vstack((prediction_subject, prob_scores_po))
            prediction_object = np.vstack((prediction_object, prob_scores_sp))
        else:
            prediction_subject = prob_scores_po
            prediction_object = prob_scores_sp



        #remove scores given to filtered labels
        for i, el in enumerate(batch_input):
            s_idx, p_idx, o_idx = el.numpy()
            sp_key = (s_idx, p_idx)
            po_key = (p_idx, o_idx)

            o_to_remove = sp_to_o[sp_key]
            s_to_remove = po_to_s[po_key]

            for tmp_o_idx in o_to_remove:
                if tmp_o_idx != o_idx:
                    scores_sp[i, tmp_o_idx] = - np.infty
                    prob_scores_sp[i, tmp_o_idx] = 0
                    # clipped_sp[i, tmp_o_idx] = - np.infty

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[i, tmp_s_idx] = - np.infty
                    prob_scores_po[i, tmp_s_idx] = 0

         # logger.info(f'gone through batch input')

        if prediction_subject_filtered is not None:
            prediction_subject_filtered = np.vstack((prediction_subject_filtered, prob_scores_po))
            prediction_object_filtered = np.vstack((prediction_object_filtered, prob_scores_sp))
        else:
            prediction_subject_filtered = prob_scores_po
            prediction_object_filtered = prob_scores_sp

        #calculate the two mrr
        rank_object = rank(scores_sp, batch_input[:, 2])
        mrr_object = np.mean(1/rank_object)
        # mrr_object = mrr(scores_sp, batch_input[:, 2])
        # rank_object = rank(scores_sp, batch_input[:, 2]) #redundancy in that i am doing this also inside mrr
        hits_rate(rank_object, hits, hits_at)

        rank_subject = rank(scores_po, batch_input[:, 0])
        mrr_subject = np.mean(1/rank_subject)
        # mrr_subject = mrr(scores_po, batch_input[:, 0])
        hits_rate(rank_subject, hits, hits_at)

        mrr_val += mrr_object
        mrr_val += mrr_subject


        batch_start += batch_size
        if (batch_start % 10000) == 0:
            logger.info(f'batch start \t{batch_start}')



    mrr_val /= counter
    for n in hits_at:
        hits[n] /= counter_hits
    metrics['MRR'] = mrr_val
    metrics['H@1'] = hits[1]
    metrics['H@3'] = hits[3]
    metrics['H@10'] = hits[10]

    logger.info('done')

    auc_roc_raw_subj = auc_roc(prediction_subject, test_triples[:, 0])
    auc_roc_raw_obj = auc_roc(prediction_object, test_triples[:, 2])

    logger.info('done not filtered aucroc')

    auc_roc_filt_subj = auc_roc(prediction_subject_filtered, test_triples[:, 0])
    auc_roc_filt_obj = auc_roc(prediction_object_filtered, test_triples[:, 2])

    metrics['AU-ROC_raw'] = (auc_roc_raw_obj + auc_roc_raw_subj)/2
    metrics['AU-ROC_fil'] = (auc_roc_filt_obj + auc_roc_filt_subj)/2
    logger.info('metrics done')

    return metrics




#   NOTE: NEED TO MAKE SURE THAT TRAIN TRIPLES ARE INDEED NP ARRAY AND NOT A TENSRO?
def evaluate_auc(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor,
            batch_size: int, device: torch.device):
    '''
    Evaluation method immediately returns the metrics wanted
    Parameters:
    ---------------
    all_triples: all triples in train/test/dev for calculaing filtered options
    '''

    #store the labels for subject_predicate and predicte_object that were seen in training
    #some sp and po situations may have more than one label assigned to it (i.e. may link to different entitites)
    #when calculating the score we would like to filter them out

    sp_to_o = {}
    po_to_s = {}
    metrics = {}

    predicate_indeces = set(all_triples[:, 1])
    se_facts_full_dict = {se: set() for se in predicate_indices}

    for s, p, o in all_triples:
        se_facts_full_dict[p].add((s, p, o))


    logger.info(f'all tiples \t{all_triples.size()}')


    batch_start = 0

    predicate_ap_list = []
    predicate_auc_roc_list = []
    predicate_auc_pr_list = []
    predicate_p50_list = []

    entities = list(set(list(np.concatenate([all_triples[:, 0], all_triples[:, 2]]))))
    ents_combinations =  np.array([[d1, d2] for d1, d2 in list(itertools.product(entities, entities)) if d1 != d2])

    eps = 1-10

    softmax = nn.Softmax(dim=1)


    for pred in predicate_indeces:
        predicate_all_facts_set = se_facts_full_dict[se]
        predicate_test_facts_pos = np.array([[s, p, o] for s, p, o in test_triples if p == se])
        predicate_test_facts_pos_size = len(predicate_test_facts_pos)

        #get negative samples
        se_test_facts_neg = np.array([[d1, se, d2] for d1, d2 in ents_combinations
                                      if (d1, se, d2) not in predicate_all_facts_set
                                      and (d2, se, d1) not in predicate_all_facts_set])

        #ensure it's 1:1 positive to negative
        np.random.shuffle(se_test_facts_neg)
        se_test_facts_neg = se_test_facts_neg[:se_test_facts_pos_size, :]

        se_test_facts_all = np.concatenate([se_test_facts_pos, se_test_facts_neg])
        se_test_facts_labels = np.concatenate([np.ones([len(se_test_facts_pos)]), np.zeros([len(se_test_facts_neg)])])
        with torch.no_grad():
            se_test_facts_all = se_test_facts_all.to(device)
            se_test_facts_scores = model.score(set_test_facts_all)
            se_test_facts_scores = softmax(se_test_facts_scores.cpu()).numpy()


        # se_auc_pr = auc_pr(se_test_facts_labels, se_test_facts_scores)
        se_auc_roc = roc_auc_score(se_test_facts_labels, se_test_facts_scores)
        predicate_auc_roc_list.append(se_auc_roc)

    logger.info('done')

    metrics['AU-ROC'] = np.mean(predicate_auc_roc_list)
    logger.info('metrics done')
    logger.info(metrics['AUC-ROC'])

    return metrics
