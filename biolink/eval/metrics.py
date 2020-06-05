#contains possible different metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
from torch import nn

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
    order_rank  = np.argsort(np.argsort(-y_pred))
    rank  = order_rank[np.arange(len(y_pred)), true_idx] + 1
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


def auc_roc(y_pred: np.array, true_idx: np.array):
    '''
    Compute the area under the ROC curve

    y_pred: np.array 2-dim of predictions - num instances x num labels
    true_idx: np array of idx of true labels - 1d, (num_labels, )
    '''
    #first is to translate the ture label idx in matrix of num_instance x num_labels
    #this matrix will be binary, 1 at the label index and 0 everywhere else.

    logger.info('auc_roc')
    logger.info(f'auc_roc shape ypred \t{y_pred.shape}')
    # logger.info(f'predicted values \t{y_pred[0]}, \t{y_pred[10]}')
    # logger.in
    logger.info(f'auc_roc shape ytrue \t{true_idx.shape}')

    labels = np.zeros_like(y_pred)

    logger.info(f'labels shape \t{labels.shape}')
    labels[np.arange(len(labels)), true_idx] = 1

    logger.info(f'\t{labels[0]}')

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

#   NOTE: NEED TO MAKE SURE THAT TRAIN TRIPLES ARE INDEED NP ARRAY AND NOT A TENSRO?
def evaluate(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor,
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

    #store all the different metrics
    complete_metrcis = {}

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


    batch_start = 0
    mrr_val = 0.0
    counter = 0

    mrr_pasquale = 0.0

    logger.info(f'test triples type \t{type(test_triples)}')
    logger.info(f'test triples shape \t{test_triples.shape}')


    prediction_subject = None
    prediction_object = None

    prediction_subject_filtered = None
    prediction_object_filtered = None

    eps = 1-10

    softmax = nn.Softmax(dim=1)

    #get whole rhs
    whole_rhs = model.embeddings[0].weight.data.transpose(0,1)
    logger.info(f'whole_rh shape \t{whole_rhs.shape}')

    while batch_start < test_triples.shape[0]:
        counter += 2
        batch_end = min(batch_start + batch_size, test_triples.shape[0])
        batch_input = test_triples[batch_start:batch_end]
        with torch.no_grad():
            batch_tensor = batch_input.to(device)

            #score triples
            queries_sp = model.get_queries(batch_tensor).to(device)
            queries_po = model.get_queries(torch.index_select(batch_tensor, 1, torch.LongTensor([2,1,0]))).to(device)

            scores_sp = queries_sp @ rhs
            scores_po = queries_po @ rhs

            #CHANGED THE FOLLOWING LINE
            # scores_sp, scores_po, factors = model.forward(batch_tensor)


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

        #calculate the two mrr
        mrr_object = mrr(scores_sp, batch_input[:, 2])
        mrr_subject = mrr(scores_po, batch_input[:, 0])
        mrr_val += mrr_object
        mrr_val += mrr_subject

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

            rank_l = 1 + np.argsort(np.argsort(- scores_po[i, :]))[s_idx]
            rank_r = 1 + np.argsort(np.argsort(- scores_sp[i, :]))[o_idx]

            mrr_pasquale += 1.0 / rank_l
            mrr_pasquale += 1.0 / rank_r
        # logger.info(f'gone through batch input')

        if prediction_subject_filtered is not None:
            prediction_subject_filtered = np.vstack((prediction_subject_filtered, prob_scores_po))
            prediction_object_filtered = np.vstack((prediction_object_filtered, prob_scores_sp))
        else:
            prediction_subject_filtered = prob_scores_po
            prediction_object_filtered = prob_scores_sp




        batch_start += batch_size
        if (batch_start % 10000) == 0:
            logger.info(f'batch start \t{batch_start}')
    mrr_val /= counter
    mrr_pasquale /= (test_triples.shape[0]*2)

    metrics['MRR'] = mrr_val
    metrics['mrr_pasquale'] = mrr_pasquale
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
