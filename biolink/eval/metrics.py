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
    true_idx: np array of idx of true labels - 1d, (num_labels, )
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

    labels = np.zeros_like(y_pred)
    labels[np.arange(len(lables)), true_idx] = 1
    return roc_auc_score(labels, y_pred)

def auc_pr(y_pred: np.array, true_idx: np.array):
    '''
    Compute the area under the precision-recall curve. The outcome summarizes a precision-recall curve as the
    weighted mean of precisions achieved at each threshold

    y_pred: np.array 2-dim of predictions - num instances x num labels
    true_idx: np array of idx of true labels - 1d, (num_labels, )
    '''
    labels = np.zeros_like(y_pred)
    labels[np.arange(len(lables)), true_idx] = 1
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

    logger.info(f'test triples type \t{type(test_triples)}')
    logger.info(f'test triples shape \t{test_triples.shape}')


    prediction_subject = None
    prediction_object = None

    prediction_subject_filtered = None
    prediction_object_filtered = None

    while batch_start < test_triples.shape[0]:
        counter += 2
        batch_end = min(batch_start + batch_size, test_triples.shape[0])
        batch_input = test_triples[batch_start:batch_end]
        with torch.no_grad():
            batch_tensor = batch_input.to(device)
            scores_sp, scores_po, factors = model.forward(batch_tensor)
            #remove them from device
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

        # logger.info(f'in evaluate:')
        if prediction_subject is not None:
            prediction_subject = np.vstack((prediction_subject, scores_po))
            prediction_object = np.vstack((prediction_object, scores_sp))
        else:
            prediction_subject = scores_po
            prediction_object = scores_sp

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

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[i, tmp_s_idx] = - np.infty
        # logger.info(f'gone through batch input')

        if prediction_subject_filtered is not None:
            prediction_subject_filtered = np.vstack((prediction_subject_filtered, scores_po))
            prediction_object_filtered = np.vstack((prediction_object_filtered, scores_sp))
        else:
            prediction_subject_filtered = scores_po
            prediction_object_filtered = scores_sp


        #calculate the two mrr
        mrr_object = mrr(scores_sp, batch_input[:, 2])
        mrr_subject = mrr(scores_po, batch_input[:, 0])
        mrr_val += mrr_object
        mrr_val += mrr_subject

        batch_start += batch_size
        logger.info(f'batch start \t{batch_start}')
    mrr_val /= counter

    metrics['MRR'] = mrr_val

    auc_roc_raw_subj = auc_roc(prediction_subject, test_triples[:, 0])
    auc_roc_raw_obj = auc_roc(prediction_object, test_triples[:, 2])

    auc_roc_filt_subj = auc_roc(prediction_subject_filtered, test_triples[:, 0])
    auc_roc_filt_obj = auc_roc(prediction_object_filtered, test_triples[:, 2])

    metrics['AU-ROC_raw'] = (auc_roc_raw_obj + auc_roc_raw_subj)/2
    metrics['AU-ROC_fil'] = (auc_roc_filt_obj + auc_roc_filt_subj)/2

    return metrics
