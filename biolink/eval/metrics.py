#contains possible different metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
from torch import nn
import itertools

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



    #scored_true = y_pred[np.arange(len(y_pred)), true_idx]
    #rank = 1 + np.sum(y_pred > scored_true.reshape(-1,1), axis=1)
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



def evaluate(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor, batch_size: int, device: torch.device, validate: bool = False, auc: bool = False, harder: bool = False):
    if auc:
        return evaluate_auc(model, test_triples, all_triples, batch_size, device, harder)

    if isinstance(model, KBCModelMCL):
        return evaluate_mc(model, test_triples, all_triples, batch_size, device)
    elif isinstance(model, KBCModel):
        return evaluate_non_mc(model, test_triples, all_triples, batch_size, device, validate)
    else:
        raise ValueError("Incorrect model instance given