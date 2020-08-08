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
        raise ValueError("Incorrect model instance given (%s)" %type(model))


def evaluate_non_mc(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor, batch_size: int, device: torch.device, validate: bool, mode: str = None):
    '''
    Evaluation method immediately returns the metrics wanted for non mc
    Parameters:
    ---------------
    all_triples: all triples in train/test/dev for calculaing filtered options
    '''
    metrics = {}
    sp_to_o = {}
    po_to_s = {}


    batch_size=2028

#     batch_size=1024

    if isinstance(model, TransE):
        batch_size = 90
    elif not isinstance(model, RotatE):
        batch_size = 1024
    print(batch_size)


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

        batch_end = min(batch_start + batch_size, test_triples.shape[0])
        counter_hits += 2*min(batch_size, batch_end - batch_start)
        batch_input = test_triples[batch_start:batch_end]

        #need to create negative instances
        with torch.no_grad():
            batch_tensor = batch_input.to(device)

            #CHANGED THE FOLLOWING LINE
            scores_sp, scores_po, factors = model.forward(batch_tensor)
            #slightly confused as to whether I should be attempting to score

            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()


        del batch_tensor
        torch.cuda.empty_cache()

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
                    # if not validate:
                    #     prob_scores_sp[i, tmp_o_idx] = 0
                    # # clipped_sp[i, tmp_o_idx] = - np.infty

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[i, tmp_s_idx] = - np.infty
                    # if not validate:
                    #     prob_scores_po[i, tmp_s_idx] = 0

        #calculate the two mrr
        if (mode is None) or (mode == 'head'):
            counter += 1
            counter_hits += min(batch_size, batch_end - batch_start)
            rank_subject = rank(scores_po, batch_input[:, 0])
            mrr_subject = np.mean(1/rank_subject)
            # mrr_subject = mrr(scores_po, batch_input[:, 0])
            hits_rate(rank_subject, hits, hits_at)
            mrr_val += mrr_subject
        if (mode is None) or (mode == 'tail'):
            counter += 1
            counter_hits += min(batch_size, batch_end - batch_start)
            rank_object = rank(scores_sp, batch_input[:, 2])
            mrr_object = np.mean(1/rank_object)

            hits_rate(rank_object, hits, hits_at)
            mrr_val += mrr_object


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
    logger.info('metrics done')

    return metrics



#   NOTE: NEED TO MAKE SURE THAT TRAIN TRIPLES ARE INDEED NP ARRAY AND NOT A TENSRO?
def evaluate_mc(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor,
            batch_size: int, device: torch.device, mode: str = None):
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

    # logger.info(f'all tiples \t{all_triples.size()}')


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
    #
    # logger.info(f'test triples type \t{type(test_triples)}')
    # logger.info(f'test triples shape \t{test_triples.shape}')


    prediction_subject = None
    prediction_object = None

    prediction_subject_filtered = None
    prediction_object_filtered = None

    eps = 1-10

    softmax = nn.Softmax(dim=1)

    #get whole rhs
    # whole_rhs = model.embeddings[0].weight.data.transpose(0,1)
    # logger.info(f'whole_rh shape \t{whole_rhs.shape}')
    model.eval()
    while batch_start < test_triples.shape[0]:
        counter += 2
        batch_end = min(batch_start + batch_size, test_triples.shape[0])
        counter_hits += 2*min(batch_size, batch_end - batch_start)
        batch_input = test_triples[batch_start:batch_end]
        with torch.no_grad():
            batch_tensor = batch_input.to(device)


            #CHANGED THE FOLLOWING LINE
            scores_sp, scores_po, factors = model.forward(batch_tensor)


            # # logger.info(f'socre_sp shape \t{scores_sp.shape}, score_po shape \t{scores_po.shape}')
            #
            # #remove them from device
            # #need to have probability scores for auc calculations
            # prob_scores_sp = softmax(scores_sp.cpu()).numpy()
            # prob_scores_po = softmax(scores_po.cpu()).numpy()

            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

        # # logger.info(f'in evaluate:')
        # if prediction_subject is not None:
        #     prediction_subject = np.vstack((prediction_subject, prob_scores_po))
        #     prediction_object = np.vstack((prediction_object, prob_scores_sp))
        # else:
        #     prediction_subject = prob_scores_po
        #     prediction_object = prob_scores_sp



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
                    # prob_scores_sp[i, tmp_o_idx] = 0
                    # clipped_sp[i, tmp_o_idx] = - np.infty

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[i, tmp_s_idx] = - np.infty
                    # prob_scores_po[i, tmp_s_idx] = 0

        #calculate the two mrr
        if (mode is None) or (mode == 'head'):
            counter += 1
            counter_hits += min(batch_size, batch_end - batch_start)
            rank_subject = rank(scores_po, batch_input[:, 0])
            mrr_subject = np.mean(1/rank_subject)
            # mrr_subject = mrr(scores_po, batch_input[:, 0])
            hits_rate(rank_subject, hits, hits_at)
            mrr_val += mrr_subject
        if (mode is None) or (mode == 'tail'):
            counter += 1
            counter_hits += min(batch_size, batch_end - batch_start)
            rank_object = rank(scores_sp, batch_input[:, 2])
            mrr_object = np.mean(1/rank_object)

            hits_rate(rank_object, hits, hits_at)
            mrr_val += mrr_object


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
    metrics['AU-ROC_fil'] =  -1
    logger.info('metrics done')

    return metrics


#   NOTE: NEED TO MAKE SURE THAT TRAIN TRIPLES ARE INDEED NP ARRAY AND NOT A TENSRO?
def evaluate_per_relation(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor,
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

    # logger.info(f'all tiples \t{all_triples.size()}')


    test_by_relation = dict()


    for training_instance in all_triples:
        s_idx, p_idx, o_idx = training_instance.numpy()
        if training_instance in test_triples:
            if p_idx in test_by_relation.keys():
                test_by_relation[p_idx] = torch.cat((test_by_relation[p_idx], training_instance), dim=0)
            else:
                test_by_relation[p_idx] = training_instance

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


    # batch_start = 0
    mrr_val = 0.0
    mrr_val_relations = dict()
    counter = 0
    counter_hits = 0
    #
    # logger.info(f'test triples type \t{type(test_triples)}')
    # logger.info(f'test triples shape \t{test_triples.shape}')


    prediction_subject = None
    prediction_object = None

    prediction_subject_filtered = None
    prediction_object_filtered = None


    #get whole rhs
    # whole_rhs = model.embeddings[0].weight.data.transpose(0,1)
    # logger.info(f'whole_rh shape \t{whole_rhs.shape}')
    model.eval()

    for rel, test_rels in test_by_relation.items():
        counter_rel = 2
        hits_relation = dict()
        counter_hits_relation = 2*test_rels.shape[0]
        # hits_at = [1, 3, 5, 10, 50, 100]

        for hits_at_value in hits_at:
            hits_relation[hits_at_value] = 0.0
        if test_rels.shape[0] > batch_size:
            batch_start = 0
            counter_rel = 0
            counter_hits_relation = 0
            while batch_start < test_rels.shape[0]:
                counter += 2
                counter_rel += 2
                batch_end = min(batch_start + batch_size, test_rels.shape[0])
                counter_hits += 2*min(batch_size, batch_end - batch_start)
                counter_hits_relation += 2*min(batch_size, batch_end - batch_start)
                batch_input = test_rels[batch_start:batch_end]
                with torch.no_grad():
                    batch_tensor = batch_input.to(device)
                    scores_sp, scores_po, factors = model.forward(batch_tensor)
                    scores_sp = scores_sp.cpu().numpy()
                    scores_po = scores_po.cpu().numpy()

                for i, el in enumerate(batch_input):
                    s_idx, p_idx, o_idx = el.numpy()
                    sp_key = (s_idx, p_idx)
                    po_key = (p_idx, o_idx)

                    o_to_remove = sp_to_o[sp_key]
                    s_to_remove = po_to_s[po_key]

                    for tmp_o_idx in o_to_remove:
                        if tmp_o_idx != o_idx:
                            scores_sp[i, tmp_o_idx] = - np.infty
                            # prob_scores_sp[i, tmp_o_idx] = 0
                            # clipped_sp[i, tmp_o_idx] = - np.infty

                    for tmp_s_idx in s_to_remove:
                        if tmp_s_idx != s_idx:
                            scores_po[i, tmp_s_idx] = - np.infty
                            # prob_scores_po[i, tmp_s_idx] = 0


                #calculate the two mrr
                rank_object = rank(scores_sp, batch_input[:, 2])
                mrr_object = np.mean(1/rank_object)
                # mrr_object = mrr(scores_sp, batch_input[:, 2])
                # rank_object = rank(scores_sp, batch_input[:, 2]) #redundancy in that i am doing this also inside mrr
                hits_rate(rank_object, hits, hits_at)
                hits_rate(rank_object, hits_relation, hits_at)

                rank_subject = rank(scores_po, batch_input[:, 0])
                mrr_subject = np.mean(1/rank_subject)
                # mrr_subject = mrr(scores_po, batch_input[:, 0])
                hits_rate(rank_subject, hits, hits_at)
                hits_rate(rank_subject, hits_relation, hits_at)

                if rel in mrr_val_relations.keys():
                    mrr_val_relations[rel] += (mrr_object + mrr_subject)
                else:
                    mrr_val_relations[rel] += (mrr_object + mrr_subject)
                mrr_val += mrr_object
                mrr_val += mrr_subject


                batch_start += batch_size
                if (batch_start % 10000) == 0:
                    logger.info(f'batch start \t{batch_start}')

            mrr_val_relations[rel] /= counter_rel
            logger.info(f'INFO RELATION \t{rel}')
            for n in hits_at:
                hits_relation[n] /= counter_hits_relation

            logger.info(f'MRR \t{mrr_val_relations[rel]}, H@1 \t{hits_relation[1]}, H@3 \t{hits_relation[3]}, H@10 \t{hits_relation[10]}')
        else:
            counter_hits += 2*test_rels.shape[0]
            conter += 2
            with torch.no_grad():
                batch_tensor = test_rels.to(device)
                scores_sp, scores_po, factors = model.forward(batch_tensor)
                scores_sp = scores_sp.cpu().numpy()
                scores_po = scores_po.cpu().numpy()

            for i, el in enumerate(test_rels):
                s_idx, p_idx, o_idx = el.numpy()
                sp_key = (s_idx, p_idx)
                po_key = (p_idx, o_idx)

                o_to_remove = sp_to_o[sp_key]
                s_to_remove = po_to_s[po_key]

                for tmp_o_idx in o_to_remove:
                    if tmp_o_idx != o_idx:
                        scores_sp[i, tmp_o_idx] = - np.infty
                        # prob_scores_sp[i, tmp_o_idx] = 0
                        # clipped_sp[i, tmp_o_idx] = - np.infty

                for tmp_s_idx in s_to_remove:
                    if tmp_s_idx != s_idx:
                        scores_po[i, tmp_s_idx] = - np.infty
                        # prob_scores_po[i, tmp_s_idx] = 0

            rank_object = rank(scores_sp, test_rels[:, 2])
            mrr_object = np.mean(1/rank_object)
            # mrr_object = mrr(scores_sp, batch_input[:, 2])
            # rank_object = rank(scores_sp, batch_input[:, 2]) #redundancy in that i am doing this also inside mrr
            hits_rate(rank_object, hits, hits_at)
            hits_rate(rank_object, hits_relation, hits_at)

            rank_subject = rank(scores_po, test_rels[:, 0])
            mrr_subject = np.mean(1/rank_subject)
            # mrr_subject = mrr(scores_po, batch_input[:, 0])
            hits_rate(rank_subject, hits, hits_at)
            hits_rate(rank_subject, hits_relation, hits_at)

            if rel in mrr_val_relations.keys():
                mrr_val_relations[rel] += (mrr_object + mrr_subject)
            else:
                mrr_val_relations[rel] += (mrr_object + mrr_subject)
            mrr_val += mrr_object
            mrr_val += mrr_subject

        mrr_val_relations[rel] /= counter_rel
        logger.info(f'INFO RELATION \t{rel}')
        for n in hits_at:
            hits_relation[n] /= counter_hits_relation

        logger.info(f'MRR \t{mrr_val_relations[rel]}, H@1 \t{hits_relation[1]}, H@3 \t{hits_relation[3]}, H@10 \t{hits_relation[10]}')


    mrr_val /= counter
    for n in hits_at:
        hits[n] /= counter_hits
    metrics['MRR'] = mrr_val
    metrics['H@1'] = hits[1]
    metrics['H@3'] = hits[3]
    metrics['H@10'] = hits[10]

    logger.info('done')

    metrics['AU-ROC_raw'] = -1
    metrics['AU-ROC_fil'] =  -1
    logger.info('metrics done')

    return metrics


def precision_at_k(y_true: np.array, y_pred: np.array, k: int, pos_label=1.0):
    """ Compute the mean precision at k of a rank of predicted scores
        Parameters
        ----------
        y_true: np.ndarray
            true labels
        y_pred: np.ndarray
            predicted scores
        k: int
            the position `k`
        pos_label: float
            label of the positive true instances
        Returns
        -------
        float
            the mean reciprocal rank of the true labels in the rank
    """
    if k < 1 or k > len(y_true):
        raise ValueError('Invalid k value: %s' % k)

    rank_order = np.argsort(y_pred)[::-1]
    y_true_k_sorted = y_true[rank_order[:k]]
    return np.count_nonzero(y_true_k_sorted == pos_label) / k


def ranks_for_precision(y_true, y_pred, pos_label=1.0):
    """ Compute ranks of the true labels in a rank
    Parameters
    ----------
    y_true: np.ndarray
        true labels
    y_pred: np.ndarray
        predicted scores
    pos_label: float
        label of the positive true instances
    Returns
    -------
    np.ndarray
        ranks of the true labels in the rank
    """
    rank_order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[rank_order]
    pos_label_mask = y_true_sorted == pos_label
    return np.nonzero(pos_label_mask)[0] + 1

def average_precision(y_true: np.array, y_pred:np.array, pos_label=1.0):
    """ Compute the average precision of a rank of predicted scores
        Parameters
        ----------
        y_true: np.ndarray
            true labels
        y_pred: np.ndarray
            predicted scores
        pos_label: float
            label of the positive true instances
        Returns
        -------
        float
            the mean reciprocal rank of the true labels in the rank
    """
    ranks_array = ranks_for_precision(y_true, y_pred, pos_label=1.0)

    pk_list = []
    for k in ranks_array:
        pk_list.append(precision_at_k(y_true, y_pred, k, pos_label=pos_label))
    return np.mean(pk_list)



#   NOTE: NEED TO MAKE SURE THAT TRAIN TRIPLES ARE INDEED NP ARRAY AND NOT A TENSRO?
def evaluate_auc(model: nn.Module, test_triples: torch.Tensor, all_triples: torch.Tensor,
            batch_size: int, device: torch.device, harder: bool = False):
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
    if harder:
        metrics_five = {}
        metrics_ten = {}
    else:
        metrics = {}

    predicate_indeces = list(set(all_triples[:, 1].numpy()))
    logger.info(f'length predicate_instanes \t{len(predicate_indeces)}')
    # print(predicate_indeces)
    # print(type(predicate_indeces[0]))
    se_facts_full_dict = {se: set() for se in predicate_indeces}

    logger.info('predicate instances done')

    test_triples_pred = {}

    for instance in all_triples:
        s, p, o = instance.numpy()
        if instance in test_triples:
            if p in test_triples_pred:
                test_triples_pred[p] = np.vstack((test_triples_pred[p], instance.numpy()))
            else:
                test_triples_pred[p] = instance.numpy()
        se_facts_full_dict[p].add((s, p, o))


    logger.info(f'all tiples \t{all_triples.size()}')
    test_triples = test_triples.numpy()
    logger.info('make test triples numpy')

    if harder:
        predicate_ap_list_five = []
        predicate_auc_roc_list_five = []
        predicate_auc_pr_list_five = []
        predicate_p50_list_five = []

        predicate_ap_list_ten = []
        predicate_auc_roc_list_ten = []
        predicate_auc_pr_list_ten = []
        predicate_p50_list_ten = []
    else:
        predicate_ap_list = []
        predicate_auc_roc_list = []
        predicate_auc_pr_list = []
        predicate_p50_list = []

    entities = list(set(list(np.concatenate([all_triples[:, 0], all_triples[:, 2]]))))
    ents_combinations =  np.array([[d1, d2] for d1, d2 in list(itertools.product(entities, entities)) if d1 != d2])

    # softmax = nn.Softmax(dim=1)

    metrics_per_se = dict()

    logger.info('about to start predicting')

    for pred in predicate_indeces:
        predicate_all_facts_set = se_facts_full_dict[pred]
        # predicate_test_facts_pos = np.array([[s, p, o] for s, p, o in test_triples if p == pred])
        if pred in test_triples_pred:
            predicate_test_facts_pos = test_triples_pred[pred]
        else:
            logger.info(f'\t{pred} not in test_triples_pred')
        predicate_test_facts_pos_size = len(predicate_test_facts_pos)

        #get negative samples
        print('length pred pos', predicate_test_facts_pos_size)
        se_test_facts_neg = np.array([[d1, pred, d2] for d1, d2 in ents_combinations
                                      if (d1, pred, d2) not in predicate_all_facts_set
                                      and (d2, pred, d1) not in predicate_all_facts_set])

        #ensure it's 1:1 positive to negative
        np.random.shuffle(se_test_facts_neg)
        if harder:
            se_test_facts_neg_five = se_test_facts_neg[:(predicate_test_facts_pos_size*5), :]
            se_test_facts_neg_ten = se_test_facts_neg[:(predicate_test_facts_pos_size*10), :]
            se_test_facts_all_five = np.concatenate([predicate_test_facts_pos, se_test_facts_neg_five])
            se_test_facts_labels_five = np.concatenate([np.ones([len(predicate_test_facts_pos)]), np.zeros([len(se_test_facts_neg_five)])])
            se_test_facts_all_ten = np.concatenate([predicate_test_facts_pos, se_test_facts_neg_ten])
            se_test_facts_labels_ten = np.concatenate([np.ones([len(predicate_test_facts_pos)]), np.zeros([len(se_test_facts_neg_ten)])])
        else:
            se_test_facts_neg = se_test_facts_neg[:predicate_test_facts_pos_size, :]
            se_test_facts_all = np.concatenate([predicate_test_facts_pos, se_test_facts_neg])
            se_test_facts_labels = np.concatenate([np.ones([len(predicate_test_facts_pos)]), np.zeros([len(se_test_facts_neg)])])

        with torch.no_grad():
            if harder:
                se_test_facts_all_five = torch.from_numpy(se_test_facts_all_five).to(device)
                se_test_facts_scores_five = model.score(se_test_facts_all_five)
                se_test_facts_scores_five = se_test_facts_scores_five.cpu().numpy()

                se_test_facts_all_ten = torch.from_numpy(se_test_facts_all_ten).to(device)
                se_test_facts_scores_ten = model.score(se_test_facts_all_ten)
                se_test_facts_scores_ten = se_test_facts_scores_ten.cpu().numpy()
            else:
                se_test_facts_all = torch.from_numpy(se_test_facts_all).to(device)
                se_test_facts_scores = model.score(se_test_facts_all)
                se_test_facts_scores = se_test_facts_scores.cpu().numpy()


        # se_auc_pr = auc_pr(se_test_facts_labels, se_test_facts_scores)
        if harder:
            se_ap_five = average_precision(se_test_facts_labels_five, se_test_facts_scores_five)
            se_p50_five = precision_at_k(se_test_facts_labels_five, se_test_facts_scores_five, k=50)
            se_auc_pr_five = average_precision_score(se_test_facts_labels_five, se_test_facts_scores_five)
            se_auc_roc_five = roc_auc_score(se_test_facts_labels_five, se_test_facts_scores_five)

            predicate_auc_roc_list_five.append(se_auc_roc_five)
            predicate_ap_list_five.append(se_ap_five)
            predicate_auc_pr_list_five.append(se_auc_pr_five)
            predicate_p50_list_five.append(se_p50_five)

            se_ap_ten = average_precision(se_test_facts_labels_ten, se_test_facts_scores_ten)
            se_p50_ten = precision_at_k(se_test_facts_labels_ten, se_test_facts_scores_ten, k=50)
            se_auc_pr_ten = average_precision_score(se_test_facts_labels_ten, se_test_facts_scores_ten)
            se_auc_roc_ten = roc_auc_score(se_test_facts_labels_ten, se_test_facts_scores_ten)


            predicate_auc_roc_list_ten.append(se_auc_roc_ten)
            predicate_ap_list_ten.append(se_ap_ten)
            predicate_auc_pr_list_ten.append(se_auc_pr_ten)
            predicate_p50_list_ten.append(se_p50_ten)

            # metrics_per_se[pred] = {"ap": se_ap, "auc-roc": se_auc_roc, "auc-pr": se_auc_pr, "p@50": se_p50}
            print('AUC FIVE NEGS')
            print("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - P@50: %1.4f > %s" %
                  (se_ap_five, se_auc_roc_five, se_auc_pr_five, se_p50_five, pred), flush=True)

            print('AUC TEN NEGS')
            print("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - P@50: %1.4f > %s" %
                  (se_ap_ten, se_auc_roc_ten, se_auc_pr_ten, se_p50_ten, pred), flush=True)
        else:

            se_ap = average_precision(se_test_facts_labels, se_test_facts_scores)
            se_p50 = precision_at_k(se_test_facts_labels, se_test_facts_scores, k=50)
            se_auc_pr = average_precision_score(se_test_facts_labels, se_test_facts_scores)
            se_auc_roc = roc_auc_score(se_test_facts_labels, se_test_facts_scores)



            metrics_per_se[pred] = {"ap": se_ap, "auc-roc": se_auc_roc, "auc-pr": se_auc_pr, "p@50": se_p50}
            print("AP: %1.4f - AUC-ROC: %1.4f - AUC-PR: %1.4f - P@50: %1.4f > %s" %
                  (se_ap, se_auc_roc, se_auc_pr, se_p50, pred), flush=True)

            predicate_auc_roc_list.append(se_auc_roc)
            predicate_ap_list.append(se_ap)
            predicate_auc_pr_list.append(se_auc_pr)
            predicate_p50_list.append(se_p50)

    logger.info('done')
    if harder:

        metrics_five['AUC-ROC'] = np.mean(predicate_auc_roc_list_five)
        metrics_five['AP'] = np.mean(se_ap_five)
        metrics_five['P@50'] = np.mean(se_p50_five)
        metrics_five['AUC_PR'] = np.mean(se_auc_pr_five)

        metrics_ten['AUC-ROC'] = np.mean(predicate_auc_roc_list_ten)
        metrics_ten['AP'] = np.mean(se_ap_ten)
        metrics_ten['P@50'] = np.mean(se_p50_ten)
        metrics_ten['AUC_PR'] = np.mean(se_auc_pr_ten)


        logger.info('metrics done')
        logger.info(f'AUC_ROC five \t{metrics_five["AUC-ROC"]}')
        logger.info(f'AUC_ROC ten \t{metrics_ten["AUC-ROC"]}')

        return metrics_five, metrics_ten

    else:
        metrics['AUC-ROC'] = np.mean(predicate_auc_roc_list)
        metrics['AP'] = np.mean(se_ap)
        metrics['P@50'] = np.mean(se_p50)
        metrics['AUC_PR'] = np.mean(se_auc_pr)



        logger.info('metrics done')
        logger.info(f'AUC_ROC \t{metrics["AUC-ROC"]}')

        return metrics
