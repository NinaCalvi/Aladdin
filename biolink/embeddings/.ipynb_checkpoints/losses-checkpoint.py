import torch
from torch import nn
from typing import Tuple, List, Dict

def reduce_loss(loss: torch.Tensor, reduction_type: str):
    # print('loss shape', loss.size)
    if reduction_type == 'sum':
        return torch.sum(loss, dim=0)
    if reduction_type == 'avg':
        return torch.mean(loss, dim=0)


def compute_kge_loss(predictions: torch.Tensor, loss: str, device: torch.device, pos_size: int, reduction_type: str = "avg", margin_value: float =1.0):
    '''
    predictions: (N,) scores vector of a triple
    loss: type of loss function
    reduction_type: type of reduction
    '''
    #looking at sameh's code for this situation
    #they binarize the positive results (1) and the negative results (0)
    #and use that as the targets
    #essentially the predictions are equally split between positive and negative predictions
    #through the middle

    #assuming that the positive and negative samples are perfectly balanced
    # print(predictions.shape)
    scrs = torch.split(predictions, [pos_size, int(predictions.shape[0]-pos_size)], dim=0)

#     print(scrs[0].shape)

    pw_targets = torch.cat((torch.ones(scrs[0].shape), -1*torch.ones(scrs[1].shape)), dim=0)

    pos_scores = scrs[0].repeat(int(predictions.shape[0]/pos_size)-1, 1)
    neg_scores = scrs[1]
#     assert pos_scores == neg_scores
#     print(pos_scores.shape)
#     print(neg_scores.shape)
#     predictions = torch.cat((pos_scores, neg_scores),dim=0)
#     print(predictions.shape)
    
    #setting the targets in this way is needed for the different losses we will be workign with
    targets = torch.cat((torch.ones(pos_scores.shape), -1*torch.ones(neg_scores.shape)), dim=0)
#     print(targets.shape)


    if loss == 'pw_hinge':
        return pointwise_hinge_loss(predictions, pw_targets, device, reduction_type)
    elif loss == 'pair_hinge':
        return pairwise_hinge_loss(pos_scores, neg_scores, reduction_type, device, margin_value)
    elif loss == 'pw_logistic':
        return pointwise_logistic_loss(predictions, pw_targets, device, reduction_type)
    elif loss == 'pair_logistic':
        return pairwise_logistic_loss(pos_socres, neg_scores, reduction_type, device)
    elif loss == 'pw_square':
        #targets need to be bingary
        pw_targets = (pw_targets + 1)/2
        return pointwise_square_loss(predictions, pw_targets, reduction_type, device)
    elif loss == 'ce':
        return cross_entropy_neg_sampling(predictions, reduction_type, device)
    elif loss == 'bce':
        targets = (targets + 1)/2
        return bce_loss(predictions, pw_targets, reduction_type)

def bce_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction_type: str):
    loss = nn.BCELoss()
    if reduction_type == 'avg':
        loss = nn.BCELoss(reduction='mean')
    elif reduction_type == 'sum':
        loss = nn.BCELoss(reduction='sum')
    else:
        raise ValueError('Unknown reduction type (%s)' % reduction_type)
    label_smoothing = 0.1
    targets = ((1.0-label_smoothing)*targets) + (1.0/targets.size(1))
    return loss(predictions, targets)



def pointwise_logistic_loss(predictions: torch.Tensor, targets: torch.Tensor, device: torch.device, reduction_type: str):
    '''
    Pointwise log loss softplus(- targets * score)
    targets = label: 1 for ositive, -1 for negative
    score = raw scores
    '''

    # print('max pred', torch.max(predictions))
    softplus = nn.Softplus()
    predictions = torch.clamp(predictions, -75.0, 75.0) #applying clipping avoid expl gradients

    losses = softplus(-targets.to(device) * predictions)

    return reduce_loss(losses, reduction_type)

def pairwise_logistic_loss(pos_predictions: torch.Tensor, neg_predictions: torch.Tensor, reduction_type: str, device: torch.device):
    softplus = nn.Softplus()
    pos_predictions = torch.clamp(pos_predictions, -75.0, 75.0)
    neg_predictions = torch.clamp(neg_predictions, -75.0, 75.0)

    loss = softplus(neg_predictions - pos_predictions)
    return reduce_loss(loss, reduction_type)



def pointwise_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction_type: str, device: torch.device, margin_value: float = 1.0):
    '''
    Point hinge loss: (1-f(x)*l(x))
    l(x) is the label: 1 for positive, -1 for negative sample
    '''

    losses = nn.relu(margin - predictions * targets.to(device))
    return reduce_loss(losses, reduction_type)

def pairwise_hinge_loss(pos_predictions: torch.Tensor, neg_predictions: torch.Tensor, reduction_type: str, device: torch.device, margin_value: float=1.0):
    '''
    pos_socres = scores for positive instances
    neg_scores = scores for neg instances (i.e.)
    pariwise hing loss: relu(margin + positive_scores - negative_scores)
    '''
    loss = torch.relu(margin_value + neg_predictions - pos_predictions)
    return reduce_loss(loss, reduction_type)


def cross_entropy_neg_sampling(predictions: torch.Tensor, reduction_type: str, device: torch.device):
    '''
    predictions: (B, T) = B is the numebr of original test triples, T = 1 + num_neg_samples
    '''
    labels = torch.ones(predictions.size(0))
    # labels[:, 0] = 1

    if reduction_type == 'avg':
        loss = nn.CrossEntropyLoss(reduction='mean')
    elif reduction_type == 'sum':
        loss = nn.CrossEntropyLoss(reduction='sum')

    return loss(predictions, labels)

def pointwise_square_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction_type: str, device: torch.device):
    '''
    Pointwise square loss: (f(x)-l(x))^2
    l(x) is the label : 1 for positive, 0 for negative sample

    '''
    print(predictions[0:10])
    reduction_type = 'avg'
    losses = torch.pow(predictions - targets.to(device), 2)
    return reduce_loss(losses, reduction_type)


def mc_log_loss(predictions: Tuple[torch.Tensor, torch.Tensor],obj_idx: torch.Tensor, subj_idx: torch.Tensor, reduction_type: str ="avg", istucker: bool = False, label_smoothing: float =0.0):
    '''
    Compute the multi class log loss as defined in CP
    predictions: Tuple (sp_prediction tensor, po_prediction tensor).
        Each tensor is a (batch, num_classes) scores matrix - batch is the data size, num_classes is the number of scores
    reduction_type: str
        loss reduction. options ['sum', 'avg']
    obj_idx: tensor
        indeces of the object present in dataset
    subj_idx: Tensor
        indices of the subjects present in dataset

    Returns
    -------
    loss value

    '''
    if not istucker:
        if reduction_type == 'avg':
            loss = nn.CrossEntropyLoss(reduction='mean')
        elif reduction_type == 'sum':
            loss = nn.CrossEntropyLoss(reduction='sum')
        else:
            raise ValueError('Unknown reduction type (%s)' % reduction_type)

        sp_prediction, po_prediction = predictions
        sp_loss = loss(sp_prediction, obj_idx)
        po_loss = loss(po_prediction, subj_idx)
    else:
        if reduction_type == 'avg':
            loss = nn.BCELoss(reduction='mean')
        elif reduction_type == 'sum':
            loss = nn.BCELoss(reduction='sum')
        else:
            raise ValueError('Unknown reduction type (%s)' % reduction_type)

        sp_prediction, po_prediction = predictions
        sp_prediction = torch.sigmoid(sp_prediction)
        po_prediction = torch.sigmoid(po_prediction)

        obj_targets = torch.zeros_like(sp_prediction)
        subj_targets = torch.zeros_like(po_prediction)

        obj_targets[torch.arange(len(obj_targets)), obj_idx.long()]=1
        subj_targets[torch.arange(len(subj_targets)), subj_idx.long()]=1

        label_smoothing = 0.1

        #label smoothing
        obj_targets = ((1.0-label_smoothing)*obj_targets) + (1.0/obj_targets.size(1))
        subj_targets = ((1.0-label_smoothing)*subj_targets) + (1.0/subj_targets.size(1))

        sp_loss = loss(sp_prediction, obj_targets)
        po_loss = loss(po_prediction, subj_targets)

    total_loss = sp_loss + po_loss

    return total_loss
