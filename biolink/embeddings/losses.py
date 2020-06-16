import torch
from torch import nn
from typing import Tuple, List, Dict

def reduce_loss(loss: torch.Tensor, reduction_type: str):
    # print('loss shape', loss.size)
    if reduction_type == 'sum':
        return torch.sum(loss, dim=0)
    if reduction_type == 'avg':
        return torch.mean(loss, dim=0)


def compute_kge_loss(predictions: torch.Tensor, loss: str, device: torch.device, reduction_type: str = "avg"):
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
    scrs = torch.split(predictions, int(predictions.shape[0]/2), dim=0)
    pos_scores = scrs[0]
    neg_scores = scrs[1]
    #setting the targets in this way is needed for the different losses we will be workign with
    targets = torch.cat((torch.ones(pos_scores.shape), -1*torch.ones(neg_scores.shape)), dim=0)


    if loss == 'pw_hinge':
        return pointwise_hinge_loss(predictions, targets, device, reduction_type)
    elif loss == 'pw_logistic':
        return pointwise_logistic_loss(predictions, targets, device, reduction_type)
    elif loss == 'pw_square':
        #targets need to be bingary
        targets = (targets + 1)/2
        return pointwise_square_loss(predictions, targets, reduction_type)


def pointwise_logistic_loss(predictions: torch.Tensor, targets: torch.Tensor, device: torch.device, reduction_type: str):
    '''
    Pointwise log loss softplus(- targets * score)
    targets = label: 1 for ositive, -1 for negative
    score = raw scores
    '''

    softplus = nn.Softplus()
    predictions = torch.clamp(predictions, -75.0, 75.0) #applying clipping avoid expl gradients

    losses = softplus(-targets.to(device) * predictions)
    return reduce_loss(losses, reduction_type)

def pointwise_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction_type: str, device: torch.device, margin_value: float = 1.0):
    '''
    Point hinge loss: (1-f(x)*l(x))
    l(x) is the label: 1 for positive, -1 for negative sample
    '''

    losses = nn.relu(margin - predictions * targets.to(device))
    return reduce_loss(losses, reduction_type)

def pointwise_square_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction_type: str):
    '''
    Pointwise square loss: (f(x)-l(x))^2
    l(x) is the label : 1 for positive, 0 for negative sample

    '''
    losses = torch.pow(predictions.cpu() - targets, 2)
    return reduce_loss(losses, reduction_type)


def mc_log_loss(predictions: Tuple[torch.Tensor, torch.Tensor],obj_idx: torch.Tensor, subj_idx: torch.Tensor, reduction_type: str ="avg"):
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
    if reduction_type == 'avg':
        loss = nn.CrossEntropyLoss(reduction='mean')
    elif reduction_type == 'sum':
        loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        raise ValueError('Unknown reduction type (%s)' % reduction_type)

    sp_prediction, po_prediction = predictions
    sp_loss = loss(sp_prediction, obj_idx)
    po_loss = loss(po_prediction, subj_idx)

    total_loss = sp_loss + po_loss

    return total_loss
