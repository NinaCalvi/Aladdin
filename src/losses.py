import torch
from torch import nn


def compute_kge_loss(predictions: torch.Tensor, loss: string, reduction_type: string = "avg"):
    #looking at sameh's code for this situation
    #they binarize the positive results (1) and the negative results (0)
    #and use that as the targets
    #essentially the predictions are equally split between positive and negative predictions
    #through the middle
    pass

def mc_log_loss(predictions: Tuple[torch.Tensor, torch.Tensor],obj_idx: torch.Tensor, subj_idx: torch.Tensor, reduction_type: string ="avg"):
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
