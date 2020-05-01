import torch
from torch import nn


def mc_log_loss(predictions: torch.LongTensor, targets: torch.LongTensor, reduction_type: string ="avg"):
    '''
    Compute the multi class log loss as defined in CP
    predictions: tensor (batch, num_classes)
        scores matrix - batch is the data size, num_classes is the number of scores
    targets: Tensor
        indices of the true triples of scoring matrix
    reduction_type: str
        loss reduction. options ['sum', 'avg']

    Returns
    -------
    loss value

    '''
