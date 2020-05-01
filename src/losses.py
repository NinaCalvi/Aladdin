import torch
from torch import nn


def compute_kge_loss(predictions: torch.Tensor, loss: string, reduction_type: string = "avg"):
    #looking at sameh's code for this situation
    #they binarize the positive results (1) and the negative results (0)
    #and use that as the targets
    #essentially the predictions are equally split between positive and negative predictions
    #through the middle

def mc_log_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction_type: string ="avg"):
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
