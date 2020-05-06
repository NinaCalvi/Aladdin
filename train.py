from embeddings import KBCModel, KBCModelMCL
from torch import nn
from torch import optim
from argparse import Namespace

def train(model: nn.Module,
        regulariser: string,
        optimiser: optim.Optimizer,
        data: torch.Tensor,
        args: Namespace):
    '''
    Method that instantiates the trianing of the model

    Parameters:
    -----------
    model: model to be trained (with the chosen loss defined)
    regulariser: regulariser to be used
    optimiser: what optimiser to use
    data: the training data
    args: arguments of input
    '''
    if isinstance(model, KBCModelMCL):
        train_mc(model, regulariser, optimiser)
    elif isinstance(model, KBCModel):
        train_not_mc(model,regulariser, optimiser)
    else:
        raise ValueError("Inccorrect model instance given (%s)" %type(model))

def train_not_mc():
    pass

def train_mc(model: KBCModel, regulariser: string, optimiser: optim.Optimizer, data: torch.Tensor, args: Namespace):
    nb_negs = args.nb_negs

    batch_size = args.batch_size
    emb_size = args.emb_size
    nb_epochs = args.nb_eppchs


    pass
