from embeddings import KBCModel, KBCModelMCL
from torch import nn
from torch import optim
from argparse import Namespace
import numpy as np

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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    emb_size = args.emb_size
    nb_epochs = args.nb_epochs
    seed = args.seed

    #set seed
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    #the embeddings should be initialised with the model that has been passed on
    inputs = data[torch.randperm(data.shape[0]),:]
    for epoch in range(nb_epochs):
        batch_start = 0
        while batch_start < data.shape[0]:
            batch_end = min(batch_start + batch_size, data.shape[0])
            input_batch = input[batch_start:batch_end].to(device)
            




    pass
