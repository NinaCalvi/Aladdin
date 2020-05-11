from embeddings import KBCModel, KBCModelMCL, mc_log_loss, regulariser
from torch import nn
from torch import optim
from argparse import Namespace
import numpy as np
import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

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

def get_regulariser(regulariser_str: string, reg_weight: int):
    '''
    Return the regulariser wanted
    '''
    if regulariser_str == 'n3':
        return regulariser.N3(reg_weight)
    elif regulariser_str == 'f2':
        return regulariser.F2(reg_weight)
    else:
        raise ValueError("Incorrect regulariser name given (%s)" %regulariser_str)


def train_mc(model: KBCModelMCL, regulariser_str: string, optimiser: optim.Optimizer, data: torch.Tensor, args: Namespace):

    '''
    Training method for MC models

    Parameters:
    ----------------
    data: training data

    '''
    nb_negs = args.nb_negs

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    batch_size = args.batch_size
    emb_size = args.emb_size
    nb_epochs = args.nb_epochs
    seed = args.seed
    reg_weight = args.reg_weight
    is_quiet = args.quiet

    #set seed
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    #the embedding matrices should be initialised with the model that has been passed on
    inputs = data[torch.randperm(data.shape[0]),:]
    epoch_loss = []
    for epoch in range(nb_epochs):
        batch_start = 0
        while batch_start < data.shape[0]:
            batch_end = min(batch_start + batch_size, data.shape[0])
            input_batch = input[batch_start:batch_end].to(device)
            pred_sp, pred_po, factors = model.forward(input_batch)

            loss = mc_log_loss((pred_sp, pred_po), input_batch[:, 2], input_batch[:, 0])
            reg = get_regulariser(regulariser_str, reg_weight).forward(factors)
            loss += reg

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss.append(loss)
            if not is_quiet:
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f}')

            batch_start += batch_size

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

    #should evaluate after the trianing is done
