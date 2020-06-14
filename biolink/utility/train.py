from biolink.embeddings import KBCModel, KBCModelMCL, mc_log_loss, regulariser
from torch import nn
from torch import optim
from argparse import Namespace
import numpy as np
import logging
import os
import sys
import torch
from biolink.utility.utils import generate_neg_instances

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

def train(model: nn.Module,
        regulariser: str,
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
        train_mc(model, regulariser, optimiser, data, args)
    elif isinstance(model, KBCModel):
        train_not_mc(model,regulariser, optimiser, data, args)
    else:
        raise ValueError("Incorrect model instance given (%s)" %type(model))

def train_not_mc(model: KBCModel, regulariser_str: str, optimiser: optim.Optimizer, data: torch.Tensor, args: Namespace):
    '''
    Training method for not MC models
    '''
    #for each positive generate some negatives
    nb_negs = args.nb_negs
    seed = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    batch_size = args.batch_size
    emb_size = args.embedding_size
    nb_epochs = args.epochs
    # seed = args.seed
    reg_weight = args.reg_weight
    is_quiet = args.quiet

    nb_ents = model.sizes[0]

    nb_batches = np.ceil(data.shape[0]/batch_size)
    regulariser = get_regulariser(regulariser_str, reg_weight)

    for epoch in range(nb_epochs):
        batch_start = 0
        epoch_loss_values = []
        batch_no = 1
        #always have a random permutation
        inputs = data[torch.randperm(data.shape[0]),:]
        while batch_start < data.shape[0]:
            batch_end = min(batch_start + batch_size, data.shape[0])
            input_batch = inputs[batch_start:batch_end]

            #need to generate negatives

            corruptions = generate_neg_instances(input_batch, nb_negs, nb_ents, seed)
            #ensuring you can split between positive and negative examples through the middle
            input_all = torch.cat((input_batch.repeat(nb_negs, 1), corruptions), axis=0).to(device)


            scores, factors = model.score(input_all)
            loss = model.compute_loss(scores)

            reg = regulariser.forward(factors)
            loss += reg

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss_values.append(loss.item())
            batch_no += 1
            if not is_quiet:
                logger.info(f'Epoch {epoch + 1}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss.item():.6f}')

            batch_start += batch_size

        # print(epoch_loss_values)
        # print(type(epoch_loss_values))
        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch + 1}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

def get_regulariser(regulariser_str: str, reg_weight: int):
    '''
    Return the regulariser wanted
    '''
    if regulariser_str == 'n3':
        return regulariser.N3(reg_weight)
    elif regulariser_str == 'f2':
        return regulariser.F2(reg_weight)
    else:
        raise ValueError("Incorrect regulariser name given (%s)" %regulariser_str)


def train_mc(model: KBCModelMCL, regulariser_str: str, optimiser: optim.Optimizer, data: torch.Tensor, args: Namespace):

    '''
    Training method for MC models

    Parameters:
    ----------------
    data: training data

    '''
    nb_negs = args.nb_negs
    seed = args.seed

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    batch_size = args.batch_size
    emb_size = args.embedding_size
    nb_epochs = args.epochs
    # seed = args.seed
    reg_weight = args.reg_weight
    is_quiet = args.quiet

    #set seed
    # np.random.seed(seed)
    # random_state = np.random.RandomState(seed)
    # torch.manual_seed(seed)


    #the embedding matrices should be initialised with the model that has been passed on
    # print('data type', type(data))

    # print('inputs type', type(inputs))

    nb_batches = np.ceil(data.shape[0]/batch_size)
    regulariser = get_regulariser(regulariser_str, reg_weight)

    for epoch in range(nb_epochs):
        batch_start = 0
        epoch_loss_values = []
        batch_no = 1
        #always have a random permutation
        inputs = data[torch.randperm(data.shape[0]),:]
        while batch_start < data.shape[0]:
            batch_end = min(batch_start + batch_size, data.shape[0])
            input_batch = inputs[batch_start:batch_end].to(device)
            pred_sp, pred_po, factors = model.forward(input_batch)

            loss = mc_log_loss((pred_sp, pred_po), input_batch[:, 2], input_batch[:, 0])
            reg = regulariser.forward(factors)
            loss += reg

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss_values.append(loss.item())
            batch_no += 1
            if not is_quiet:
                logger.info(f'Epoch {epoch + 1}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss.item():.6f}')

            batch_start += batch_size

        # print(epoch_loss_values)
        # print(type(epoch_loss_values))
        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch + 1}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

    #should evaluate after the trianing is done
    #evaluate on all dev, test, and train?
