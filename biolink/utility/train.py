from biolink.embeddings import KBCModel, KBCModelMCL, TuckEr_MC, mc_log_loss, regulariser
from biolink.eval import evaluate
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

BAD_PERFORMING = False
bad_mrr = 0

def train(model: nn.Module,
        regulariser: str,
        optimiser: optim.Optimizer,
        data: torch.Tensor,
        valid_data: torch.Tensor,
        all_data: torch.Tensor,
        args: Namespace,
        scheduler: optim.lr_scheduler = None):
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
    print(type(model))
    if isinstance(model, KBCModelMCL):
        train_mc(model, regulariser, optimiser, data, valid_data, all_data, args, scheduler=scheduler)
    elif isinstance(model, KBCModel):
        train_not_mc(model,regulariser, optimiser, data,  valid_data, all_data, args)
    else:
        raise ValueError("Incorrect model instance given (%s)" %type(model))

def train_not_mc(model: KBCModel, regulariser_str: str, optimiser: optim.Optimizer, data: torch.Tensor, valid_data: torch.Tensor, all_data: torch.Tensor, args: Namespace):
    '''
    Training method for not MC models
    '''
    #for each positive generate some negatives

    global BAD_PERFORMING
    global bad_mrr


    nb_negs = args.nb_negs
    seed = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'NOT MC')
    logger.info(f'Device: {device}')

    batch_size = args.batch_size
    emb_size = args.embedding_size
    nb_epochs = args.epochs
    # seed = args.seed
    reg_weight = args.reg_weight
    is_quiet = args.quiet

    nb_ents = model.sizes[0]
    valid_every = args.valid_stp
    valid = args.valid #boolean
    logger.info(f'valid: {valid}')

    nb_batches = np.ceil(data.shape[0]/batch_size)
    regulariser = get_regulariser(regulariser_str, reg_weight)

    best_val_mrr = None

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
            # input_batch = input_batch.repeat(nb_negs, 1)
            # input_all = torch.cat((input_batch.repeat(nb_negs, 1), corruptions), axis=0).to(device)
            input_all = torch.cat((input_batch, corruptions), axis=0).to(device)

            pos_input = input_batch.to(device)
            neg_input = corruptions.to(device)



            optimiser.zero_grad()

            scores_pos, factors_pos = model.score(pos_input)
            socres_neg, factors_neg = model.score(neg_input)
            scores_pos = scores_pos.repeat(nb_negs, 1)
            factors_pos = factors_pos.repeat(nb_negs, 1)


            # scores, factors = model.score(input_all)
            loss = model.compute_loss(torch.cat((scores_pos, scores_neg), axis=0), scores_pos.shape[0])

            # loss = model.compute_loss(scores, input_batch.shape[0])
            # print(model.embeddings[0].weight.data)
            reg = regulariser.forward(torch.cat((factors_pos, factors_neg), axis=0))
            loss += reg

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


        if (((epoch+1) % valid_every) == 0) and valid:
            logger.info(f'Validating')
            val_metrics = evaluate(model, valid_data, all_data, batch_size, device, validate=True)
            if best_val_mrr is None:
                best_val_mrr = val_metrics['MRR']
            elif best_val_mrr >= val_metrics['MRR']:
                logger.info(f'Filtered MRR validation decreased, therefore stop')
                logger.info(f'Best validation metrics {best_val_mrr}')
                break
            else:
                best_val_mrr = val_metrics['MRR']
                logger.info(f'Best validation metrics {best_val_mrr}')
            if (epoch+1) == 50:
                if best_val_mrr < 0.06:
                    logger.info(f'Applying old dog tricks, ending training. MRR is {best_val_mrr}')
                    BAD_PERFORMING = True
                    bad_mrr = best_val_mrr
                    break


def get_regulariser(regulariser_str: str, reg_weight: float, tucker_reg_weight: float = None):
    '''
    Return the regulariser wanted
    '''
    if regulariser_str == 'n3':
        return regulariser.N3(reg_weight, tucker_reg_weight)
    elif regulariser_str == 'f2':
        return regulariser.F2(reg_weight)
    else:
        raise ValueError("Incorrect regulariser name given (%s)" %regulariser_str)


def train_mc(model: KBCModelMCL, regulariser_str: str, optimiser: optim.Optimizer, data: torch.Tensor, valid_data: torch.Tensor, all_data: torch.Tensor, args: Namespace, scheduler=None):

    '''
    Training method for MC models

    Parameters:
    ----------------
    data: training data

    '''
    nb_negs = args.nb_negs
    seed = args.seed
    valid = args.valid

    global BAD_PERFORMING
    global bad_mrr

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

    if isinstance(model, TuckEr_MC):
        tucker = True
    else:
        tucker = False

    logger.info(f'is tucker {tucker}')
    logger.info(f'settin tucker to FALSE')
    tucker=False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    batch_size = args.batch_size
    emb_size = args.embedding_size
    nb_epochs = args.epochs
    # seed = args.seed
    reg_weight = args.reg_weight
    tucker_reg_weight = args.tucker_reg_weight
    is_quiet = args.quiet

    #set seed
    # np.random.seed(seed)
    # random_state = np.random.RandomState(seed)
    # torch.manual_seed(seed)


    #the embedding matrices should be initialised with the model that has been passed on
    # print('data type', type(data))

    # print('inputs type', type(inputs))

    nb_batches = np.ceil(data.shape[0]/batch_size)
    if tucker_reg_weight > -1:
        regulariser = get_regulariser(regulariser_str, reg_weight, tucker_reg_weight)
    else:
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

            loss = mc_log_loss((pred_sp, pred_po), input_batch[:, 2], input_batch[:, 0], istucker=tucker, label_smoothing=args.label_smoothing)
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
        if scheduler is not None:
            scheduler.step()


        if ((epoch+1) == 50) and valid:
            logger.info(f'Validating')
            val_metrics = evaluate(model, valid_data, all_data, batch_size, device, validate=True)

            mrr = val_metrics['MRR']
            if mrr < 0.06:
                logger.info(f'Applying old dog tricks, ending training. MRR is {mrr}')
                BAD_PERFORMING = True
                bad_mrr = mrr
                break
        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch + 1}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

    #should evaluate after the trianing is done
    #evaluate on all dev, test, and train?
