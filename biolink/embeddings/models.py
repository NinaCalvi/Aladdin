# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
from .losses import compute_kge_loss

from torch.nn.init import xavier_normal_, xavier_uniform_
import numpy as np


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def compute_loss(self, scores: torch.Tensor, pos_size: int,
        reduction_type: str = 'avg'):
        '''
        scores: (batch, num_classes) scores matrix)
        targets: indeces of correct prediction
        '''
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                #start by selecting possible rhs
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    #retrieve lhs and rel of queires
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    #computes acutal target scores
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        #removes potential valid choices that could
                        #also be valid assignments
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    #calculate how many have scored higher than target
                    #ideally should be none
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, loss: str,
            device: torch.device, optimiser_name: str, *args, init_size: float = 1e-3,
    ):
        '''
        loss - what type of loss
        '''
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank


        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True


        self.lhs = nn.Embedding(sizes[0], rank, sparse=sparse_)
        self.rel = nn.Embedding(sizes[1], rank, sparse=sparse_)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=sparse_)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

        self.loss = loss
        self.device = device

        self.args = args

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True), (lhs, rel, rhs)

    def forward(self, x, predict_lhs = False):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        #score subject predicate
        score_sp =  (lhs * rel) @ self.rhs.weight.t()

        score_po = (rhs * rel) @ self.lhs.weight.t()

        return score_sp, score_po, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

    def compute_loss(self, scores, pos_size, reduction_type='sum'):
        return compute_kge_loss(scores, self.loss, self.device, pos_size, reduction_type, self.args[0].loss_margin)


class TransE(KBCModel):
    def __init__(
            self, sizes:Tuple[int, int, int], rank: int, loss: str,
            device: torch.device, optimiser_name: str, *args,  init_size: float = 1e-3, norm_: str = 'l1'
    ):
        """
        Parameters
        ------
        sizes: number of each lhs, rel, rhs entities
        rank: size of embeddings
        init_size: value to initialize embeddings
        norm_: how to normalise the scoring function
        *args: should ideally just be the argument parser object
        """
        super(TransE, self).__init__()
        self.sizes = sizes
        self.rank = rank


        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.lhs = nn.Embedding(sizes[0], rank, sparse=sparse_) #removed sparse
        self.rel = nn.Embedding(sizes[1], rank, sparse=sparse_)
        # self.rhs = nn.Embedding(sizes[2], rank)

        # self.lhs.weight.data *= init_size
        # self.rel.weight.data *= init_size
        # self.rhs.weight.data *= init_size

        self.norm_ = norm_
        self.loss = loss

        self.device = device
        self.args = args

    def score(self, x):
        """
        Compute TransE scores for a set of triples
        """

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.lhs(x[:, 2])

        interactions = lhs + rel - rhs
        if self.norm_ == 'l1':
            scores = torch.norm(interactions, 1, -1)
        elif self.norm_ == 'l2':
            scores = torch.norm(interactions, 2, -1)
        else:
            raise ValueError("Unknwon norm type given (%s)" % self.norm_)

        #NOTE: am returning negative score
        #from sameh he does this to comply with loss objective?
        return -scores, (lhs, rel, rhs)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.lhs(x[:, 2])

        #need to compute the difference with each
        #TODO: FINISH THIS!!
        #adding new vertical dimension
        scores_sp = None
        scores_po = None

        if self.norm_ == 'l1':
            norm = 1
        elif self.norm_ == 'l2':
            norm = 2
        else:
            raise ValueError("Unknwon norm type given (%s)" % self.norm_)


        # interactions_sp = (l + rl)[:,None] - self.rhs.weight
        scores_sp = torch.norm((lhs + rel)[:,None] - self.lhs.weight, norm, dim=2)

        scores_po = torch.norm((self.lhs.weight + rel[:,None]) - rhs[:,None], norm, dim=2)
            # scores_po_tmp = torch.norm(interactions_po, norm, dim=2)
            # del interactions_po
            # torch.cuda.empty_cache()
            #should take the norm across each row of matrix

        return -scores_sp, -scores_po, (lhs, rel, rhs)


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """
        Get the chunk of the target vars
        """
        return self.lhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data + self.rel(queries[:, 1]).data

    def compute_loss(self, scores, pos_size, reduction_type='sum'):
        return compute_kge_loss(scores, self.loss, self.device, pos_size, reduction_type, self.args[0].loss_margin)



class RotatE(KBCModel):

    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, loss: str, device: torch.device,
            optimiser_name: str,  *args, init_size: float = 1e-3):
        super(RotatE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.init_size = init_size

        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        # self.embeddings = nn.ModuleList([
        #     nn.Embedding(s, 2 * rank, sparse=sparse_)
        #     for s in sizes[:2]
        # ])

        self.embeddings = nn.Embedding(sizes[0], 2* rank, sparse=sparse_)
        self.rels = nn.Embedding(sizes[1], rank, sparse=sparse_)

        self.args =args
        self.device = device
        self.loss = loss

    def init(self):
#         if self.loss == 'rotate_loss':
#             nn.init.xavier_normal_(self.embeddings.weight.data)
#             nn.init.normal_(self.rels.weight.data)
#         else:  
#             nn.init.uniform_(self.embeddings.weight.data, a = -1, b=1)
#             nn.init.uniform_(self.rels.weight.data, a = -1, b=1)
        
#         nn.init.uniform_(self.embeddings.weight.data, a = -1, b=1)
#         nn.init.uniform_(self.rels.weight.data, a = -1, b=1)
        
        nn.init.xavier_normal_(self.embeddings.weight.data)
        nn.init.uniform_(self.rels.weight.data, a = 0, b=1)
#         nn.init.normal_(self.rels.weight.data)

    def score(self, x):
        lhs = self.embeddings(x[:, 0])
        ph_rel = self.rels(x[:, 1])
#         pi = 3.14159265358979323846
#         if self.loss == 'rotate_loss':
#             rel = ph_rel * 2 * np.pi
#         else:
#             rel = ph_rel*np.pi
            
        rel = ph_rel*np.pi*2
        rhs = self.embeddings(x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rel_re, rel_im = torch.cos(rel), torch.sin(rel)

        score_sp_re = lhs[0] * rel_re - lhs[1] * rel_im
        score_sp_im = lhs[0] * rel_im + lhs[1] * rel_re

        score_sp_re = score_sp_re - rhs[0]
        score_sp_im = score_sp_im - rhs[1]
        
#         print('score sp re shape', score_sp_re.shape)
#         print('score sp im shape', score_sp_im.shape)

        score_sp_re = torch.stack((score_sp_re, score_sp_im), dim=0)
        
#         print('stacked score sp shape', score_sp_re.shape)
        
        score_sp_re = torch.norm(score_sp_re, dim=0)
        

        return -torch.norm(score_sp_re, dim=1, p=1, keepdim=True), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward(self, x):

        lhs = self.embeddings(x[:, 0])
        ph_rel = self.rels(x[:, 1])
        rhs = self.embeddings(x[:, 2])
        
#         pi = 3.14159265358979323846
        rel = ph_rel*np.pi*2

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        # rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rel_re, rel_im = torch.cos(rel), torch.sin(rel)

        # phase_relation = rel/(self.embedding_range.item()/pi)

        to_score = self.embeddings.weight[:, :self.rank], self.embeddings.weight[:, self.rank:]

        score_sp_re = lhs[0] * rel_re - lhs[1] * rel_im
        score_sp_im = lhs[0] * rel_im + lhs[1] * rel_re
        # print('sc sp shape', score_sp_re.shape)
        score_sp_re = score_sp_re.unsqueeze(1) - to_score[0]
        score_sp_im = score_sp_im.unsqueeze(1) - to_score[1]
        # print('sc sp shape after toscore', score_sp_re.shape)
        score_sp = torch.stack([score_sp_re, score_sp_im], dim=0)
        score_sp = torch.norm(torch.norm(score_sp, dim=0), dim=2, p=1)
        
        del score_sp_re
        del score_sp_im
        

#         print('score_sp shape', score_sp.shape)

        score_po_re = to_score[0].unsqueeze(1) * rel_re - to_score[1].unsqueeze(1) * rel_im
        score_po_im = to_score[0].unsqueeze(1) * rel_im + to_score[1].unsqueeze(1) * rel_re
        # print('sc po shape', score_po_re.shape)
        score_po_re = score_po_re - rhs[0]
        score_po_im = score_po_im - rhs[1]
        # print('sc po after diff', score_po_re.shape)
        score_po = torch.stack([score_po_re, score_po_im], dim=0)
        score_po = torch.norm(torch.norm(score_po, dim=0), dim=2, p=1).t()
        
        del score_po_re
        del score_po_im
        torch.cuda.empty_cache()
        

#         print('score po, ', score_po.shape)

        return -score_sp, -score_po, (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def compute_loss(self, scores, pos_size, reduction_type='sum'):
        return compute_kge_loss(scores,self.loss, self.device, pos_size, reduction_type, self.args[0].loss_margin)



class TuckEr(KBCModel):
    """
    The Tucker Embedding model
    """

    def __init__(
            self, sizes: Tuple[int, int, int], rank_e: int, rank_rel: int, loss: str,
            device: torch.device, optimiser_name: str, *args, init_size: float = 1e-3, **kwargs,
    ):
        '''
        loss - what type of loss
        '''
        super(TuckEr, self).__init__()
        self.sizes = sizes
        self.rank_e = rank_e
        self.rank_r = rank_rel


        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True


        #suggests that relations and entities have different ranks as well potentailly
        self.ent = nn.Embedding(sizes[0], rank_e)
        self.rel = nn.Embedding(sizes[1], rank_rel)



        if torch.cuda.is_available():
            self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank_rel, rank_e, rank_e)), dtype=torch.float, device="cuda", requires_grad=True))
        else:
            self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank_rel, rank_e, rank_e)), dtype=torch.float, requires_grad=True))


        # self.bn0 = nn.BatchNorm1d(rank_e)
        # self.bn1 = nn.BatchNorm1d(rank_e)

        self.loss = loss
        self.device = device

        # self.input_dropout = nn.Dropout(kwargs["input_dropout"])
        # self.hidden_dropout1 = nn.Dropout(kwargs["hidden_dropout1"])
        # self.hidden_dropout2 = nn.Dropout(kwargs["hidden_dropout2"])

        self.args = args


    def init(self):
        xavier_normal_(self.ent.weight.data)
        xavier_normal_(self.rel.weight.data)

    def score(self, x):
        lhs = self.ent(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.ent(x[:, 2])

        W_mat = torch.mm(rel, self.W.view(rel.size(1), -1))
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))

        return torch.sum(lhs * rel * rhs, 1, keepdim=True), (lhs, rel, rhs, self.W)

    def forward(self, x, predict_lhs = False):
        lhs = self.ent(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.ent(x[:, 2])

        # x = self.bn0(lhs)
        # x2 = self.bn0(rhs)

        # x = self.input_dropout(x)
        # x2 = self.input_dropout(x2)

        x = x.view(-1, 1, lhs.size(1))
        x2 = x2.view(-1, rhs.size(1), 1)

        W_mat = torch.mm(rel, self.W.view(rel.size(1), -1))
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))

        #THIS HIDDEN DROPOUT I NEED TO UNDERSTAND BETTER
        # W_mat = self.hidden_dropout1(W_mat)


        x = torch.bmm(x, W_mat)
        x2 = torch.bmm(W_mat, x2)

        x = x.view(-1, lhs.size(1))
        x2 = x2.view(-1, rhs.size(1))

        # x = self.bn1(x)
        # x2 = self.bn1(x2)

        # x = self.hidden_dropout2(x)
        # x2 = self.hidden_dropout2(x2)

        x = torch.mm(x, self.ent.weight.transpose(1,0))
        x2 = torch.mm(x2, self.ent.weight.transpose(1,0))

        # pred_sp = torch.sigmoid(x)
        # pred_po = torch.sigmoid(x2)


        return x, x2, (lsh, rel, rhs, self.W) #unsure whether should add w?

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.ent.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.ent(queries[:, 0]).data * self.rel(queries[:, 1]).data

    def compute_loss(self, scores, pos_size, reduction_type='avg'):
        return compute_kge_loss(scores, self.loss, self.device, pos_size, reduction_type, self.args[0].loss_margin)





class DistMult(KBCModel):
    """
    The DistMult Embedding model (DistMult)
    """

    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, loss: str,
            device: torch.device, optimiser_name: str, *args, init_size: float = 1e-3,
    ):
        '''
        loss - what type of loss
        '''
        super(DistMult, self).__init__()
        self.sizes = sizes
        self.rank = rank

        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.ent = nn.Embedding(sizes[0], rank, sparse=sparse_)
        self.rel = nn.Embedding(sizes[1], rank, sparse=sparse_)

        self.ent.weight.data *= init_size
        self.rel.weight.data *= init_size

        self.loss = loss
        self.device = device

        self.args = args

    def score(self, x):
        lhs = self.ent(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.ent(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True), (lhs, rel, rhs)

    def forward(self, x, predict_lhs = False):
        lhs = self.ent(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.ent(x[:, 2])
        #score subject predicate
        score_sp =  (lhs * rel) @ self.ent.weight.t()

        score_po = (rhs * rel) @ self.ent.weight.t()

        return score_sp, score_po, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.ent.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.ent(queries[:, 0]).data * self.rel(queries[:, 1]).data

    def compute_loss(self, scores, pos_size, reduction_type='sum'):
        return compute_kge_loss(scores, self.loss, self.device, pos_size, reduction_type, self.args[0].loss_margin)





class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, loss: str, device: torch.device, optimiser_name: str, *args, init_size: float = 1e-3):
        '''
        loss - what type of loss to use
        '''
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True


        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=sparse_) #REMOVED SPARSE TRUE
            for s in sizes[:2]
        ])

        # nn.init.xavier_normal(self.embeddings[0].weight)
        # nn.init.xavier_normal(self.embeddings[1].weight)
        #
        # self.embeddings[0].weight.data *= init_size
        # self.embeddings[1].weight.data *= init_size

        self.loss = loss
        self.device = device
        self.args = args

    def init(self):
        nn.init.xavier_normal_(self.embeddings[0].weight)
        nn.init.xavier_normal_(self.embeddings[1].weight)

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        score_sp =  (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
            )

        score_po = (
                (rhs[0] * rel[0] + rhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (rhs[1] * rel[0] - rhs[0] * rel[1]) @ to_score[1].transpose(0, 1)
            )


        return score_sp, score_po, (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def compute_loss(self, scores, pos_size, reduction_type='sum'):
        return compute_kge_loss(scores,self.loss, self.device, pos_size, reduction_type, self.args[0].loss_margin)



#STILL NEEDS TO BE FINISHED
class TriVec(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, loss: str, device: torch.device, optimiser_name: str, *args, init_size: float = 1e-3):
        '''
        loss - what type of loss to use
        '''
        super(TriVec, self).__init__()
        self.sizes = sizes
        self.rank = rank

        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 3 * rank, sparse=sparse_) #REMOVED SPARSE TRUE
            for s in sizes[:2]
        ])

        nn.init.xavier_normal(self.embeddings[0].weight)
        nn.init.xavier_normal(self.embeddings[1].weight)
        #
        # self.embeddings[0].weight.data *= init_size
        # self.embeddings[1].weight.data *= init_size

        self.loss = loss
        self.device = device
        self.args = args

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:]



        return torch.sum(
            lhs[0] * rel[0] * rhs[2] +
            lhs[1] * rel[1] * rhs[1] +
            lhs[2] * rel[2] * rhs[0],
            1, keepdim=True
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2 + lhs[2] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2 + rhs[2] ** 2)
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:self.rank*2], to_score[:, self.rank*2:]

        score_sp =  (
                lhs[0] * rel[0] @ to_score[2].transpose(0, 1) +
                lhs[1] * rel[1] @ to_score[1].transpose(0, 1) +
                lhs[2] * rel[2] @ to_score[0].transpose(0, 1)
            )

        score_po = (
                rhs[2] * rel[0] @ to_score[0].transpose(0, 1) +
                rel[1] * rhs[1] @ to_score[1].transpose(0, 1) +
                rel[2] * rhs[0] @ to_score[2].transpose(0, 1)
            )


        return score_sp, score_po, (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2 + lhs[2] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2 + rhs[2] ** 2)
        )


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:]

        return torch.cat([
            lhs[2] * rel[2],
            lhs[1] * rel[1],
            lhs[0] * rel[0]
        ], 1)

    def compute_loss(self, scores, pos_size, reduction_type='sum'):
        return compute_kge_loss(scores,self.loss, self.device, pos_size, reduction_type, self.args[0].loss_margin)
