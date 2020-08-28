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
from .losses import mc_log_loss

from torch.nn.init import xavier_normal_
from biolink.embeddings import TransE
import numpy as np
import os


class KBCModelMCL(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass


    def compute_loss(self, predictions: Tuple[torch.Tensor, torch.Tensor],
        obj_idx: torch.Tensor, subj_idx: torch.Tensor, reduction_type: str = 'avg'):
        '''
        obj_idx, subj_idx: all indeces in the training for subject/object needed to compute neg lloss
        '''
        return mc_log_loss(scores, subj_idx, obj_idx,  reduction_type)


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


class CP_MC(KBCModelMCL):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            optimiser_name: str, init_size: float = 1e-3,
    ):
        '''
        loss - what type of loss
        '''
        super(CP_MC, self).__init__()
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

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        #score subject predicate
        score_sp =  (lhs * rel) @ self.rhs.weight.t()

        #score predicate object
        score_po = (rhs * rel) @ self.lhs.weight.t()
        return score_sp, score_po, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class DistMult_MC(KBCModelMCL):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            optimiser_name: str, init_size: float = 1e-3,
    ):
        '''
        loss - what type of loss
        '''
        super(DistMult_MC, self).__init__()
        self.sizes = sizes
        self.rank = rank

        # self.emb = nn.Embedding(sizes[0], rank, sparse=True)
        # self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.emb = nn.Embedding(sizes[0], rank, sparse=sparse_)
        self.rel = nn.Embedding(sizes[1], rank, sparse=sparse_) #adam does not supposrt sparse

        self.emb.weight.data *= init_size
        self.rel.weight.data *= init_size

    def score(self, x):
        lhs = self.emb(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.emb(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.emb(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.emb(x[:, 2])
        #score subject predicate
        score_sp =  (lhs * rel) @ self.emb.weight.t()

        #score predicate object
        score_po = (rhs * rel) @ self.emb.weight.t()
        return score_sp, score_po, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.emb.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.emb(queries[:, 0]).data * self.rel(queries[:, 1]).data

class TransE_MC(KBCModelMCL):
    def __init__(
            self, sizes:Tuple[int, int, int], rank: int,
            optimiser_name: str, init_size: float = 1e-3, norm_: str = 'l1',
    ):
        """
        Parameters
        ------
        sizes: number of each lhs, rel, rhs entities
        rank: size of embeddings
        init_size: value to initialize embeddings
        norm_: how to normalise the scoring function
        """
        super(TransE_MC, self).__init__()
        self.sizes = sizes
        self.rank = rank

        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.lhs = nn.Embedding(sizes[0], rank, sparse=sparse_) #removed sparse - ADAM does not accept this should add option
        self.rel = nn.Embedding(sizes[1], rank, sparse=sparse_) #removed sparse - ADAM does not accept this should add option
        # self.hs = nn.Embedding(sizes[2], rank) #removed sparse - ADAM does not accept this should add option

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        # self.rhs.weight.data *= init_size

        self.norm_ = norm_

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
        return -scores

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.lhs(x[:, 2]) #corrected initially has rhs

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




class ComplEx_MC(KBCModelMCL):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            optimiser_name: str, pret: bool, transe: str = None, init_size: float = 1e-3):

        super(ComplEx_MC, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.init_size = init_size
        self.transe = transe

        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.pret = pret
        
        if self.pret:
            self.rank = 384
            
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=sparse_)
            for s in sizes[:2]
        ])

    def init(self):
        if self.pret:
            if self.transe is None:
                print('PRET')
                embs = np.loadtxt(os.path.join(os.getcwd(), 'embeddings/correct_order.txt'))
                print('embs shape', embs.shape)
                self.embeddings[0] = self.embeddings[0].from_pretrained(torch.FloatTensor(embs), freeze=False)
                print('ents shape', self.embeddings[0].weight.data.shape)
                self.embeddings[1].weight.data *= self.init_size
                print('rels shape', self.embeddings[1].weight.data.shape)
            else:
                transe_model = TransE((self.sizes[0], self.sizes[1], self.sizes[0]), 200, 'pair_hinge', None, 'adagrad', None, 'l2')
                transe_model.load_state_dict(torch.load(self.transe + '.pt'))
                self.embeddings[0] = self.embeddings[0].from_pretrained(transe_model.lhs.weight.data)
                self.emebddings[1] = self.embeddigns[1].from_pretrained(transe_model.rel.weight.data)
        else:   
            self.embeddings[0].weight.data *= self.init_size
            self.embeddings[1].weight.data *= self.init_size



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
        )
    
    
    def predict(self, pred, head=None, tail=None):
        if head is None and tail is None:
            logger.info('Error, either add head or tail')
            return
        
        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        rel = self.embeddings[1](pred[:, 0])
        rel = rel[:, :self.rank], rel[:, self.rank:]

        if head is None:
            rhs = self.embeddings[0](tail[:, 0])
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            print(rhs[0].shape, rhs[1].shape)
            score_po = (
            (rhs[0] * rel[0] + rhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (rhs[1] * rel[0] - rhs[0] * rel[1]) @ to_score[1].transpose(0, 1)
        )
            return score_po
        
        else:
            lhs = self.embeddings[0](head[:, 0])
            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            score_sp =  (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))
            return score_sp
        

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
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

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



class RotatE_MC(KBCModelMCL):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            optimiser_name: str, init_size: float = 1e-3):
        super(RotatE_MC, self).__init__()
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

    def init(self):
        self.embeddings.weight.data *= self.init_size
        self.rels.weight.data *= self.init_size



    def score(self, x):
        lhs = self.embeddings(x[:, 0])
        rel = self.rels(x[:, 1])
        rhs = self.embeddings(x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):

        lhs = self.embeddings(x[:, 0])
        rel = self.rels(x[:, 1])
        rhs = self.embeddings(x[:, 2])

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
        score_sp = torch.norm(torch.norm(score_sp, dim=0), dim=2, p=2)

        # print('score_sp shape', score_sp.shape)

        score_po_re = to_score[0].unsqueeze(1) * rel_re - to_score[1].unsqueeze(1) * rel_im
        score_po_im = to_score[0].unsqueeze(1) * rel_im + to_score[1].unsqueeze(1) * rel_re
        # print('sc po shape', score_po_re.shape)
        score_po_re = score_po_re - rhs[0]
        score_po_im = score_po_im - rhs[1]
        # print('sc po after diff', score_po_re.shape)
        score_po = torch.stack([score_po_re, score_po_im], dim=0)
        score_po = torch.norm(torch.norm(score_po, dim=0), dim=2, p=2).t()

        # print('score po, ', score_po.shape)

        return score_sp, score_po, (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            rel,
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



class TuckEr_MC(KBCModelMCL):
    """
    The Tucker Embedding model
    """

    def __init__(
            self, sizes: Tuple[int, int, int], rank_e: int, rank_rel: int, optimiser_name: str, init_size: float = 1e-3, **kwargs,
    ):
        '''
        loss - what type of loss
        '''
        super(TuckEr_MC, self).__init__()
        self.sizes = sizes
        self.rank_e = rank_e
        self.rank_r = rank_rel
        self.init_size = init_size
        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.ent = nn.Embedding(sizes[0], rank_e, sparse=sparse_) #removed sparse - ADAM does not accept this should add option
        self.rel = nn.Embedding(sizes[1], rank_rel, sparse=sparse_) #removed sparse - ADAM does not accept this should add option
        # self.hs = nn.Embedding(sizes[2], rank) #removed sparse - ADAM does not accept this should add option



        # #suggests that relations and entities have different ranks as well potentailly
        # self.ent = nn.Embedding(sizes[0], rank_e)
        # self.rel = nn.Embedding(sizes[1], rank_rel)
        #
        # xavier_normal_(self.ent.weight.data)
        # xavier_normal_(self.rel.weight.data)


        if torch.cuda.is_available():
            self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank_rel, rank_e, rank_e)), dtype=torch.float, requires_grad=True))
        else:
            self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank_rel, rank_e, rank_e)), dtype=torch.float, requires_grad=True))


        # self.bn0 = nn.BatchNorm1d(rank_e)
        # self.bn1 = nn.BatchNorm1d(rank_e)

        #
        # self.input_dropout = nn.Dropout(kwargs["input_dropout"])
        # self.hidden_dropout1 = nn.Dropout(kwargs["hidden_dropout1"])
        # self.hidden_dropout2 = nn.Dropout(kwargs["hidden_dropout2"])


    def init(self):
        self.ent.weight.data *= self.init_size
        self.rel.weight.data *= self.init_size


    def score(self, x):

        lhs = self.ent(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.ent(x[:, 2])

        # x = self.bn0(lhs)
        # x2 = self.bn0(rhs)
        #
        # x = self.input_dropout(x)
        # x2 = self.input_dropout(x2)

        x = lhs
        x2 = rhs

        x = x.view(-1, 1, lhs.size(1))

        W_mat = torch.mm(rel, self.W.view(rel.size(1), -1))
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))

        #THIS HIDDEN DROPOUT I NEED TO UNDERSTAND BETTER
        # W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, lhs.size(1))
        # x = self.bn1(x)
        # x = self.hidden_dropout2(x)
        x = torch.sum(x * x2, 1, keepdim=True)
        return x
        #return torch.sigmoid(x)

        # return torch.sigmoid(x)

    def forward(self, x, predict_lhs = False):
        lhs = self.ent(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.ent(x[:, 2])

        # x = self.bn0(lhs)
        # x2 = self.bn0(rhs)

        # x = self.input_dropout(x)
        # x2 = self.input_dropout(x2)

        x = lhs
        x2 = rhs

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

        pred_sp = x
        pred_po = x2

        # pred_sp = torch.sigmoid(x)
        # pred_po = torch.sigmoid(x2)


        return pred_sp, pred_po, (lhs, rel, rhs, self.W) #unsure whether should add w?

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.ent.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.ent(queries[:, 0]).data * self.rel(queries[:, 1]).data



class TriVec_MC(KBCModelMCL):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            optimiser_name: str, init_size: float = 1e-3):
        super(TriVec_MC, self).__init__()
        self.sizes = sizes
        self.rank = rank

        if optimiser_name == 'adam':
            sparse_ = False
        else:
            sparse_ = True

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 3 * rank, sparse=sparse_)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size


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
