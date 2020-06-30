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
            init_size: float = 1e-3,
    ):
        '''
        loss - what type of loss
        '''
        super(CP_MC, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

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

class TransE_MC(KBCModelMCL):
    def __init__(
            self, sizes:Tuple[int, int, int], rank: int,
            init_size: float = 1e-3, norm_: str = 'l1',
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

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

        self.norm_ = norm_

    def score(self, x):
        """
        Compute TransE scores for a set of triples
        """

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        interactions = lhs + rel - rhs
        if self.norm_ == 'l1':
            scores = torch.norm(interactions, 1, -1)
        if self.norm_ == 'l2':
            scores = torch.norm(interactions, 2, -1)
        else:
            raise ValueError("Unknwon norm type given (%s)" % self.norm_)

        #NOTE: am returning negative score
        #from sameh he does this to comply with loss objective?
        return -scores

    def forward(self, x):
        lhs = torch.split(self.lhs(x[:, 0]), 30)
        rel = torch.split(self.rel(x[:, 1]), 30)
        rhs = torch.split(self.rhs(x[:, 2]), 30)

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

        for l, rl, rh in zip(rhs, rel, rhs):
            # interactions_sp = (l + rl)[:,None] - self.rhs.weight
            scores_sp_tmp = torch.norm((l + rl)[:,None] - self.rhs.weight, norm, dim=2)
            print(scores_sp_shape)

            # print(torch.cuda.memory_allocated())
            # scores_sp_tmp = torch.norm(interactions_sp, norm, dim=2)

            # del interactions_sp
            # torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated())


            # interactions_po = (self.lhs.weight + rl[:,None]) - rh[:,None]
            scores_po_tmp = torch.norm((self.lhs.weight + rl[:,None]) - rh[:,None], norm, dim=2)
            # scores_po_tmp = torch.norm(interactions_po, norm, dim=2)
            # del interactions_po
            # torch.cuda.empty_cache()

            #should take the norm across each row of matrix

            if scores_po is None:
                scores_po = scores_po_tmp
                scores_sp = scores_sp_tmp
            else:
                scores_po = torch.cat((scores_po, scores_po_tmp), 0)
                scores_sp = torch.cat((scores_sp, scores_sp_tmp), 0)
            del scores_sp_tmp
            del scrose_po_tmp
            torch.cuda.empty_cache()

            print(scores_sp.shape)
            print(scores_po.shape)

        return -scores_sp, -scores_po, (lhs, rel, rhs)


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        """
        Get the chunk of the target vars
        """
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data + self.rel(queries[:, 1]).data




class ComplEx_MC(KBCModelMCL):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3):
        super(ComplEx_MC, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size


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
