# -*- coding: utf-8 -*-

import torch
from torch import LongTensor, Tensor
from torch.nn import Parameter

from abc import ABC, abstractmethod

from typing import Tuple, Optional, List


class BaseModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def score(self,
              s_idx: LongTensor,
              p_idx: LongTensor,
              o_idx: LongTensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def score_subjects(self,
                       p_idx: LongTensor,
                       o_idx: LongTensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def score_objects(self,
                       s_idx: LongTensor,
                       p_idx: LongTensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError



class DistMult(BaseModel):

    def __init__(self, size: Tuple[int, int, int], rank: int) -> None:
        super().__init__()

        self.emb_so = Parameter(torch.empty((size[0], rank)).normal_())
        self.emb_p = Parameter(torch.empty((size[1], rank)).normal_())
        self.rank = rank

    def _scoring_func(self,
                      s: Tensor,
                      p: Tensor,
                      o: Tensor) -> Tensor:
        return torch.sum(s * p * o)

    def score(self,
              s_idx: LongTensor,
              p_idx: LongTensor,
              o_idx: LongTensor) -> Tensor:

        s, p, o = self.emb_so[s_idx], self.emb_p[p_idx], self.emb_so[o_idx]
        return torch.sum(s * p * o, 1)

    def score_subjects(self,
                       p_idx: LongTensor,
                       o_idx: LongTensor) -> Tensor:

        p, o = self.emb_p[p_idx], self.emb_so[o_idx]
        return (p * o) @ self.emb_so.t()

    def score_objects(self,
                       s_idx: LongTensor,
                       p_idx: LongTensor) -> Tensor:

        s, p = self.emb_so[s_idx], self.emb_p[p_idx]
        return (p * s) @ self.emb_so.t()

    def factors(self,
                s_idx: LongTensor,
                p_idx: LongTensor,
                o_idx: LongTensor) -> Tensor:

        return self.emb_so[s_idx], self.emb_p[p_idx], self.emb_so[o_idx]

    def forward(self, *args):
        return self.score(*args)



class ComplEx(BaseModel):

    def __init__(self, size: Tuple[int, int, int], rank: int) -> None:
        super().__init__()

        self.emb_so = Parameter(torch.empty((size[0], 2*rank)).normal_(0, 1e-3))
        self.emb_p = Parameter(torch.empty((size[1], 2*rank)).normal_(0, 1e-3))
        self.rank = 2*rank

    def _scoring_func(self,
                      s: Tensor,
                      p: Tensor,
                      o: Tensor) -> Tensor:


        p_real, p_img = p[:, :self.rank // 2], p[:, self.rank // 2:]
        s_real, s_img = s[:, :self.rank // 2], s[:, self.rank // 2:]
        o_real, o_img = o[:, :self.rank // 2], o[:, self.rank // 2:]

        score1 = torch.sum(p_real * s_real * o_real, dim=1)
        score2 = torch.sum(p_real * s_img * o_img, dim=1)
        score3 = torch.sum(p_img * s_real * o_img, dim=1)
        score4 = torch.sum(p_img * s_img * o_real, dim=1)

        res = score1 + score2 + score3 - score4

        return res 

    def score(self,
              s_idx: LongTensor,
              p_idx: LongTensor,
              o_idx: LongTensor) -> Tensor:

        s, p, o = self.emb_so[s_idx], self.emb_p[p_idx], self.emb_so[o_idx]

        p_real, p_img = p[:, :self.rank // 2], p[:, self.rank // 2:]
        s_real, s_img = s[:, :self.rank // 2], s[:, self.rank // 2:]
        o_real, o_img = o[:, :self.rank // 2], o[:, self.rank // 2:]

        score1 = torch.sum(p_real * s_real * o_real, 1)
        score2 = torch.sum(p_real * s_img * o_img, 1)
        score3 = torch.sum(p_img * s_real * o_img, 1)
        score4 = torch.sum(p_img * s_img * o_real, 1)

        res = score1 + score2 + score3 - score4

        return res

    def score_subjects(self,
                       p_idx: LongTensor,
                       o_idx: LongTensor) -> Tensor:

        p, o = self.emb_p[p_idx], self.emb_so[o_idx]

        p_real, p_img = p[:, :self.rank // 2], p[:, self.rank // 2:]
        o_real, o_img = o[:, :self.rank // 2], o[:, self.rank // 2:]
        emb_real, emb_img = self.emb_so[:, :self.rank // 2], self.emb_so[:, self.rank // 2:]

        score1_po = (p_real * o_real) @ emb_real.t()
        score2_po = (p_real * o_img) @ emb_img.t()
        score3_po = (p_img * o_img) @ emb_real.t()
        score4_po = (p_img * o_real) @ emb_img.t()

        score_po = score1_po + score2_po + score3_po - score4_po

        return score_po

    def score_objects(self,
                       s_idx: LongTensor,
                       p_idx: LongTensor) -> Tensor:

        s, p = self.emb_so[s_idx], self.emb_p[p_idx]

        s_real, s_img = s[:, :self.rank // 2], s[:, self.rank // 2:]
        p_real, p_img = p[:, :self.rank // 2], p[:, self.rank // 2:]
        emb_real, emb_img = self.emb_so[:, :self.rank // 2], self.emb_so[:, self.rank // 2:]

        score1_sp = (p_real * s_real) @ emb_real.t()
        score2_sp = (p_real * s_img) @ emb_img.t()
        score3_sp = (p_img * s_real) @ emb_img.t()
        score4_sp = (p_img * s_img) @ emb_real.t()

        score_sp = score1_sp + score2_sp + score3_sp - score4_sp

        return score_sp

    def factors(self,
                s_idx: LongTensor,
                p_idx: LongTensor,
                o_idx: LongTensor) -> Tensor:

        s, p, o = self.emb_so[s_idx], self.emb_p[p_idx], self.emb_so[o_idx]

        p_real, p_img = p[:, :self.rank // 2], p[:, self.rank // 2:]
        s_real, s_img = s[:, :self.rank // 2], s[:, self.rank // 2:]
        o_real, o_img = o[:, :self.rank // 2], o[:, self.rank // 2:]

        return [torch.sqrt(p_real ** 2 + p_img ** 2), torch.sqrt(s_real ** 2 + s_img ** 2), torch.sqrt(o_real ** 2 + o_img ** 2)]

    def forward(self, *args):
        return self.score(*args)