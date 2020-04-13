# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize

from metakbc.models import DistMult
from metakbc.datasets import Dataset
from metakbc.evaluation import evaluate
from metakbc.regularizers import InconsistencyLoss

from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn(dataset: Dataset,
          n_epochs: int,
          n_valid: int,
          rank: int,
          batch_size: int) -> None:

    l = 1e-1

    model = DistMult(size=dataset.get_shape(), rank=4).to(device)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    reg = InconsistencyLoss()

    for epoch in range(n_epochs):

        #print("\rEpoch #{}".format(epoch+1), end="")

        # ==========================================
        # TRAINING
        # ==========================================
        for batch in dataset.get_batches("train", batch_size, shuffle=True):

            batch = batch.to(device)
            s_idx = batch[:, 0]
            p_idx = batch[:, 1]
            o_idx = batch[:, 2]

            score_s = model.score_subjects(p_idx, o_idx)
            score_o = model.score_objects(s_idx, p_idx)
            loss_fact = loss(score_s, s_idx) + loss(score_o, o_idx)
            loss_inc = reg([model.emb_p])
            loss_total = loss_fact + l * loss_inc

            print("Loss", loss_total.item())

            optim.zero_grad() 
            loss_total.backward() 
            optim.step()

            with torch.no_grad():
                model.emb_p /= model.emb_p.norm(dim=1, p=2).view(-1, 1)
                model.emb_so /= model.emb_so.norm(dim=1, p=2).view(-1, 1)

        # ==========================================
        # EVALUATION
        # ==========================================
        if epoch % n_valid == 0:
            print("Metrics", evaluate(dataset, ["train", "valid"], model, batch_size))