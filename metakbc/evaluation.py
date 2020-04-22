import torch
from torch import Tensor, LongTensor

from typing import Tuple, List, Optional
from collections import defaultdict

from metakbc.datasets import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(scores: Tensor, true_idx: LongTensor, masks: Optional[Tensor] = None) -> dict:

    if not masks == None: scores[masks] = -float('inf')
    argsort_score = torch.argsort(scores, dim=1, descending=True)
    onehot_match = (argsort_score == true_idx.view(-1, 1)).float()
    
    # mean reciprocal rank
    ranks = torch.argmax(onehot_match, dim=1) + 1
    mrr = torch.mean(1 / ranks.float()).item()

    # hits score
    hits_1  = torch.mean(torch.sum(onehot_match[:, :1], dim=1)).item()
    hits_3  = torch.mean(torch.sum(onehot_match[:, :3], dim=1)).item()
    hits_5  = torch.mean(torch.sum(onehot_match[:, :5], dim=1)).item()
    hits_10 = torch.mean(torch.sum(onehot_match[:, :10], dim=1)).item()

    return {"MRR":     mrr, 
            "HITS@1":  hits_3,
            "HITS@3":  hits_3,
            "HITS@5":  hits_5,
            "HITS@10": hits_10}


def add_averages(average_a: float, size_a: int, average_b: float, size_b: int) -> float:
    return (average_a * size_a + average_b * size_b) / (size_a + size_b)


def evaluate(dataset: Dataset,
             splits: List[str],
             model,
             batch_size: int,
             filters: Optional[dict] = None) -> dict:

    metrics_dict = {s: {m: 0 for m in ["MRR", "HITS@1", "HITS@3", "HITS@5", "HITS@10"]} for s in splits}

    for s in splits:

        n_samples = 0
        for batch in dataset.get_batches(s, batch_size, shuffle=False):
        
            batch = batch.to(device)
            n_batch_samples = batch.shape[0]
            s_idx = batch[:, 0]
            p_idx = batch[:, 1]
            o_idx = batch[:, 2]

            with torch.no_grad():
                score_s = model.score_subjects(p_idx, o_idx)
                score_o = model.score_objects(s_idx, p_idx)
            scores = torch.cat((score_s, score_o))
            idx = torch.cat((s_idx, o_idx))

            if not filters == None:
                s_masks = torch.zeros(n_batch_samples, dataset.n_entities, dtype=torch.bool)
                o_masks = torch.zeros(n_batch_samples, dataset.n_entities, dtype=torch.bool)
                for i in range(n_batch_samples):
                    for hide_s in filters['s'][(p_idx[i].item(), o_idx[i].item())]:
                        s_masks[i,hide_s] = True
                    for hide_o in filters['o'][(s_idx[i].item(), p_idx[i].item())]:
                        o_masks[i,hide_o] = True
                    s_masks[i,s_idx[i].item()] = False
                    o_masks[i,o_idx[i].item()] = False
                masks = torch.cat((s_masks, o_masks))
            else:
                masks = None

            batch_metrics = metrics(scores, idx, masks)
            for k,v in metrics_dict[s].items():
                metrics_dict[s][k] = add_averages(v, n_samples, batch_metrics[k], n_batch_samples)
            n_samples += n_batch_samples

    return metrics_dict


def build_filters(dataset: Dataset) -> dict:
    filters = {'s': defaultdict(set), 'o': defaultdict(set)}
    for split in dataset.splits:
        for e in dataset.get_examples(split):
            s, p, o = e[0].item(), e[1].item(), e[2].item()
            filters['s'][(p,o)].add(s)
            filters['o'][(s,p)].add(o)
    return filters