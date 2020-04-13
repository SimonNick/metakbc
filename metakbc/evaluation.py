import torch
from torch import Tensor, LongTensor

from typing import Tuple, List

from metakbc.datasets import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(scores: Tensor, true_idx: LongTensor) -> dict:

    argsort_score = torch.argsort(scores, dim=1, descending=True)
    onehot_match = (argsort_score == true_idx.view(-1, 1)).float()
    
    # mean reciprocal rank
    ranks = torch.argmax(onehot_match, dim=1) + 1
    mrr = torch.mean(1 / ranks.float()).item()

    # hits score
    hits_3  = torch.mean(torch.sum(onehot_match[ :3], dim=1)).item()
    hits_5  = torch.mean(torch.sum(onehot_match[ :5], dim=1)).item()
    hits_10 = torch.mean(torch.sum(onehot_match[:10], dim=1)).item()

    return {"MRR":     mrr, 
            "HITS@3":  hits_3,
            "HITS@5":  hits_5,
            "HITS@10": hits_10}


def add_averages(average_a: float, size_a: int, average_b: float, size_b: int) -> float:
    return (average_a * size_a + average_b * size_b) / (size_a + size_b)


def evaluate(dataset: Dataset,
         splits: List[str],
         model,
         batch_size: int) -> dict:

    metrics_dict = {s: {m: 0 for m in ["MRR", "HITS@3", "HITS@5", "HITS@10"]} for s in splits}

    for split in splits:

        n_samples = 0
        for batch in dataset.get_batches(split, batch_size, shuffle=False):
        
            batch = batch.to(device)
            s_idx = batch[:, 0]
            p_idx = batch[:, 1]
            o_idx = batch[:, 2]

            with torch.no_grad():
                score_s = model.score_subjects(p_idx, o_idx)
                score_o = model.score_objects(s_idx, p_idx)
            scores = torch.cat((score_s, score_o))
            idx = torch.cat((s_idx, o_idx))

            batch_metrics = metrics(scores, idx)
            n_batch_samples = batch.shape[0]
            for k,v in metrics_dict[split].items():
                metrics_dict[split][k] = add_averages(v, n_samples, batch_metrics[k], n_batch_samples)
            n_samples += n_batch_samples

    return metrics_dict