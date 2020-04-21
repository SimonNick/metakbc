import pytest
import torch

from metakbc.evaluation import metrics

def test_metrics():

    scores = torch.Tensor([[10, 5, -5]])
    true_idx = torch.Tensor([1])

    metrics_dict = metrics(scores, true_idx)

    assert metrics_dict['MRR'] == 0.5
    assert metrics_dict['HITS@3'] == 1
    assert metrics_dict['HITS@5'] == 1
    assert metrics_dict['HITS@10'] == 1

    scores = torch.Tensor([[10, 5, -5]])
    true_idx = torch.Tensor([0])

    metrics_dict = metrics(scores, true_idx)

    assert metrics_dict['MRR'] == 1
    assert metrics_dict['HITS@3'] == 1
    assert metrics_dict['HITS@5'] == 1
    assert metrics_dict['HITS@10'] == 1

    scores = torch.Tensor([[10, 5, -5]])
    true_idx = torch.Tensor([2])

    metrics_dict = metrics(scores, true_idx)

    assert metrics_dict['MRR'] == pytest.approx(1/3)
    assert metrics_dict['HITS@3'] == 1
    assert metrics_dict['HITS@5'] == 1
    assert metrics_dict['HITS@10'] == 1

    # 12 predictions
    scores = torch.Tensor([[10, 5, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [10, 5, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    true_idx = torch.Tensor([1, 0])

    metrics_dict = metrics(scores, true_idx)

    assert metrics_dict['MRR'] == 0.75
    assert metrics_dict['HITS@3'] == 1
    assert metrics_dict['HITS@5'] == 1
    assert metrics_dict['HITS@10'] == 1

    # 12 predictions
    scores = torch.Tensor([[10, 5, -5, 0, 4, 3, 1, 0, 0, 0, 0, -10], # 5th
                           [10, 5, -5, 0, 0, 0, 1, 0, 0, 0, 0, -10], # 3rd
                           [10, 5, -4, 0, 0, 0, -5, 0, 0, -10, -10, -10], # 9th
                           [10, 5, -5, 0, 0, 0, 0, 0, 0, 0, 0, -10]]) # 12th
    true_idx = torch.Tensor([6, 6, 6, 11])

    metrics_dict = metrics(scores, true_idx)

    assert metrics_dict['MRR'] == pytest.approx((1/5 + 1/3 + 1/9 + 1/12)/4)
    assert metrics_dict['HITS@3'] == 0.25
    assert metrics_dict['HITS@5'] == 0.5
    assert metrics_dict['HITS@10'] == 0.75