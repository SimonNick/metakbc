import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np

import torch

from metakbc.clause import LearnedClause
from metakbc.models import BaseModel

def visualize_clause(clause: LearnedClause):
    fig = plt.figure()
    for i in range(clause.n_relations):
        softmax_values = torch.softmax(clause.weights[i], dim=1).cpu().detach().numpy()
        ax = fig.add_subplot(1, clause.n_relations, i+1)
        im = ax.matshow(softmax_values, cmap=matplotlib.cm.Greys_r, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(softmax_values):
            ax.text(j, i, '{:0.2f}'.format(v), ha='center', va='center')
    return fig

def visualize_embeddings(model: BaseModel):
    """Draw Hinton diagram for visualizing a weight matrix."""

    fig, axes = plt.subplots(1, 2, figsize=(30,30))

    for ax, matrix in zip(axes, [model.emb_so.cpu().detach().numpy(), model.emb_p.cpu().detach().numpy()]):

        if matrix.shape[0] > 100:
            matrix = matrix[:100]

        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()

    return fig