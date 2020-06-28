import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np

import torch

from metakbc.clause import LearnedClause
from metakbc.models import BaseModel
from metakbc.datasets import Dataset

from sklearn.decomposition import PCA

def visualize_clause(clause: LearnedClause):

    fig = plt.figure(figsize=(30,30))
    for i in range(clause.n_relations):
        softmax_values = torch.softmax(clause.weights[i], dim=1).cpu().detach().numpy()
        ax = fig.add_subplot(1, clause.n_relations, i+1)
        im = ax.matshow(softmax_values, cmap=matplotlib.cm.Greys_r, vmin=0, vmax=1)
        # for (i, j), v in np.ndenumerate(softmax_values):
        #     ax.text(j, i, '{:0.2f}'.format(v), ha='center', va='center')
    return fig

def visualize_embeddings(model: BaseModel):
    """Draw Hinton diagram for visualizing a weight matrix."""

    fig, axes = plt.subplots(2, figsize=(30,30))

    for ax, matrix in zip(axes, [model.emb_so.cpu().detach().numpy(), model.emb_p.cpu().detach().numpy()]):

        if matrix.shape[0] > 100:
            matrix = matrix[:100]

        matrix = matrix.T

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

def PCA_entity_embeddings(model: BaseModel, dataset: Dataset):

    emb_so = model.emb_so.cpu().detach().numpy()

    if emb_so.shape[0] > 100:
        emb_so = emb_so[:100]

    pca = PCA(n_components=2)
    pca.fit(emb_so)
    x_pca = pca.transform(emb_so)

    fig, ax = plt.subplots(figsize=(30, 30))
    ax.plot(x_pca[:, 0], x_pca[:, 1], '+')

    for i, xy in enumerate(x_pca):
        ax.annotate(dataset.entities[i], xy)

    return fig