# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Word2vec visualization. """

import _init_paths
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import torchwordemb
import torch


def load_word_vec(model_path):
    vocab, vec = torchwordemb.load_word2vec_bin(model_path)
    return vocab, vec


def main():
    model_path = "../data/GoogleNews-vectors-negative300.bin"
    vocab, vec = load_word_vec(model_path)

    # color list
    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
    # class list
    classes = ['cat', 'dog', 'woman', 'man', 'king', 'queen',
               'ate', 'eat', 'food', 'kitchen', 'music',
               'piano', 'guitar']

    arr = np.asarray([vec[vocab[c]].numpy() for c in classes])

    pca = PCA(n_components=2)
    reduced_vecs = pca.fit_transform(arr)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(reduced_vecs)):
        ax.scatter(reduced_vecs[i, 0], reduced_vecs[i, 1],
                   100, color=colors[i % 7])

        if classes[i] == "cat":
            xy_coord = (reduced_vecs[i, 0] + 0.2,
                        reduced_vecs[i, 1] - 0.2)
        elif classes[i] == "piano" or classes[i] == 'guitar' \
                or classes[i] == 'dog' or classes[i] == "ate":
            xy_coord = (reduced_vecs[i, 0],
                        reduced_vecs[i, 1] - 0.36)
        else:
            xy_coord = (reduced_vecs[i, 0] + 0.1,
                        reduced_vecs[i, 1] - 0.2)

        ax.annotate(classes[i], xy=(reduced_vecs[i, 0], reduced_vecs[i, 1]),
                    fontsize=12,
                    xycoords='data',
                    xytext=xy_coord,
                    textcoords="data",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.2),
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3,rad=.3",
                                    color="0.7",
                                    shrinkB=0))

    # plt.axis([-3, 3, -3, 3])
    # plt.axis('off')
    # plt.show()

    plt.savefig("../results/wv_visu.png")

main()
