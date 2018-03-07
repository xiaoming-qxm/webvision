# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cPickle as pickle

import shutil
from os.path import join as pjoin
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def pca(X=np.array([]), no_dims=50):

    # print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def cluster_algo(X, algo_name='kmeans', n_clusters=2):
    if algo_name == "kmeans":
        model = KMeans(n_clusters=n_clusters,
                       random_state=0).fit(X)
        labels = model.labels_
        centers = model.cluster_centers_

        return labels, n_clusters, centers
    elif algo_name == "dbscan":
        # X = StandardScaler().fit_transform(X)
        # print(X)

        # model = DBSCAN(eps=5, min_samples=80).fit(X)
        model = DBSCAN(eps=5, min_samples=80).fit(X)
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return labels + 1, n_clusters, None
    elif algo_name == "agglom":
        model = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        labels = model.labels_

        return labels, n_clusters, None

    else:
        return None


def grid_search_dbscan(X):
    min_value = np.inf
    max_value = -np.inf

    for i in range(0, X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            dist = np.linalg.norm(X[i, :] - X[j, :])
            if dist < min_value:
                min_value = dist
            if dist > max_value:
                max_value = dist

    eps_list = np.linspace(min_value + 1e-3, max_value,
                           int(max_value - min_value))
    num_spl_list = np.linspace(10, 200, 20)

    for eps in eps_list:
        for num in num_spl_list:
            model = DBSCAN(eps=eps, min_samples=num).fit(X)
            labels = model.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print("{} {} {}".format(eps, num, n_clusters))


def clustering_main(feat_file, im_name_file,
                    save_path, cls_id):
    with open(feat_file, 'r') as f:
        X = pickle.load(f)
    with open(im_name_file, 'r') as f:
        image_list = f.readlines()

    image_list = [l.strip('\n').split('/')[-1] for l in image_list]

    reduced_dims = 100

    # PCA dimension reduction
    X = pca(X, reduced_dims).real

    n_clusters = 5

    labels, n_clusters, _ = cluster_algo(X, algo_name='agglom',
                                         n_clusters=n_clusters)

    # grid_search_dbscan(X)

    print(set(labels))

    with open(pjoin(save_path, cls_id + '.lst'), 'wb') as f:
        for i in range(len(image_list)):
            f.write("{} {} {}\n".format(
                image_list[i], labels[i], labels[i]))


if __name__ == "__main__":
    cls_id = 26
    img_name_file = "../data/img_feat/{}/img_name.lst".format(cls_id)
    feature_file = "../data/img_feat/{}/0.pkl".format(cls_id)
    save_path = "../data/k-means"

    clustering_main(feature_file, img_name_file,
                    save_path, str(cls_id))
