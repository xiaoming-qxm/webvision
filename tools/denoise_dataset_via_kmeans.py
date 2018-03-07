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
from sklearn.cluster import KMeans


def pca(X=np.array([]), no_dims=50):

    # print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


# 计算类内标准差
def calc_std(x):
    mean = np.mean(x, axis=0, dtype=np.float64)
    diff = np.mean((x - mean) ** 2, axis=0, dtype=np.float64)

    return np.mean(np.sqrt(diff))


def kmeans_cluster_and_denoise_dataset(feat_file, im_name_file,
                                       save_path, cls_id):
    with open(feat_file, 'r') as f:
        X = pickle.load(f)
    with open(im_name_file, 'rb') as f:
        image_list = f.readlines()
    image_list = [l.strip('\n').split('/')[-1] for l in image_list]

    reduced_dims = 400
    # PCA dimension reduction
    X = pca(X, reduced_dims).real
    # k-means clustering

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    c0_idx = list(np.where(labels == 0)[0])
    c1_idx = list(np.where(labels == 1)[0])
    x0 = X[c0_idx, :]
    x1 = X[c1_idx, :]
    s0 = calc_std(x0)
    s1 = calc_std(x1)

    with open(pjoin(save_path, cls_id + '.lst'), "wb") as f:
        if s0 < s1:
            for i in range(len(image_list)):
                f.write("{} {} {}\n".format(
                    image_list[i], 1 - labels[i], 1 - labels[i]))
        else:
            for i in range(len(image_list)):
                f.write("{} {} {}\n".format(
                    image_list[i], labels[i], labels[i]))


def main():
    cls_id = 14
    img_name_file = "../data/img_feat/{}/img_name.lst".format(cls_id)
    feature_file = "../data/img_feat/{}/0.pkl".format(cls_id)
    save_path = "../data/k-means"

    kmeans_cluster_and_denoise_dataset(feature_file, img_name_file,
                                       save_path, str(cls_id))

main()
