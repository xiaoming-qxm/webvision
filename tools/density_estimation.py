# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Density estimation on webvision dataset."""

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
from os.path import join as pjoin
import os
import cPickle as pickle


def KDE(X, img_list, save_path, cls_id):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    kd_vals = kde.score_samples(X)
    kd_vals = np.exp(kd_vals)
    cut_off = np.percentile(kd_vals, 50)
    labels = kd_vals > cut_off
    labels = labels.astype(np.int)

    with open(pjoin(save_path, cls_id + '.lst'), 'wb') as f:
        for i in range(len(img_list)):
            f.write("{} {} {}\n".format(
                    img_list[i], labels[i], labels[i]))


def main(base_path, save_path):
    cls_id = '3'
    with open(pjoin(base_path, cls_id, '0.pkl'), 'rb') as f:
        X = pickle.load(f)
    with open(pjoin(base_path, cls_id, 'img_name.lst'), 'rb') as f:
        img_list = f.readlines()
    img_list = [i.strip().split('/')[-1] for i in img_list]

    X = PCA(n_components=20).fit_transform(X)

    KDE(X, img_list, save_path, cls_id)


if __name__ == "__main__":
    base_path = "../data/img_feat"
    save_path = "../data/dens_est/"
    main(base_path, save_path)
