# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Sort by."""

from os.path import join as pjoin
import numpy as np
import os


def sort_by_prob(base_path, save_path, cls_id, quan=50):
    with open(pjoin(base_path, cls_id + '.lst'), 'rb') as f:
        lines = f.readlines()

    img_names = [i.strip().split(' ')[0].split('.')[0] for i in lines]
    probs = [float(i.strip().split(' ')[1]) for i in lines]
    origin_labels = [int(i.strip().split(' ')[2]) for i in lines]
    probs = np.asarray(probs)
    cut_off = np.percentile(probs, quan)
    # print("num of origin true labels: {}".format(sum(origin_labels)))
    # print("cut off: {}".format(cut_off))
    # print("max: {}".format(np.max(probs)))
    # print("min: {}".format(np.min(probs)))

    labels = probs >= cut_off
    labels = labels.astype(np.int)
    # print(len(labels))
    # print(sum(labels))

    true_set = []
    with open(pjoin(save_path, cls_id + '.lst'), 'wb') as f:
        for i in range(len(img_names)):
            f.write("{} {} {}\n".format(
                    img_names[i], labels[i], labels[i]))
            if labels[i] == 1:
                true_set.append((int(cls_id), img_names[i]))

    return true_set


def sort_main(base_path, save_path, quan=50):
    true_set = []
    for cls_file in os.listdir(base_path):
        if not os.path.isfile(pjoin(base_path, cls_file)):
            continue
        cls_id = cls_file.split(".")[0]
        res_set = sort_by_prob(base_path, save_path, cls_id, quan)
        true_set.extend(res_set)
    true_set = sorted(true_set)
    with open("../data/train_q{}.lst".format(quan), 'wb') as f:
        for i in range(len(true_set)):
            f.write("{} {}\n".format(true_set[i][0], true_set[i][1]))

if __name__ == "__main__":
    base_path = "../data/pred_probs"
    save_path = "../data/dens_est/"
    sort_main(base_path, save_path)
