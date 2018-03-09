# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create denoised dataset by clustering. """

from os.path import join as pjoin
import os
import random


def denoised_main(cls_id):
    probs_root = "../data/pred_probs"
    cluster_root = "../data/k-means"
    noisy_root = "../data/noisy"
    save_root = "../data/denoised"
    lbl_map_file = "../data/labels_queries_map.txt"

    with open(lbl_map_file, 'r') as f:
        lines = f.readlines()

    lbl_map = {}
    for l in lines:
        idx, name = l.split(' ')[0:2]
        lbl_map[str(idx)] = name

    # For data cleaner noisy data
    with open(pjoin(noisy_root, lbl_map[cls_id] + '.txt'), 'r') as f:
        lines = f.readlines()

    noisy_list = set([l.strip('\n') + ".jpg" for l in lines])

    name_list = []

    with open(pjoin(probs_root, cls_id + '.lst'), 'r') as f:
        lines = f.readlines()

    name_list = [l.split(' ')[0] for l in lines]
    probs_list = [float(l.split(' ')[1]) for l in lines]

    with open(pjoin(cluster_root, cls_id + '.lst'), 'r') as f:
        lines = f.readlines()

    cluster_list = [int(l.split(' ')[1]) for l in lines]

    clean_labels = set([1, 2])

    new_arr = []

    counter_clean = 0
    counter_noisy = 0

    for i in range(len(name_list)):
        if cluster_list[i] in clean_labels \
                and probs_list[i] >= 0.5:
                # and (name_list[i] not in noisy_list):
            new_arr.append((name_list[i], 1))
            counter_clean += 1
        else:
            new_arr.append((name_list[i], 0))
            counter_noisy += 1

    print(len(name_list))
    print(counter_clean)
    print(counter_noisy)

    with open(pjoin(save_root, cls_id + '.lst'), 'wb') as f:
        for i in range(len(new_arr)):
            f.write("{} {} {}\n".format(
                new_arr[i][0], new_arr[i][1], new_arr[i][1]))


def over_sampling(data_list, target_num):
    random.seed(0)
    dst_list = []
    for i in range(target_num - len(data_list)):
        dst_list.append(random.choice(data_list))

    return dst_list


def create_final_dataset():
    src_train_f = "../data/train_q10.lst"
    dst_train_f = "../data/train_q10_denos.lst"

    denos_root = "../data/denoised"

    arr = [[] for _ in range(40)]

    with open(src_train_f, 'r') as f:
        lines = f.readlines()

    for l in lines:
        idx, name = l.strip('\n').split(' ')
        arr[int(idx)].append(name)

    over_sam_fld = "../data/over_samp"

    for fname in os.listdir(denos_root):
        cls_id = int(fname.split('.')[0])

        with open(pjoin(denos_root, fname), 'r') as f:
            lines = f.readlines()

        repl = []

        for l in lines:
            name, label = l.strip('\n').split(' ')[0:2]
            name = name[:-4]
            label = int(label)
            if label == 1:
                repl.append(name)

        print("{}: old {}".format(cls_id, len(arr[cls_id])))

        over_repl = over_sampling(repl, len(arr[cls_id]))

        print(len(over_repl))

        with open(pjoin(over_sam_fld, fname), 'w') as f:
            for i in range(len(over_repl)):
                f.write("{}.jpg\n".format(over_repl[i]))

        arr[cls_id] = repl

        print("{}: new {}".format(cls_id, len(arr[cls_id])))

    # for i in range(40):
    #     print("{}: {}".format(i, len(arr[i])))

    with open(dst_train_f, 'w') as f:
        for i in xrange(40):
            for j in xrange(len(arr[i])):
                f.write("{} {}\n".format(i, arr[i][j]))


if __name__ == "__main__":
    cls_id = 26
    # denoised_main(str(cls_id))
    create_final_dataset()
