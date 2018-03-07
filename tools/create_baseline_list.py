# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create clean baseline dataset list"""

import os
import numpy as np
from os.path import join as pjoin

map_file = "../data/labels_queries_map.txt"
file_root = "../data/clean"

lbl_id_map = {}
with open(map_file, 'r') as f:
    lines = f.readlines()

# store filename by label index
arr = [[] for _ in range(len(lines))]

for line in lines:
    idx, lbl_name, _ = line.split(" ")
    lbl_id_map[lbl_name] = int(idx)

fl = os.listdir(file_root)

for fname in fl:
    with open(pjoin(file_root, fname), 'r') as f:
        contents = f.readlines()

    cls_name = fname.split('.')[0]
    arr[lbl_id_map[cls_name]].extend(contents)

new_arr = [[] for _ in range(len(lines))]

path_tst_val = "/data/wv-40/"

for i in range(40):
    cls_id = str(i)
    test_img = os.listdir(pjoin(path_tst_val, 'test', cls_id))
    val_img = os.listdir(pjoin(path_tst_val, 'valid', cls_id))

    cand_img = test_img + val_img
    cand_img = [im.split('.')[0] for im in cand_img]
    cand_img = set(sorted(cand_img))

    for o_im in arr[i]:
        o_im = o_im.strip('\n')
        if o_im not in cand_img:
            new_arr[i].append(o_im)


with open("../data/train_baseline.lst", 'w') as f:
    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            f.write("{} {}\n".format(i, new_arr[i][j]))
