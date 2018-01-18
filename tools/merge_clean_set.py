# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Merge clean dataset. """

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

    cls_name = '_'.join(fname.split('_')[0:-1])
    arr[lbl_id_map[cls_name]].extend(contents)

with open("../data/train.lst", 'w') as f:
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            f.write("{} {}".format(i, arr[i][j]))
