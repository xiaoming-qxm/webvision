# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

from os.path import join as pjoin


src_q_name = "../data/train_q30.lst"
noisy_root = "../data/noisy"
lqm_fname = "../data/labels_queries_map.txt"

lbl_id_map = {}
with open(lqm_fname, 'r') as f:
    lines = f.readlines()

# store filename by label index
arr = [[] for _ in range(len(lines))]

for line in lines:
    idx, lbl_name, _ = line.split(" ")
    lbl_id_map[int(idx)] = lbl_name


with open(src_q_name, 'r') as f:
    contents = f.readlines()

for cnt in contents:
    cls_id, img = cnt.strip('\n').split(' ')
    arr[int(cls_id)].append(img)

for i in range(40):
    with open(pjoin(noisy_root, lbl_id_map[i] + '.txt'), 'r') as f:
        lines = f.readlines()
    nos_im = [l.strip('\n') for l in lines]
    print len(nos_im)
