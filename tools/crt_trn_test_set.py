#-*- coding: utf-8 -*-

""" create train and test dataset. """

import os
import shutil
from os.path import join as pjoin


lm_file = "/data/wv-40/label_map.txt"
src_trn_path = "/data/webvision/trn_images"
src_tst_path = "/data/webvision/val_images"

dst_trn_path = "/data/wv-40/train"
dst_tst_path = "/data/wv-40/test"

valid_file = "/data/wv-40/valid_list.txt"

lmap = []

with open(lm_file, 'rb') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip().split(' ')
    lmap.append((line[1], line[0]))


val_set = [set([]) for _ in range(40)]

with open(valid_file, 'rb') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip().split(' ')
    val_set[int(line[1])].add(line[0])

trn_set = []
tst_set = []

for tpl in lmap:
    cls_src_trn_fld = pjoin(src_trn_path, tpl[0])
    cls_src_tst_fld = pjoin(src_tst_path, tpl[0])

    cls_dst_trn_fld = pjoin(dst_trn_path, tpl[1])
    cls_dst_tst_fld = pjoin(dst_tst_path, tpl[1])

    if not os.path.exists(cls_dst_trn_fld):
        os.mkdir(cls_dst_trn_fld)

    if not os.path.exists(cls_dst_tst_fld):
        os.mkdir(cls_dst_tst_fld)

    for img in os.listdir(cls_src_trn_fld):
        if img not in val_set[int(tpl[1])]:
            shutil.copyfile(pjoin(cls_src_trn_fld, img),
                            pjoin(cls_dst_trn_fld, img))

    for img in os.listdir(cls_src_tst_fld):
        shutil.copyfile(pjoin(cls_src_tst_fld, img),
                        pjoin(cls_dst_tst_fld, img))
