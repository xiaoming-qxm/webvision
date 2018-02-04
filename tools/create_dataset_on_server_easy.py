#-*- coding: utf-8 -*-

import os
import shutil
from os.path import join as pjoin


lm_file = "/data/xiaoming/wv-40/label_map.txt"

src_train_path = "/data/xiaoming/webvision/trn_images"
src_valid_path = "/data/xiaoming/webvision/trn_images"
src_test_path = "/data/xiaoming/webvision/val_images"

dst_train_path = "/data/xiaoming/wv-40/train"
dst_test_path = "/data/xiaoming/wv-40/test"
dst_valid_path = "/data/xiaoming/wv-40/valid"

train_file = "/data/xiaoming/wv-40/train_list.txt"
valid_file = "/data/xiaoming/wv-40/valid_list.txt"
test_file = "/data/xiaoming/wv-40/test_list.txt"

lmap = {}

with open(lm_file, 'rb') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip().split(' ')
    lmap[line[0]] = line[1]


def copy_files(src_path, dst_path, file_name, lmap):
    os.mkdir(dst_path)

    with open(file_name, 'rb') as f:
        lines = f.readlines()

    for line in lines:
        img, dst_cls_id = line.strip().split(' ')
        src_cls_id = lmap[dst_cls_id]

        if not os.path.exists(pjoin(dst_path, dst_cls_id)):
            os.mkdir(pjoin(dst_path, dst_cls_id))

        shutil.copyfile(pjoin(src_path, src_cls_id, img),
                        pjoin(dst_path, dst_cls_id, img))


copy_files(src_train_path, dst_train_path, train_file, lmap)
copy_files(src_test_path, dst_test_path, test_file, lmap)
copy_files(src_valid_path, dst_valid_path, valid_file, lmap)
