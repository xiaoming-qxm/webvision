# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create dataset via list. """

import shutil
import os
from os.path import join as pjoin


def create_dataset_via_list(src_data_root,
                            dst_data_root,
                            list_file):
    with open(list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        lbl, name = line.strip('\n').split(' ')
        if not os.path.exists(pjoin(dst_data_root, lbl)):
            os.mkdir(pjoin(dst_data_root, lbl))
        shutil.copyfile(pjoin(src_data_root, lbl, name + '.jpg'),
                        pjoin(dst_data_root, lbl, name + '.jpg'))


if __name__ == "__main__":
    src_data_root = "/data/wv-40/train"
    dst_data_root = "/data/wv-clean/train"
    train_list = "../data/train.lst"

    create_dataset_via_list(src_data_root,
                            dst_data_root,
                            train_list)
