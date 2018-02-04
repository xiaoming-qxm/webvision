# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create validation dataset. """

import shutil
import os
from os.path import join as pjoin
import random


random.seed(66)


def create_ori_val_set(src_data_root, dst_data_root):
    for cls_id in os.listdir(src_data_root):
        src_fld = pjoin(src_data_root, cls_id)
        dst_fld = pjoin(dst_data_root, cls_id)
        img_lists = os.listdir(src_fld)

        random.shuffle(img_lists)
        if not os.path.exists(dst_fld):
            os.mkdir(dst_fld)

        for img in img_lists[:100]:
            shutil.copyfile(pjoin(src_fld, img),
                            pjoin(dst_fld, img))


def create_final_val_set(src_data_root, dst_data_root):
    for cls_id in os.listdir(src_data_root):
        src_fld = pjoin(src_data_root, cls_id)
        dst_fld = pjoin(dst_data_root, cls_id)
        img_lists = os.listdir(src_fld)

        random.shuffle(img_lists)
        if not os.path.exists(dst_fld):
            os.mkdir(dst_fld)

        for img in img_lists[:50]:
            shutil.copyfile(pjoin(src_fld, img),
                            pjoin(dst_fld, img))


if __name__ == "__main__":
    src_data_root = "/data/wv-40/train"
    dst_data_root_ori = "/data/wv-40/valid_ori"
    dst_data_root_fin = "/data/wv-40/valid"
    # create_ori_val_set(src_data_root, dst_data_root_ori)
    # create_final_val_set(dst_data_root_ori, dst_data_root_fin)
