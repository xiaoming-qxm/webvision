# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create tiny dataset train list. """

import os
from os.path import join as pjoin

data_root = "/data/wv-40/"

with open(pjoin(data_root, 'train.lst'), 'wb') as f:
    cls_list = os.listdir(pjoin(data_root, 'train'))
    count = 0
    for cls_idx in cls_list:
        img_list = os.listdir(
            pjoin(data_root, 'train', cls_idx))
        for img in img_list:
            f.write("{} {}\n".format(
                    pjoin(data_root, 'train',
                          cls_idx, img), count))
        count += 1
