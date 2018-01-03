# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Calculate image set's mean and standard deviation"""

from PIL import Image
import os
from os.path import join as pjoin
import numpy as np


def calc_img_mean_std(data_root, size):
    fld_lst = os.listdir(data_root)
    num_img = 0
    mean_imgs = np.array([0., 0., 0.]).astype(np.float64)
    arrs = []

    for fld in fld_lst:
        cls_path = pjoin(data_root, fld)
        for im_name in os.listdir(cls_path):
            img = Image.open(pjoin(cls_path, im_name))
            img = img.resize(size)
            arr = np.array(img).astype(np.float64) / 255.
            sum_pc = np.sum(arr, axis=0).sum(0)
            mean_pc = sum_pc / (size[0] * size[1])
            mean_imgs += mean_pc

            num_img = num_img + 1

    # average on the whole dataset
    img_mean = mean_imgs / num_img

    img_stds = np.array([0., 0., 0.]).astype(np.float64)

    for fld in fld_lst:
        cls_path = pjoin(data_root, fld)
        for im_name in os.listdir(cls_path):
            img = Image.open(pjoin(cls_path, im_name))
            img = img.resize(size)
            arr = np.array(img).astype(np.float64) / 255.
            arr = np.reshape(arr, (size[0] * size[1], -1))
            arr = np.power((arr - img_mean), 2)
            arr = np.sum(arr, axis=0) / (size[0] * size[1])

            img_stds += arr

    img_std = np.sqrt(img_stds / num_img, dtype=np.float64)

    print("image mean: {}".format(img_mean))
    print("image std: {}".format(img_std))


if __name__ == "__main__":
    data_path = "/data/wv-40/train"
    size = (299, 299)
    calc_img_mean_std(data_path, size)

