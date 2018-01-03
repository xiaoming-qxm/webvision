# -*- coding: utf-8 -*-

import os
import shutil
from os.path import join as pjoin

root_path = "/data/webvision"


def organize_valid_data():
    ori_img_path = pjoin(root_path, 'val_images_256')
    save_path = pjoin(root_path, 'val_images')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(pjoin(root_path, 'info/val_filelist.txt'), 'rb') as f:

        val_list = f.readlines()
        for line in val_list:
            line = line.strip()
            im_name, cls_idx = line.split(' ')

            cls_path = pjoin(save_path, cls_idx)
            if not os.path.exists(cls_path):
                os.mkdir(cls_path)

            shutil.copyfile(pjoin(ori_img_path, im_name),
                            pjoin(cls_path, im_name))


def re_arange_train_data(set_name, save_path):
    fname = pjoin(root_path, 'info/train_filelist_{}.txt'.format(set_name))
    with open(fname, 'rb') as f:
        trn_list = f.readlines()
        for line in trn_list:
            line = line.strip()
            im_name, cls_idx = line.split(' ')
            cls_path = pjoin(save_path, cls_idx)
            if not os.path.exists(cls_path):
                os.mkdir(cls_path)

            shutil.copyfile(pjoin(root_path, im_name),
                            pjoin(cls_path, im_name.split('/')[-1]))


def organize_train_data():

    save_path = pjoin(root_path, 'trn_images')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for set_name in ['flickr', 'google']:
        re_arange_train_data(set_name, save_path)


if __name__ == "__main__":
    organize_train_data()
