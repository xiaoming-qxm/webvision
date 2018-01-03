#-*- coding: utf-8 -*-

import os
import shutil
import random
from os.path import join as pjoin
import matplotlib.pyplot as plt


def create_tiny_val_test_data(num_val_set=25):
    random.seed(10)
    src_path = "/data/webvision/val_images"
    val_dst_path = "/data/wv-40/valid"
    test_dst_path = "/data/wv-40/test"

    cls_lst = os.listdir("/data/wv-40/train")

    for cls_idx in cls_lst:
        cls_path = pjoin(src_path, cls_idx)
        imgs = os.listdir(cls_path)
        idx = random.sample(xrange(len(imgs)), num_val_set)
        val_imgs = [imgs[i] for i in idx]
        test_imgs = [imgs[i] for i in xrange(len(imgs)) if i not in idx]

        for im in val_imgs:
            dst_path = pjoin(val_dst_path, cls_idx)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)

            shutil.copyfile(pjoin(cls_path, im),
                            pjoin(dst_path, im))

        for im in test_imgs:
            dst_path = pjoin(test_dst_path, cls_idx)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)

            shutil.copyfile(pjoin(cls_path, im),
                            pjoin(dst_path, im))


def create_tiny_train_data():
    random.seed(10)

    path = "/data/webvision/trn_images"
    stats = {}

    for cls_idx in os.listdir(path):
        stats[int(cls_idx)] = len(os.listdir(pjoin(path, cls_idx)))

    arr = sorted(stats.items(), key=lambda item: item[1], reverse=True)

    X = [idx[0] for idx in arr]
    Y = [idx[1] for idx in arr]

    # # fig = plt.figure()
    # plt.bar(X[:10], Y[:10])
    # # plt.bar(range(len(Y)), Y, color='black')
    # plt.xlim(0, 1000)
    # plt.xlabel('class index')
    # plt.ylabel('number of images')
    # plt.title('The statistics of webvision dataset')
    # plt.show()

    new_path = "/data/wv-40/train"
    # create new dataset
    tiny_data = random.sample(arr[300:800], 40)

    with open("/data/webvision/info/synsets.txt", 'rb') as f:
        concepts = f.readlines()
    concepts = [c[10:] for c in concepts]

    # print concepts

    # print tiny_data

    tiny_label = []
    for item in tiny_data:
        tiny_label.append(item[0])
        shutil.copytree(pjoin(path, str(item[0])),
                        pjoin(new_path, str(item[0])))

    tiny_label = sorted(tiny_label)

    # print tiny_label_map
    with open("/data/wv-40/label_map.txt", 'wb') as f:
        for lbl in tiny_label:
            f.write("{} {}".format(lbl, concepts[lbl]))


create_tiny_val_test_data()
