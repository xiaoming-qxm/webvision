# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin

data_root = "/data/wv-40"


def trans_label():
    with open("/data/webvision/info/synsets.txt", 'rb') as f:
        concepts = f.readlines()
    concepts = [c[10:] for c in concepts]

    tiny_label_set = os.listdir(pjoin(data_root, 'train'))
    tiny_label_set = [int(l) for l in tiny_label_set]
    tiny_label_set = sorted(tiny_label_set)
    print(tiny_label_set)

    with open(pjoin(data_root, 'label_map.txt'), 'wb') as f:
        for i in xrange(len(tiny_label_set)):
            f.write("{} {} {}".format(i, tiny_label_set[
                    i], concepts[int(tiny_label_set[i])]))


def rename_class_folder():
    with open(pjoin(data_root, 'label_map.txt'), 'rb') as f:
        lm = f.readlines()
        for line in lm:
            new_name, old_name = line.split(' ')[:2]
            print(new_name, old_name)
            for data_set in ["train", 'valid', 'test']:
                os.rename(pjoin(data_root, data_set, old_name),
                          pjoin(data_root, data_set, new_name))


rename_class_folder()
