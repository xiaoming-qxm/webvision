# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Target dataset (wv-40) visualization. """

import os
import _init_paths
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from os.path import join as pjoin


def add_border(in_image, border, color=None):
    return ImageOps.expand(in_image, border=border, fill=color)


def plot_gallery(image_list, n_row, n_col, with_border=True):
    plt.figure(figsize=(6*n_col, 6*n_row))
    colors = ['green', 'red']
    titles = ['clean', 'noisy']
    for i, image in enumerate(image_list):
        ax = plt.subplot(n_row, n_col, i + 1)
        im = Image.open(image, 'r')
        if with_border:
            im = add_border(im, 6, color=colors[i / n_col])
        if i % n_col == (n_col / 2):
            ax.set_title(titles[i / n_col], fontsize=20)
        im = im.resize((256, 256))
        plt.imshow(im)
        plt.axis("off")
    # plt.suptitle('Tench', fontsize=30)
    plt.savefig("../results/clean_noisy_visu.png")


def plot_clean_nosiy_examples():
    n_row, n_col = 2, 3
    image_list = ["_0YNOoqGft0HJM.jpg", "-9IV_ZUDnVrndM.jpg",
                  "41unEs1GIXGpfM.jpg", "0lv77S5PaW5vlM.jpg",
                  "286877395.jpg", "719569543.jpg"]

    base_path = "/data/wv-40/train/0"
    image_list = [pjoin(base_path, im) for im in image_list]
    plot_gallery(image_list, n_row, n_col)


def plot_data_stats():
    base_path = "/data/wv-40/train"
    stats = [0] * len(os.listdir(base_path))
    for cls_id in os.listdir(base_path):
        cls_dir = pjoin(base_path, cls_id)
        if not os.path.isdir(cls_dir):
            continue
        im_list = [s for s in os.listdir(cls_dir) if
                   s.endswith('.jpg') or s.endswith('.png')]
        stats[int(cls_id)] = len(im_list)

    plt.figure(figsize=(10, 6))
    index = range(len(stats))
    plt.bar(index, stats, 0.8, color='lightblue', edgecolor="lightblue")
    plt.plot([(i + 0.4) for i in index], stats, 'b-')
    # plt.fill_between(index, stats, 0, color='b')
    plt.axis([0, 40, 0, 3000])
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.title("wv-40 dataset")
    # plt.show()
    plt.tight_layout()
    plt.savefig("../results/data_stats.png")


def plot_train_loss_acc(fname):
    from cnn.utils import plot_history
    plot_history(fname)


if __name__ == "__main__":
    # plot_clean_nosiy_examples()
    # plot_data_stats()
    plot_train_loss_acc("../logs/inception_v3.log")
