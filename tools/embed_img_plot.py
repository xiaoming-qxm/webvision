# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Embedding images according to a two dimension point."""

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def add_border(in_image, border, color=None):
    return ImageOps.expand(in_image, border=border, fill=color)


def image_point_plot(loc_list, image_list,
                     label_file=None, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax = min(loc_list[:, 0]), max(loc_list[:, 0])
    ymin, ymax = min(loc_list[:, 1]), max(loc_list[:, 1])

    with open(label_file, 'rb') as f:
        labels = f.readlines()
    labels = [int(l.strip().split()[-1]) for l in labels]

    markers = ["^", "o", "s", "*", "D"]
    colors = ['red', 'green', 'blue', "gray"]

    # markers = ["o", "^", "s", "*", "D"]
    # colors = ['green', 'red', 'blue', "gray"]

    for i in range(len(loc_list)):
        point = loc_list[i, :]
        ax.plot(point[0], point[1],
                marker=markers[labels[i]],
                markerfacecolor=colors[labels[i]],
                markeredgecolor=colors[labels[i]],
                markersize=8)

    ax.set_xlim(1.1 * xmin, 1.1 * xmax)
    ax.set_ylim(1.1 * ymin, 1.1 * ymax)

    # plt.tight_layout()
    plt.axis('off')
    plt.show()


def embed_image_plot(loc_list, image_list, label_file=None,
                     size=(64, 64), figsize=(20, 20),
                     with_border=False):
    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax = min(loc_list[:, 0]), max(loc_list[:, 0])
    ymin, ymax = min(loc_list[:, 1]), max(loc_list[:, 1])

    if with_border:
        with open(label_file, 'r') as f:
            labels = f.readlines()
        labels = [int(l.strip().split()[-1]) for l in labels]
        # green for True, red for False
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        # border size
        border = 10

        for i in range(len(image_list)):
            img_name = image_list[i]
            im = Image.open(img_name)
            # add border
            im = add_border(im, border, colors[labels[i]])
            im = im.resize(size)
            im = OffsetImage(im)
            im.image.axes = ax
            xy = list(loc_list[i, :])
            ab = AnnotationBbox(im, xy, xycoords='data',
                                boxcoords="offset points", pad=0.)
            ax.add_artist(ab)
    else:
        for i in range(len(image_list)):
            img_name = image_list[i]
            im = Image.open(img_name)
            im = im.resize(size)
            im = OffsetImage(im)
            im.image.axes = ax
            xy = list(loc_list[i, :])
            ab = AnnotationBbox(im, xy, xycoords='data',
                                boxcoords="offset points", pad=0.)
            ax.add_artist(ab)
    ax.set_xlim(2 * xmin, 2 * xmax)
    ax.set_ylim(2 * ymin, 2 * ymax)

    plt.tight_layout()
    plt.show()
    plt.axis("off")
    # plt.savefig("temp.jpg", dpi=1200)
