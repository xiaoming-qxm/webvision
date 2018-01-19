# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def embed_image_plot(loc_list, image_list, size=(64, 64), figsize=(20, 20)):
    fig, ax = plt.subplots(figsize=figsize)

    xmin, xmax = min(loc_list[:, 0]), max(loc_list[:, 0])
    ymin, ymax = min(loc_list[:, 1]), max(loc_list[:, 1])

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


# image_name = "/data/webvision/flickr/q0001/277044371.jpg"

# im = Image.open(image_name)

# im = im.resize((64, 64))

# implot = plt.imshow(im)

# # put a blue dot at (10, 20)
# plt.scatter([10], [20])

# # put a red dot, size 40, at 2 locations:
# plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)

# plt.show()

# fig, ax = plt.figure(figsize=(20, 20))

# fig, ax = plt.subplots(figsize=(20, 20))

# im = OffsetImage(im)

# im.image.axes = ax

# xy = [0.3, 0.55]

# ab = AnnotationBbox(im, xy,
#					 xycoords='data',
#					 boxcoords="offset points",
#					 pad=0.)

# ax.add_artist(ab)

# # plt.axis("off")
# plt.tight_layout()
# plt.show()
