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
    plt.bar(index, stats, 0.8, color='black', edgecolor="black")
    # plt.plot([(i + 0.4) for i in index], stats, 'b-')
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


def parse_log(log_file):
    with open(log_file, 'rb') as f:
        infos = f.readlines()

    model_name = infos[0].strip('\n').split(": ")[1]
    batch_size = infos[1].strip('\n').split(": ")[1]
    epochs = infos[2].strip('\n').split(": ")[1]
    best_epoch = infos[3].strip('\n').split(": ")[1]
    save_name = log_file.split('/')[-1].split('.')[0]

    train_loss = []
    train_acc = []
    train_steps = []

    ln_i = 5
    while infos[ln_i].startswith('valid') is False:
        _, step, loss, acc = infos[ln_i].strip('\n').split(' ')
        train_loss.append(float(loss))
        train_acc.append(float(acc))
        train_steps.append(int(step))
        ln_i = ln_i + 1

    valid_loss = []
    valid_acc = []
    valid_epoch = []

    for i in xrange(ln_i + 1, len(infos)):
        epoch, loss, acc = infos[i].strip('\n').split(' ')
        valid_loss.append(float(loss))
        valid_acc.append(float(acc))
        valid_epoch.append(int(epoch))

    record_per_epoch = len(train_steps) / max(valid_epoch)

    for i in xrange(len(train_steps)):
        train_steps[i] = record_per_epoch * train_steps[0] * \
            (i // record_per_epoch) + train_steps[i]

    valid_epoch = [e * record_per_epoch * train_steps[0] for e in valid_epoch]

    train_steps = [t / 1000 for t in train_steps]
    valid_epoch = [v / 1000 for v in valid_epoch]

    return train_steps, train_acc, train_loss, \
        valid_epoch, valid_acc, valid_loss, \
        model_name, save_name


def plot_pair_training_hist(fname1, fname2, save_name):
    (train_steps, train_acc_1,
     train_loss_1, valid_epoch,
     valid_acc_1, valid_loss_1,
     model_name_1, _) = parse_log(fname1)

    (train_steps, train_acc_2,
     train_loss_2, valid_epoch,
     valid_acc_2, valid_loss_2,
     model_name_2, _) = parse_log(fname2)

    plt.figure(figsize=(14, 5))

    p1 = plt.subplot(1, 2, 1)
    p1.plot(train_steps, train_loss_1, 'k-',
            label='train loss',
            lw=1.2)
    p1.plot(valid_epoch, valid_loss_1, 'k--',
            label='valid loss',
            lw=1.2)

    p1.plot(train_steps, train_loss_2, 'k-',
            label='train loss',
            lw=1.2)
    p1.plot(valid_epoch, valid_loss_2, 'k--',
            label='valid loss',
            lw=1.2)

    p1.set_xlabel('training steps (k)')
    p1.set_ylabel('loss')
    p1.set_title(" loss")
    p1.legend(loc='upper right')

    p2 = plt.subplot(1, 2, 2)
    p2.plot(train_steps, train_acc_1, 'k-',
            label='train acc',
            lw=1.2)
    p2.plot(valid_epoch, valid_acc_1, 'k--',
            label='valid acc',
            lw=1.2)

    p2.plot(train_steps, train_acc_1, 'k-',
            label='train acc',
            lw=1.2)
    p2.plot(valid_epoch, valid_acc_1, 'k--',
            label='valid acc',
            lw=1.2)

    p2.set_xlabel('training steps (k)')
    p2.set_ylabel('accuracy')
    p2.set_title(" accuracy")
    p2.legend(loc='upper left')
    plt.show()


def plot_single_training_hist(fname):

    (train_steps, train_acc,
        train_loss, valid_epoch,
        valid_acc, valid_loss,
        model_name, save_name) = parse_log(fname)

    plt.figure(figsize=(14, 5))

    p1 = plt.subplot(1, 2, 1)
    p1.plot(train_steps, train_loss, 'r-',
            label='train loss',
            lw=1.2)
    p1.plot(valid_epoch, valid_loss, 'c--',
            label='valid loss',
            lw=1.2)

    p1.set_xlabel('training steps (k)')
    p1.set_ylabel('loss')
    p1.set_title(model_name + " loss")
    p1.legend(loc='upper right')

    p2 = plt.subplot(1, 2, 2)
    p2.plot(train_steps, train_acc, 'r-',
            label='train acc',
            lw=1.2)
    p2.plot(valid_epoch, valid_acc, 'c--',
            label='valid acc',
            lw=1.2)

    p2.set_xlabel('training steps (k)')
    p2.set_ylabel('accuracy')
    p2.set_title(model_name + " accuracy")
    p2.legend(loc='upper left')

    plt.savefig('../results/' + model_name +
                "/{}.png".format(
                    save_name))


if __name__ == "__main__":
    # plot_clean_nosiy_examples()
    # plot_data_stats()
    plot_single_training_hist("../logs/inception_v3_q10_v2.log")

    # plot_pair_training_hist("../logs/resnet50_a.log",
    #                         "../logs/inception_v3_a.log", "test.jpg")
