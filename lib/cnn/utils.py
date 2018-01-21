# -*- coding: utf-8 -*-

"""Utility function for train(test) a neural network"""

import torch
import shutil
import os
from os.path import join


def save_ckpt(state, arch, epoch, is_best=False):
    fname = join('../results', arch, 'ckpt.tar')

    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, join('../results', arch, 'model_best.tar'))


def adjust_lr(optimizer, epoch, lr, dec_freq=40):
    """Sets the learning rate to the init `lr`
       decayed by 10 every `dec_freq` epochs
    """
    new_lr = lr * (0.1 ** (epoch // dec_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_history(fname):
    import matplotlib.pyplot as plt

    with open(fname, 'rb') as f:
        infos = f.readlines()

    model_name = infos[0].strip('\n').split(": ")[1]
    batch_size = infos[1].strip('\n').split(": ")[1]
    epochs = infos[2].strip('\n').split(": ")[1]

    train_loss = []
    train_acc = []
    train_steps = []

    ln_i = 4
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

    plt.figure(1)

    plt.plot(train_steps, train_loss, 'r-',
             label='train loss',
             lw=1.2)
    plt.plot(valid_epoch, valid_loss, 'c-',
             label='valid loss',
             lw=1.2)

    plt.xlabel('train steps')
    plt.ylabel('loss')
    plt.title(model_name + " loss")
    plt.legend(loc='upper right')
    plt.savefig('../results/' + model_name + "/loss.png")

    plt.figure(2)
    plt.plot(train_steps, train_acc, 'r-',
             label='train acc',
             lw=1.2)
    plt.plot(valid_epoch, valid_acc, 'c-',
             label='valid acc',
             lw=1.2)

    plt.xlabel('train steps')
    plt.ylabel('accuracy')
    plt.title(model_name + " accuracy")
    plt.legend(loc='upper left')
    plt.savefig('../results/' + model_name + "/acc.png")
