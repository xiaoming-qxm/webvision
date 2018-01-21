# -*- coding: utf-8 -*-
# Author: Xiaoming　Qin

"""Predict probability of the groud-truth class."""

import argparse
import os
import sys
import torch
import numpy as np
from os.path import join as pjoin
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from cnn.factory import get_model
from cnn.config import cfg
from datasets.folder import SpecImageFolder
from PIL import Image
import torch.nn as nn


def parse_args():

    parser = argparse.ArgumentParser(description='Test a deep neural network')
    parser.add_argument('--data-path', help='path to data root',
                        default='/data/wv-40', type=str)
    parser.add_argument('--gpus', help='GPU id to use',
                        default='0', type=str)
    parser.add_argument('--batch-size', help='mini-batch size',
                        default=16, type=int)
    parser.add_argument('--num-workers', help='number of workers',
                        default=4, type=int)
    parser.add_argument('--params-file', help='model to train on',
                        default='model_best.tar', type=str)
    parser.add_argument('--input-size', help='size of model input',
                        default=299, type=int)
    parser.add_argument('--save-path', help='path to save results',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def pred_gt_probs(pred_loader, model, rescale, save_path, cls_id):
    # switch to evaluate mode
    model.eval()

    if not os.path.exists(pjoin(save_path, cls_id)):
        os.mkdir(pjoin(save_path, cls_id))

    p_arr = []
    num_samples = 0.
    num_correct = 0.
    # ground truth label
    gt_label = int(cls_id)
    for idx, data in enumerate(pred_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        output = model(inputs)
        probs = rescale(output)
        gt_prob = probs[:, gt_label].cpu().data.numpy()
        p_arr.append(gt_prob)

        # # For debug
        # _, preds = torch.max(output, 1)
        # correct = (preds == labels).sum()
        # num_samples += labels.size(0)
        # num_correct += correct.data[0]

    # print("gt label: {}".format(gt_label))
    # print("num correct: {}".format(num_correct))
    # print("num samples: {}".format(num_samples))
    # print("acc: {0:.4f}".format(num_correct / float(num_samples)))

    p_arr = np.concatenate(p_arr, axis=0)

    return num_correct, num_samples


def predict_model(data_root, gpus, batch_size=16,
                  params_file='model_best.tar',
                  num_workers=4, num_classes=40,
                  in_size=224, save_path=None):

    normalize = transforms.Normalize(mean=cfg.PIXEL_MEANS,
                                     std=cfg.PIXEL_STDS)

    pred_transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        normalize])

    assert os.path.isfile(params_file), "{} is not exist.".format(params_file)
    params = torch.load(params_file)

    # define the model
    model = get_model(name=params['arch'], num_classes=num_classes)
    model.cuda()
    model.load_state_dict(params['state_dict'])

    # Operation to get probability
    rescale = nn.Softmax(dim=1)

    num_correct = 0.
    num_samples = 0.

    for cls_id in sorted(os.listdir(data_root)):
        print("Processing class {}".format(cls_id))
        pred_data = SpecImageFolder(root=pjoin(data_root, cls_id),
                                    transform=pred_transform)
        pred_loader = DataLoader(dataset=pred_data, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
        nc, ns = pred_gt_probs(pred_loader, model, rescale, save_path, cls_id)
        num_correct += nc
        num_samples += ns

    print("ave acc: {0:.5f}".format(num_correct / float(num_samples)))


def main():
    args = parse_args()

    predict_model(data_root=args.data_path,
                  gpus=args.gpus,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  in_size=args.input_size,
                  params_file=args.params_file,
                  save_path=args.save_path)


if __name__ == "__main__":
    main()
