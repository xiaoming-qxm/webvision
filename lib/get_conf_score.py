# -*- coding: utf-8 -*-
# Author: Xiaoming　Qin

"""Test a convolutional neural network"""

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
from datasets.folder import GenImageFolder, SpecImageFolder
from PIL import Image
import pprint


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
    parser.add_argument('--params-file', help='trained model',
                        default='model_best.tar', type=str)
    parser.add_argument('--input-size', help='size of model input',
                        default=299, type=int)
    parser.add_argument('--mode', help='test mode, `all` for test on all '
                        'classes, `each` for on each class', default='all',
                        type=str)
    parser.add_argument('--test-name', help='dataset name to test on',
                        default='test', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_confidence_score(test_loader, model, criterion,
                         save_path, num_classes=40):
    # switch to evaluate mode
    model.eval()

    scores = []

    for idx, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        output = model(inputs)
        loss = criterion(output, labels)

        scores.append(output.data.cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    imgs = test_loader.dataset.imgs

    for i in range(num_classes):
        conf_score = []
        for j in range(scores.shape[0]):
            gt_label = int(imgs[j][1])
            if gt_label == i:
                gt_label = 1
            else:
                gt_label = 0
            conf_score.append([j, scores[j][i], gt_label])

        conf_score = sorted(conf_score, key=lambda item: item[1], reverse=True)

        # print conf_score

        with open(pjoin(save_path, "{}.lst".format(i)), 'w') as f:
            for j in range(scores.shape[0]):
                f.write("{0:d} {1:.4f} {2:d}\n".format(
                    conf_score[j][0], conf_score[j][1], conf_score[j][2]))

        # for i in range(probs.shape[0]):
        #     print sum(probs[i, :])
        #     probs[i, :] = probs[i, :] / float(sum(probs[i, :]))
        #     print sum(probs[i, :])


def test_model(data_root, gpus, batch_size=16,
               params_file='model_best.tar',
               num_workers=4, num_classes=40,
               in_size=224, mode='all',
               test_name='test'):

    normalize = transforms.Normalize(mean=cfg.PIXEL_MEANS,
                                     std=cfg.PIXEL_STDS)
    test_transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        normalize])

    assert os.path.isfile(params_file), "{} is not exist.".format(params_file)
    params = torch.load(params_file)

    # define the model
    model = get_model(name=params['arch'], num_classes=num_classes)
    if len(gpus) > 1:
        gpus = [int(i) for i in gpus.strip('[]').split(',')]
        model = torch.nn.DataParallel(model, device_ids=gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model.cuda()
    model.load_state_dict(params['state_dict'])

    loss_func = torch.nn.CrossEntropyLoss().cuda()

    test_data = GenImageFolder(root=pjoin(data_root, test_name),
                               transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    save_path = "../data/conf_score"
    get_confidence_score(test_loader, model, loss_func, save_path)


def main():
    args = parse_args()

    test_model(data_root=args.data_path,
               gpus=args.gpus,
               batch_size=args.batch_size,
               num_workers=args.num_workers,
               in_size=args.input_size,
               params_file=args.params_file,
               mode=args.mode,
               test_name=args.test_name)


if __name__ == "__main__":
    main()
