# -*- coding: utf-8 -*-
# Author: Xiaoming　Qin

"""Test a convolutional neural network"""

import argparse
import os
import sys
import torch
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


def test(test_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    num_correct = 0.
    num_samples = 0.
    num_batch = 0
    loss_value = 0.
    for idx, data in enumerate(test_loader, 0):
        num_batch = idx + 1
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        output = model(inputs)
        loss = criterion(output, labels)

        _, preds = torch.max(output, 1)
        correct = (preds == labels).sum()

        num_samples += labels.size(0)
        num_correct += correct.data[0]
        loss_value += loss.data[0]

    ave_loss = loss_value / num_batch
    ave_acc = num_correct / float(num_samples)

    print("Test set: ave loss: {0:.4f} - ave acc: {1:.4f}".format(
        ave_loss, ave_acc))


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

    # test mode
    if mode == "each":
        base_path = pjoin(data_root, test_name)
        cls_list = os.listdir(base_path)
        cls_list = sorted([int(c) for c in cls_list])
        for cls_id in cls_list:
            print("---             class id: {}             ---".format(
                cls_id))
            test_data = SpecImageFolder(root=pjoin(base_path, str(cls_id)),
                                        transform=test_transform)
            test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers,
                                     pin_memory=True)
            test(test_loader, model, loss_func)
    else:
        test_data = GenImageFolder(root=pjoin(data_root, test_name),
                                   transform=test_transform)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
        test(test_loader, model, loss_func)


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
