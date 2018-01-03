# -*- coding: utf-8 -*-

"""Test a convolutional neural network"""

import argparse
import os
import sys
import torch
from os.path import join as pjoin
from factory import get_model
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def parse_args():

    parser = argparse.ArgumentParser(description='Test a deep neural network')
    parser.add_argument('--data-path', help='path to data root',
                        default='/data/wv-40', type=str)
    parser.add_argument('--gpus', help='GPU id to use',
                        default='[0]', type=str)
    parser.add_argument('--batch-size', help='mini-batch size',
                        default=16, type=int)
    parser.add_argument('--num-workers', help='number of workers',
                        default=4, type=int)
    parser.add_argument('--params-file', help='model to train on',
                        default='model_best.tar', type=str)
    parser.add_argument('--input-size', help='size of model input',
                        default=299, type=int)

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
               in_size=224):

    normalize = transforms.Normalize(mean=[0.502, 0.476, 0.428],
                                     std=[0.299, 0.296, 0.309])

    test_transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        normalize])

    test_data = datasets.ImageFolder(root=pjoin(data_root, 'test'),
                                     transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    assert os.path.isfile(params_file), "{} is not exist.".format(params_file)

    params = torch.load(params_file)

    # define the model
    model = get_model(name=params['arch'], num_classes=num_classes)
    if len(gpus) > 1:
        gpus = [int(i) for i in gpus.strip('[]').split(',')]
        model = torch.nn.DataParallel(model, device_ids=gpus)
    model.cuda()
    model.load_state_dict(params['state_dict'])

    loss_func = torch.nn.CrossEntropyLoss().cuda()

    test(test_loader, model, loss_func)


def main():
    args = parse_args()

    test_model(data_root=args.data_path,
               gpus=args.gpus,
               batch_size=args.batch_size,
               num_workers=args.num_workers,
               in_size=args.input_size,
               params_file=args.params_file
               )


if __name__ == "__main__":
    main()
