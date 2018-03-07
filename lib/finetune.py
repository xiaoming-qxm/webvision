# -*- coding: utf-8 -*-
# Author: Xiaoming　Qin

"""Finetune a convolutional neural network"""

import sys
import os
import torch
import argparse
from PIL import Image
from os.path import join as pjoin
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from datasets.folder import GenImageFolder
from cnn.utils import adjust_lr, save_ckpt, adjust_lr_manual
from cnn.factory import get_model
from cnn.config import cfg
from cnn.logger import Logger


def parse_args():

    parser = argparse.ArgumentParser(description='Train a deep neural network')
    parser.add_argument('--data-path', help='path to data root',
                        default='/data/wv-40', type=str)
    parser.add_argument('--gpus', help='GPU id to use',
                        default='0', type=str)
    parser.add_argument('--epochs', help='number of epochs',
                        type=int)
    parser.add_argument('--batch-size', help='mini-batch size',
                        default=16, type=int)
    parser.add_argument('--lr', help='initial learning rate',
                        default=0.001, type=float)
    parser.add_argument('--weight-decay', help='learing weight decay',
                        default=1e-4, type=float)
    parser.add_argument('--num-workers', help='number of workers',
                        default=4, type=int)
    parser.add_argument('--input-size', help='size of model input',
                        default=299, type=int)
    parser.add_argument('--print-freq', help='print frequency',
                        default=100, type=int)
    parser.add_argument('--params-file', help='pre-trained model',
                        default='model_best.tar', type=str)
    parser.add_argument('--train-name', help='dataset name to train on',
                        default='train', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def validate(valid_loader, model, criterion, epoch, log):
    # switch to evaluate mode
    model.eval()

    num_correct = 0.
    num_samples = 0.
    num_batch = 0
    v_value_loss = 0.
    for idx, data in enumerate(valid_loader, 0):
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
        v_value_loss += loss.data[0]

    ave_loss = v_value_loss / num_batch
    ave_acc = num_correct / float(num_samples)

    log.record_val([epoch + 1, ave_loss, ave_acc], epoch)

    print("[valid {0:d}] - loss: {1:.4f} - acc: {2:.4f}".format(
        epoch + 1, ave_loss, ave_acc))


def finetune(train_loader, model, criterion,
             optimizer, epoch, log, print_freq):
    # switch to train mode
    model.train()

    print("[epoch: {}]".format(epoch + 1))

    running_loss, running_acc = 0., 0.
    num_samples, num_correct = 0., 0.
    info = []

    for idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()

        # feedforward
        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)
        correct = (preds == labels).sum()

        num_samples += labels.size(0)
        num_correct += correct.data[0]
        running_loss += loss.data[0]

        if idx % print_freq == (print_freq - 1):
            running_acc = num_correct / float(num_samples)
            ave_running_loss = running_loss / print_freq

            print("[{0:d}, {1:4d}] loss: {2:.4f} - acc: {3:.4f}".format(
                epoch + 1, idx + 1,
                ave_running_loss,
                running_acc))

            log.record_trn([epoch + 1, idx + 1,
                            ave_running_loss,
                            running_acc])

            num_samples, num_correct = 0., 0.
            running_loss = 0.


def finetune_model(data_root, gpus, epochs, batch_size,
                   base_lr, weight_decay, num_workers,
                   in_size, print_freq, params_file,
                   train_name, num_classes=40):
    normalize = transforms.Normalize(mean=cfg.PIXEL_MEANS,
                                     std=cfg.PIXEL_STDS)

    # Image transformer
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(in_size),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    valid_transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        normalize])

    train_data = GenImageFolder(root=pjoin(data_root, train_name),
                                transform=train_transform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    valid_data = GenImageFolder(root=pjoin(data_root, 'valid'),
                                transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    assert os.path.isfile(params_file), "{} is not exist.".format(params_file)
    params = torch.load(params_file)

    # define the model
    model_name = params['arch']
    model = get_model(name=model_name, num_classes=num_classes)
    if len(gpus) > 1:
        gpus = [int(i) for i in gpus.strip('[]').split(',')]
        model = torch.nn.DataParallel(model, device_ids=gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model.cuda()
    model.load_state_dict(params['state_dict'])

    # define optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                weight_decay=weight_decay,
                                momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss().cuda()

    # define logs
    logger = Logger(arch=model_name, epochs=epochs,
                    batch_size=batch_size)
    # create model results folder
    res_save_fld = os.path.join('../results', model_name)
    if not os.path.exists(res_save_fld):
        os.mkdir(res_save_fld)

    lr_epoch_map = {0: 0.1, 30: 0.01, 80: 0.001, 110: 0.0001}

    for epoch in range(epochs):
        adjust_lr_manual(optimizer, epoch, lr_epoch_map)

        # finetune for one epoch
        finetune(train_loader, model, loss_func, optimizer,
                 epoch, logger, print_freq)

        # evaluate on validation set
        validate(valid_loader, model, loss_func, epoch, logger)

        if epoch == logger.best_epoch:
            save_ckpt({'arch': model_name,
                       'state_dict': model.state_dict()},
                      model_name, epoch, is_best=True)
        else:
            save_ckpt({'arch': model_name,
                       'state_dict': model.state_dict()},
                      model_name, epoch, is_best=False)

    print("Finished Training")
    logger.save()


def main():
    args = parse_args()

    finetune_model(data_root=args.data_path,
                   gpus=args.gpus,
                   epochs=args.epochs,
                   batch_size=args.batch_size,
                   base_lr=args.lr,
                   weight_decay=args.weight_decay,
                   num_workers=args.num_workers,
                   in_size=args.input_size,
                   print_freq=args.print_freq,
                   params_file=args.params_file,
                   train_name=args.train_name)


if __name__ == "__main__":
    main()
