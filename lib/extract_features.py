# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Extract bottleneck features using neural network. """

import argparse
import os
import math
import sys
import torch
import cPickle
from os.path import join as pjoin
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from cnn.factory import get_model
from cnn.config import cfg
from datasets.folder import SpecImageFolder
from PIL import Image
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser(description='Extract features using CNN')
    parser.add_argument('--data-path', help='path to data root',
                        default='/data/wv-40/train', type=str)
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
    parser.add_argument('--save-path', help='root path to save features',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def save_to_pickle(features, save_path, cls_id, fname):

    with open(pjoin(save_path, cls_id, fname + ".pkl"), 'wb') as f:
        cPickle.dump(features, f, protocol=cPickle.HIGHEST_PROTOCOL)


def save_to_txt(img_tuple, save_path, cls_id):
    with open(pjoin(save_path, cls_id, "img_name.lst"), 'w') as f:
        for img_path, _ in img_tuple:
            f.write(img_path + '\n')


def extract_feats(ext_loader, model,
                  save_path, cls_id,
                  save_freq=200):
    # switch to evaluate mode
    model.eval()

    if not os.path.exists(pjoin(save_path, cls_id)):
        os.mkdir(pjoin(save_path, cls_id))

    batch_feat = []
    img_names = ext_loader.dataset.imgs
    batch_size = ext_loader.batch_size
    num_img = len(img_names)
    init_idx = 0
    pkl_idx = 0
    last_idx = int(math.ceil(num_img / float(batch_size))) - 1

    for idx, data in enumerate(ext_loader, 0):
        inputs, _ = data
        inputs = Variable(inputs.cuda())
        feats = model(inputs)

        cpu_feat = feats.data.cpu().numpy()
        if len(cpu_feat.shape) == 1:
            cpu_feat = np.reshape(cpu_feat, (1, -1))
        batch_feat.append(cpu_feat)

        if idx % save_freq == (save_freq - 1):
            # batch_im_list = img_names[
            #     init_idx: batch_size * save_freq + init_idx]
            # init_idx = batch_size * save_freq + init_idx
            batch_feat = np.concatenate(batch_feat, axis=0)
            save_to_pickle(batch_feat, save_path, cls_id, str(pkl_idx))

            batch_feat = []
            pkl_idx += 1

        elif idx == last_idx:
            # batch_im_list = img_names[init_idx:]
            batch_feat = np.concatenate(batch_feat, axis=0)

            save_to_pickle(batch_feat, save_path, cls_id, str(pkl_idx))

    # save to text
    save_to_txt(img_names, save_path, cls_id)


def extract_model(data_root, gpus, batch_size=16,
                  params_file="model_best.tar",
                  num_workers=4, num_classes=40,
                  in_size=224, save_path=None):
    normalize = transforms.Normalize(mean=cfg.PIXEL_MEANS,
                                     std=cfg.PIXEL_STDS)

    ext_transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        normalize])

    assert os.path.isfile(params_file), "{} is not exist.".format(params_file)
    params = torch.load(params_file)

    # define the model
    model = get_model(name=params['arch'],
                      num_classes=num_classes, extract_feat=True)
    model.cuda()
    model.load_state_dict(params['state_dict'])

    for cls_id in sorted(os.listdir(data_root)):
        print("Processing class {}".format(cls_id))
        ext_data = SpecImageFolder(root=pjoin(data_root, cls_id),
                                   transform=ext_transform)
        ext_loader = DataLoader(dataset=ext_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        extract_feats(ext_loader, model, save_path, cls_id)


def main():
    args = parse_args()

    extract_model(data_root=args.data_path,
                  gpus=args.gpus,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  in_size=args.input_size,
                  params_file=args.params_file,
                  save_path=args.save_path)


if __name__ == "__main__":
    main()
