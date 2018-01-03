# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Filter images via its word-level feature from metadata. """

import json
import numpy as np
import torchwordemb
from os.path import join as pjoin


data_root = "/home/simon/webvision/data"


def load_word_vec(model_path):
    vocab, vec = torchwordemb.load_word2vec_bin(model_path)
    return vocab, vec


def find_feat_in_tags(tags, feats):
    count = 0
    for tag in tags:
        if tag in feats:
            count += 1.
    return count


# load hand-crafted word feature
with open(pjoin(data_root, 'hc_word_feat', 'tench.txt'), 'rb') as f:
    feat = json.load(f)

# positive feature
pos_feat = feat['pos']
# negative feature
neg_feat = feat['neg']

pos_vec = {}
neg_vec = {}

model_path = "/home/simon/data/GoogleNews-vectors-negative300.bin"
vocab, vec = load_word_vec(model_path)

# convert positive and negative word feature to vector
for ft in pos_feat:
    if vocab.has_key(ft):
        pos_vec[ft] = vec[vocab[ft]].numpy()
    else:
        pos_vec[ft] = np.array([])

for ft in neg_feat:
    if vocab.has_key(ft):
        neg_vec[ft] = vec[vocab[ft]].numpy()
    else:
        neg_vec[ft] = np.array([])

# for faster reason
pos_feat = set(pos_feat)
neg_feat = set(neg_feat)

# load json file
json_names = ["q0001.json", "q0002.json"]


for name in json_names:
    with open(pjoin(data_root, 'google', name), 'rb') as f:
        fl_info = json.load(f)

    clean_i_score = {}
    noisy_i_score = {}
    vague_idx = []

    # step 1
    # find if word exists
    for j in xrange(len(fl_info)):
        info = fl_info[j]
        idx = info['id']
        tags = info['tags']

        num_p = find_feat_in_tags(tags, pos_feat)
        num_n = find_feat_in_tags(tags, neg_feat)

        if num_p:
            clean_i_score[idx] = num_p
        if num_n:
            noisy_i_score[idx] = num_n

        # both are not exist
        if (not num_p) and (not num_n):
            vague_idx.append(j)

    # print("all number: ")
    # print(len(fl_info))
    # print("clean number: ")
    # print(len(clean_i_score))
    # print("noisy number: ")
    # print(len(noisy_i_score))
    # print("vague number: ")
    # print(len(vague_idx))
    # print("vague indexes: ")
    # print(vague_idx)

    # step 2
    for j in vague_idx:
        print "----------------------------------"
        print fl_info[j]

    # break

    # step 3

    # step 4

    # with open(pjoin(data_root, 'flickr', nm), 'rb') as f:
    #     fl_dt = json.load(f)
