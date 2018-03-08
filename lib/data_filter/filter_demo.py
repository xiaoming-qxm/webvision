# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Filter images via its word-level feature from metadata. """

import json
import numpy as np
import torchwordemb
from os.path import join as pjoin
from similarity import *


data_root = "../../data"


def load_word_vec(model_path):
    vocab, vec = torchwordemb.load_word2vec_bin(model_path)
    return vocab, vec


def find_feat_in_tags(tags, feats):
    count = 0
    for tag in tags:
        if tag in feats:
            count += 1.
    return count


def calc_sim_tag_with_feat(tags, feat_mat,
                           cand_feat_name,
                           vocab, vec):
    """ Calculate similarity between tags and features. """
    probs = []

    # for debugging
    cand_tags_name = []

    # for tag in tags:
    #     res = []
    #     if vocab.has_key(tag):
    #         # for debugging
    #         cand_tags_name.append(tag)

    #         tag_vec = vec[vocab[tag]].numpy()
    #         for feat in feat_vecs:
    #             res.append(cos_similarity(tag_vec, feat[1]))
    #         probs.append(res)

    for tag in tags:
        if vocab.has_key(tag):
            # for debugging
            cand_tags_name.append(tag)
            tag_vec = vec[vocab[tag]].numpy()
            # res = cos_sim(tag_vec, feat_mat)
            idx, res = hybrid_sim(tag_vec, feat_mat, top_k=50, alpha=0.5)

            # print idx
            # print res
            probs.append(res)

    # print cand_tags_name

    if not len(probs):
        return 0, "", ""

    probs = np.array(probs)

    # print probs.shape

    # Max pooling along all the tags
    # probs = np.max(probs, axis=0)

    max_p = np.max(probs)

    index = np.where(probs == max_p)

    best_tag = cand_tags_name[index[0][0]]
    best_feat = cand_feat_name[index[1][0]]

    return max_p, best_tag, best_feat


# load hand-crafted word feature
with open(pjoin(data_root, 'word_feat', 'tench.txt'), 'rb') as f:
    feat = json.load(f)

# positive feature
pos_feat = feat['pos']
# negative feature
neg_feat = feat['neg']

pos_vec = []
neg_vec = []

model_path = "../../data/GoogleNews-vectors-negative300.bin"
vocab, vec = load_word_vec(model_path)

# convert positive and negative word feature to vector
for ft in pos_feat:
    if vocab.has_key(ft):
        pos_vec.append((ft, vec[vocab[ft]].numpy()))

for ft in neg_feat:
    if vocab.has_key(ft):
        neg_vec.append((ft, vec[vocab[ft]].numpy()))

# for faster reason
pos_feat = set(pos_feat)
neg_feat = set(neg_feat)

# load json file
json_names = ["q0001.json", "q0002.json"]

# name list
cand_pos_name = []
cand_neg_name = []

tmp_pv = []
tmp_nv = []

for item in pos_vec:
    cand_pos_name.append(item[0])
    tmp_pv.append(item[1])

for item in neg_vec:
    cand_neg_name.append(item[0])
    tmp_nv.append(item[1])

pos_vec = np.asarray(tmp_pv)
neg_vec = np.asarray(tmp_nv)

for name in json_names:
    with open(pjoin(data_root, 'flickr', name), 'rb') as f:
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
    threshold = 0.3

    for j in vague_idx:
        info = fl_info[j]
        idx = info['id']
        tags = info['tags']

        p_prob, p_tags, p_feat = calc_sim_tag_with_feat(tags, pos_vec,
                                                        cand_pos_name,
                                                        vocab, vec)
        n_prob, n_tags, n_feat = calc_sim_tag_with_feat(tags, neg_vec,
                                                        cand_neg_name,
                                                        vocab, vec)

        if p_prob > threshold:
            print "-----------"
            print "POS"
            print "tags"
            print p_tags
            print p_feat
            print p_prob
            print idx
            print "-----------"
        if n_prob > threshold:
            print "-----------"
            print "NEG"
            print "tags"
            print n_tags
            print n_feat
            print n_prob
            print idx
            print "-----------"
