# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Filter images via its word-level feature from metadata. """

import os
import json
from os.path import join as pjoin
import torchwordemb
import torch
import numpy as np
from similarity import *


def load_word_vec(model_path):
    vocab, vec = torchwordemb.load_word2vec_bin(model_path)
    return vocab, vec


def find_feat_in_tags(tags, feats):
    count = 0
    for tag in tags:
        if tag in feats:
            count += 1.
    return count


def sim_tag_and_feat(tags, feats, feat_vecs,
                     vocab, vec):
    """ Calculate similarity between tags and features. """
    sim = find_feat_in_tags(tags, feats)

    if sim == 0.:
        probs = []
        # for debugging
        cand_tags_name = []

        for tag in tags:
            res = []
            if vocab.has_key(tag):
                    # for debugging
                cand_tags_name.append(tag)
                tag_vec = vec[vocab[tag]].numpy()
                for feat in feat_vecs:
                    res.append(cos_similarity(tag_vec, feat[1]))
                probs.append(res)

        if not len(probs):
            sim = 0.
            return sim

        probs = np.array(probs)
        # Max pooling along all the tags
        # probs = np.max(probs, axis=0)
        max_p = np.max(probs)
        index = np.where(probs == max_p)
        # best_tag = cand_tags_name[index[0][0]]
        # best_feat = cand_feat_name[index[1][0]]

        sim = max_p

    return sim


def calc_sim_tag_with_feat(tags, feat_vecs,
                           cand_feat_name,
                           vocab, vec):
    """ Calculate similarity between tags and features. """
    probs = []

    # for debugging
    cand_tags_name = []

    for tag in tags:
        res = []
        if vocab.has_key(tag):
            # for debugging
            cand_tags_name.append(tag)

            tag_vec = vec[vocab[tag]].numpy()
            for feat in feat_vecs:
                res.append(cos_similarity(tag_vec, feat[1]))
            probs.append(res)

    # print cand_tags_name

    if not len(probs):
        return 0, "", ""

    probs = np.array(probs)
    # Max pooling along all the tags
    # probs = np.max(probs, axis=0)

    max_p = np.max(probs)

    index = np.where(probs == max_p)

    best_tag = cand_tags_name[index[0][0]]
    best_feat = cand_feat_name[index[1][0]]

    return max_p, best_tag, best_feat


def save2file(clean_set, noisy_set,
              save_path, cls_name,
              suffix):
    save_file = pjoin(save_path, 'clean',
                      cls_name + '.txt')

    with open(save_file, 'a') as f:
        for i in range(len(clean_set)):
            f.write(clean_set[i] + '\n')

    save_file = pjoin(save_path, 'noisy',
                      cls_name + '.txt')
    with open(save_file, 'a') as f:
        for i in range(len(noisy_set)):
            f.write(noisy_set[i] + '\n')


def filter_via_pan(pos_feat, neg_feat,
                   pos_vec, neg_vec,
                   vocab, vector,
                   json_names, data_path,
                   save_path, cls_name):
    """ filter data via pos and neg features. """

    for fld in ['flickr', 'google']:
        clean_set = []
        noisy_set = []
        for name in json_names:
            js_file = pjoin(data_path, fld, name + '.json')
            with open(js_file, 'rb') as f:
                fl_info = json.load(f)

            for j in xrange(len(fl_info)):
                info = fl_info[j]
                idx = info['id']
                tags = info['tags']

                sim_p = sim_tag_and_feat(tags, pos_feat,
                                         pos_vec, vocab, vector)
                sim_n = sim_tag_and_feat(tags, neg_feat,
                                         neg_vec, vocab, vector)

                if sim_p >= 0.5 and sim_n >= 0.5:
                    if sim_n < sim_p:
                        clean_set.append(idx)
                    else:
                        noisy_set.append(idx)
                elif sim_p >= 0.5:
                    clean_set.append(idx)
                else:
                    noisy_set.append(idx)

        # write files
        save2file(clean_set, noisy_set,
                  save_path, cls_name, fld)


def filter_via_pos(pos_feat, pos_vec,
                   vocab, vector,
                   json_names, data_path,
                   save_path, cls_name):
    """ filter data via negative features. """
    for fld in ['flickr', 'google']:
        clean_set = []
        noisy_set = []
        for name in json_names:
            js_file = pjoin(data_path, fld, name + '.json')
            with open(js_file, 'rb') as f:
                fl_info = json.load(f)
            for j in xrange(len(fl_info)):
                info = fl_info[j]
                idx = info['id']
                tags = info['tags']

                sim_p = sim_tag_and_feat(tags, pos_feat,
                                         pos_vec, vocab, vector)
                if sim_p >= 0.5:
                    clean_set.append(idx)

        # TODO (step 2)
        save2file(clean_set, noisy_set,
                  save_path, cls_name, fld)


def filter(feat_path, fname, vocab,
           vector, data_path,
           json_names, save_path):
    # load hand-crafted word feature
    with open(pjoin(feat_path, fname), 'rb') as f:
        feats = json.load(f)

    cls_name = fname[:-4]
    # split into positive and negative features
    pos_feat = set(feats['pos'])
    neg_feat = set(feats['neg'])

    pos_vec = []
    neg_vec = []

    for ft in feats['pos']:
        if vocab.has_key(ft):
            pos_vec.append((ft, vector[vocab[ft]].numpy()))

    for ft in feats['neg']:
        if vocab.has_key(ft):
            neg_vec.append((ft, vector[vocab[ft]].numpy()))

    if len(pos_feat) and len(neg_feat):
        filter_via_pan(pos_feat, neg_feat,
                       pos_vec, neg_vec,
                       vocab, vector,
                       json_names, data_path,
                       save_path, cls_name)
    else:
        filter_via_pos(pos_feat, pos_vec,
                       vocab, vector,
                       json_names, data_path,
                       save_path, cls_name)


def main(feat_path, model_path, data_path, save_path):
    """ main function to ilter all data.
    Args:
        feat_path: path to hand-crafted word feature files
        model_path: path to word2vec pre-trained model
    """
    # load word2vec model
    vocab, vector = load_word_vec(model_path)
    # load labels queries map
    lq_map = {}
    map_file = pjoin(feat_path, '..',
                     'labels_queries_map.txt')
    with open(map_file, 'rb') as f:
        lines = f.readlines()

    for line in lines:
        _, lbl_name, queries = line.strip().split(" ")
        queries = queries.split(",")
        lq_map[lbl_name] = queries

    for name in os.listdir(feat_path):
        filter(feat_path, name, vocab, vector,
               data_path, lq_map[name[:-4]],
               save_path)


if __name__ == "__main__":
    feat_path = "../../data/word_feat"
    model_path = "../../data/GoogleNews-vectors-negative300.bin"
    data_path = "../../data"
    save_path = "../../data"

    main(feat_path, model_path, data_path, save_path)
