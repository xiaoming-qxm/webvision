# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Word vector arithmetic. """

import numpy as np
import torchwordemb
import torch


def load_word_vec(model_path):
    vocab, vec = torchwordemb.load_word2vec_bin(model_path)
    return vocab, vec


def cos_similarity(v1, v2):
    """cosine similarity."""
    numer = np.dot(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    return np.divide(numer, v1_norm * v2_norm)


def euclid_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


def vec_arith(s1_pos, s2_pos, s3_neg, top_k=1):
    """ Vector arithmetic.
        formula: s1_pos - s3_neg + s2_pos = ?

    Examples:
        "king" - "man" + "woman" = ?

    """
    model_path = "../data/GoogleNews-vectors-negative300.bin"
    vocab, vec = load_word_vec(model_path)

    pos_v1 = vec[vocab[s1_pos]].numpy()
    pos_v2 = vec[vocab[s2_pos]].numpy()
    neg_v3 = vec[vocab[s3_neg]].numpy()
    res_vec = pos_v1 - neg_v3 + pos_v2

    best_cos_sim = -1.
    best_euc_dist = np.inf
    best_cos_name = ""
    best_euc_name = ""

    for key in vocab.keys():
        if key == s1_pos:
            continue
        cand_vec = vec[vocab[key]].numpy()
        cand_cos_sim = cos_similarity(res_vec, cand_vec)
        cand_euc_dist = euclid_dist(res_vec, cand_vec)
        if cand_cos_sim > best_cos_sim:
            best_cos_sim = cand_cos_sim
            best_cos_name = key

        if cand_euc_dist < best_euc_dist:
            best_euc_dist = cand_euc_dist
            best_euc_name = key

    print("Using cosine similarity:")
    print("{} - {} + {} = {}".format(s1_pos, s3_neg,
                                     s2_pos, best_cos_name))
    print("Similarity: {}".format(best_cos_sim))
    print("Using euclidean distance:")
    print("{} - {} + {} = {}".format(s1_pos, s3_neg,
                                     s2_pos, best_euc_name))
    print("Distance: {}".format(best_euc_dist))


if __name__ == "__main__":
    s1_pos = "smallest"
    s2_pos = "big"
    s3_neg = "small"

    # s1_pos = "king"
    # s2_pos = "woman"
    # s3_neg = "man"
    vec_arith(s1_pos, s2_pos, s3_neg)
