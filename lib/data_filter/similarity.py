# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Similarity measurement. """

import numpy as np


def cos_sim(vec, mat):
    """Cosine similarity between vector and matrix. """
    numer = np.dot(mat, vec)
    vec_norm = np.linalg.norm(vec)
    mat_norm = np.linalg.norm(mat, axis=1)

    return np.divide(numer, vec_norm * mat_norm)


def euclid_dist(vec, mat):
    """ Euclidean distance between vector and matrix. """
    return np.linalg.norm(mat - vec, axis=1)


def _euc_sim(vec):
    """ Euclidean similarity. """
    return (1. - vec) / np.sum(vec)


def hybrid_sim(vec, mat, top_k=5, alpha=0.1):
    """ Hybrid similarity between vector and matrix. 

    Args:
        vec: source vector
        mat: target matrix
        top_k: keep top k vector
        alpha: euclidean distance impact factor

    Returns:
        idx: top k vector indexes
        hyb_vec: hybrid similarity of top k

    """
    euc_vec = 0.
    idx = np.array(range(mat.shape[0]))

    if alpha > 0:
        # step 1: caculate euclidean distance
        euc_vec = euclid_dist(vec, mat)
        if len(euc_vec) > top_k:
            idx = np.argsort(euc_vec)[:top_k]
            euc_vec = euc_vec[idx]
            mat = mat[idx, :]
        euc_vec = _euc_sim(euc_vec)

    cos_vec = cos_sim(vec, mat)
    hyb_vec = (1 - alpha) * (0.5 * cos_vec + 0.5) + alpha * euc_vec

    return idx, hyb_vec


def mm_similarity(s1, s2):
    """ The unit of measurement similarity.
        Espicially for `oz`, `lb`, `meter` etc.

    Example:
        s1 = '1oz'
        s2 = '12oz'

    """
    if filter(str.isalpha, s1) == filter(str.isalpha, s2):
        if len(s1) < len(s2):
            return float(len(s1)) / len(s2)
        else:
            return float(len(s2)) / len(s1)
    else:
        return 0.


def lcs_similarity(s1, s2):
    """ The longest common substring similarity.
        A special verison.

    Example:
        s1 = 'cyprinid'
        s2 = 'cyprinidae'
    """
    max_len = 0
    i = 0

    while s1[i] == s2[i]:
        max_len += 1
        i += 1
        if len(s1) == i or len(s2) == i:
            break

    if len(s1) < len(s2):
        return float(max_len) / len(s2)
    else:
        return float(max_len) / len(s1)


def cos_similarity(v1, v2):
    """cosine similarity."""
    numer = np.dot(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    return np.divide(numer, v1_norm * v2_norm)


# if __name__ == "__main__":
#     mat = np.array([[1, 2, 3], [10, 20, 40], [4, 6, 7], [2, 3, 1], [1, 2, 4]])
#     vec = np.array([1, 2, 4])

#     # print euclid_dist(vec, mat)
#     # print cos_sim(vec, mat)
#     idx, v = hybrid_sim(vec, mat, top_k=2, alpha=0.1)

#     print idx
#     print mat[idx]
#     print v
