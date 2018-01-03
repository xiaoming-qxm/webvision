# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

import numpy as np


def cos_similarity(v1, v2):
    """cosine similarity."""
    numer = np.dot(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    return np.divide(numer, v1_norm * v2_norm)


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
