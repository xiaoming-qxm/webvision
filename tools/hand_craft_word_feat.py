# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create hand-crafted word level feature for different classes. """

import json
from os.path import join as pjoin


def save_tench_feat(data_root):
    pos_tag = ["fishery", "fishes", "fish", "fishing",
               "angling", "angler", "anglers", "ichthyology",
               "golden", "tail", "fin", "12lbs", "1oz",
               "cyprinidae", "river", "rivers", "water",
               "lake", "lakes", "canals", "freshwater", "baits"]
    # TODO add feature weights
    pos_weights = []

    neg_tag = ["submarines", "people", "prison", "person",
               "man", "woman", "musician", "site", "submarine",
               "men", "women"]
    neg_weights = []

    feat = {"pos": pos_tag, "neg": neg_tag}

    with open(pjoin(data_root, "tench.txt"), 'w') as f:
        json.dump(feat, f, indent=4)


data_root = "../data/hc_word_feat"
save_tench_feat(data_root)
