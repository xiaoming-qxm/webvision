# -*- coding: utf-8 -*-

import torchwordemb
import torch
import numpy as np
from similarity import *


def load_word_vec(model_path):
    vocab, vec = torchwordemb.load_word2vec_bin(model_path)
    return vocab, vec


def main():
    model_path = "/home/simon/data/GoogleNews-vectors-negative300.bin"
    vocab, vec = load_word_vec(model_path)

    fish = vec[vocab['musician']].numpy()
    tail = vec[vocab['Test']].numpy()
    print cos_similarity(fish, tail)


if __name__ == "__main__":
    main()
