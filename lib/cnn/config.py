# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" WebVision configuration system. """

import os
import os.path as osp
from easydict import EasyDict as edict

__C = edict()

# Users can get config by:
# from cfgs import config
cfg = __C

# Pixel mean and stddev values (RGB order)
# We use the same pixel mean and stddev for all networks

__C.PIXEL_MEANS = [0.502, 0.476, 0.428]
__C.PIXEL_STDS = [0.299, 0.296, 0.309]

# For reprodicibility
__C.RND_SEED = 10

# Input size
__C.INPUT_SIZE = [299, 299]
