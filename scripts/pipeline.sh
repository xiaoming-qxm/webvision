#!/bin/bash

params_file=$1

cd ../lib

# extract features.py
python extract_features.py --data-path /data/wv-40/train --params-file ${params_file} --save-path ../data/img_feat/


# pred_gt_prob.py
python pred_gt_prob.py --data-path /data/wv-40/train --params-file ${params_file} --save-path ../data/pred_probs/

# get train list
python sort_prob.py
