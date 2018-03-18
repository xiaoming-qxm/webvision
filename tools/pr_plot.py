# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.interpolate import spline
from sklearn.metrics import average_precision_score


def calc_pr_ap(data_root, cls_id):
    with open(pjoin(data_root, "{}.lst".format(cls_id)), 'r') as f:
        lines = f.readlines()

    y_score = [float(l.strip().split(" ")[1]) for l in lines]
    y_true = [int(l.strip().split(" ")[2]) for l in lines]

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    precision = list(precision)
    recall = list(recall)
    precision.insert(0, 0.)
    recall.insert(0, 1.)
    precision = np.asarray(precision)
    recall = np.asarray(recall)

    return precision, recall, average_precision


def calc_mAP():
    model_name = "q10"
    data_root = "../data/conf_score/{}".format(model_name)
    sum_ap = 0.
    for i in range(40):
        p, r, ap = calc_pr_ap(data_root, i)
        sum_ap += ap

    mAP = sum_ap / 40

    print("{0:.4f}".format(mAP))


def main(cls_id=3):
    data_root = "../data/conf_score"

    bl_p, bl_r, bl_ap = calc_pr_ap(pjoin(data_root, 'baseline'), cls_id)
    t_p, t_r, t_ap = calc_pr_ap(pjoin(data_root, 'target'), cls_id)
    q10_p, q10_r, q10_ap = calc_pr_ap(pjoin(data_root, 'q10'), cls_id)

    print('Baseline average precision-recall score: {0:0.3f}'.format(bl_ap))
    print('Target average precision-recall score: {0:0.3f}'.format(t_ap))
    print('Q10 average precision-recall score: {0:0.3f}'.format(q10_ap))

    plt.plot(bl_p, bl_r, 'k:', linewidth=1.3, label='baseline')
    plt.plot(t_p, t_r, 'k-', linewidth=1.5, label='target')
    plt.plot(q10_p, q10_r, 'k--', linewidth=1.5, label='q10_denos')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.title("Precision Recall Curves - scorpion")

    plt.show()


def main_pair():
    data_root = "../data/conf_score"

    plt.figure(figsize=(14, 5))

    p1 = plt.subplot(1, 2, 1)

    cls_id = 0
    bl_p, bl_r, bl_ap = calc_pr_ap(pjoin(data_root, 'baseline'), cls_id)
    t_p, t_r, t_ap = calc_pr_ap(pjoin(data_root, 'target'), cls_id)
    q10_p, q10_r, q10_ap = calc_pr_ap(pjoin(data_root, 'q10'), cls_id)

    p1.plot(bl_p, bl_r, 'r:', linewidth=1.5, label='baseline')
    p1.plot(t_p, t_r, 'g-', linewidth=1.5, label='target')
    p1.plot(q10_p, q10_r, 'b--', linewidth=1.5, label='q10_denos')

    p1.set_xlabel('Recall')
    p1.set_ylabel('Precision')
    p1.set_ylim([0.0, 1.05])
    p1.set_xlim([0.0, 1.05])
    p1.legend(loc='lower left')
    p1.set_title("tench")

    p2 = plt.subplot(1, 2, 2)

    cls_id = 1
    bl_p, bl_r, bl_ap = calc_pr_ap(pjoin(data_root, 'baseline'), cls_id)
    t_p, t_r, t_ap = calc_pr_ap(pjoin(data_root, 'target'), cls_id)
    q10_p, q10_r, q10_ap = calc_pr_ap(pjoin(data_root, 'q10'), cls_id)

    p2.plot(bl_p, bl_r, 'r:', linewidth=1.5, label='baseline')
    p2.plot(t_p, t_r, 'g-', linewidth=1.5, label='target')
    p2.plot(q10_p, q10_r, 'b--', linewidth=1.5, label='q10_denos')

    p2.set_xlabel('Recall')
    p2.set_ylabel('Precision')
    p2.set_ylim([0.0, 1.05])
    p2.set_xlim([0.0, 1.05])
    p2.legend(loc='lower left')
    p2.set_title("bulbul")

    plt.suptitle("Precision Recall Curves",  fontsize=16)
    plt.show()


main_pair()

# calc_mAP()
