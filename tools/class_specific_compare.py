# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def parse_logs(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    loss_arr = []
    acc_arr = []

    for i in xrange(len(lines)):
        if i % 2 == 1:
            loss_arr.append(float(lines[i][20:26]))
            acc_arr.append(float(lines[i][-7:-1]))

    return loss_arr, acc_arr


def get_scale_list(a, b, c):
    # color = ['0', '0.5', '1']
    color = ['c', 'brown', 'b']
    index = np.argsort([a, b, c])
    sorted_color = [color[i] for i in index]

    sorted_val = sorted([a, b, c])

    sorted_val = [sorted_val[0],
                  sorted_val[1] - sorted_val[0],
                  sorted_val[2] - sorted_val[1]]

    return sorted_color, sorted_val


def rerange_list(l1, l2, l3):
    c0 = []
    c1 = []
    c2 = []
    m0 = []
    m1 = []
    m2 = []

    for i in range(len(l1)):
        sc, sv = get_scale_list(l1[i], l2[i], l3[i])
        c0.append(sc[0])
        c1.append(sc[1])
        c2.append(sc[2])
        m0.append(sv[0])
        m1.append(sv[1])
        m2.append(sv[2])

    return c0, c1, c2, m0, m1, m2


def plot_compare_bar():
    index = range(40)

    loss_t, acc_t = parse_logs('target.txt')
    loss_q10, acc_q10 = parse_logs('q10.txt')
    loss_qd, acc_qd = parse_logs('q10_denos.txt')

    width = 0.8

    c0, c1, c2, m0, m1, m2 = rerange_list(acc_t, acc_q10, acc_qd)

    plt.bar(index, m0, width, color=c0, label='Target')
    plt.bar(index, m1, width, bottom=m0, color=c1, label='Q10')
    plt.bar(index, m2, width, bottom=np.array(
        m0) + np.array(m1), color=c2, label='Q10_denos')

    # for i in range(40):

    #     plt.bar(i, [m0])

    #     plt.bar(i, m0[i], width, color=c0[i], label=label_map[c0[i]])
    #     plt.bar(i, m1[i], width, bottom=m0[i],
    #             color=c1[i], label=label_map[c1[i]])
    #     plt.bar(i, m2[i], width, bottom=m0[i] + m1[i],
    #             color=c2[i], label=label_map[c2[i]])

    # r = [i / float(10) for i in range(12)]

    # print r
    # plt.yticks(np.linspace(0, 1.2, 12), )

    plt.ylim(0., 1.2)
    plt.xlabel('Category')
    plt.ylabel('Classification Accuracy')
    plt.legend(loc='best')

    plt.title('Class-specific Accuracy Comparison')
    plt.tight_layout()

    plt.show()


plot_compare_bar()

# a = 0.18
# b = 0.42
# c = 0.15

# sorted_color, sorted_val = get_scale_list(a, b, c)

# print(sorted_color, sorted_val)
