# -*- coding: utf-8
# Author: Xiaoming Qin

"""Logger class for training a neural network"""

from os.path import join as pjoin


class Logger():

    def __init__(self, arch, epochs, batch_size):
        self._model_name = arch
        self._train_log = []
        self._valid_log = []
        self._epochs = epochs
        self._best_epoch = 0
        self._best_prec1 = 0.
        self._batch_size = batch_size

    @property
    def train_log(self):
        return self._train_log

    @property
    def valid_log(self):
        return self._valid_log

    @property
    def epochs(self):
        return self._epochs

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def model_name(self):
        return self._model_name

    def record_trn(self, info):
        self._train_log.append(info)

    def record_val(self, info, epoch):
        self._valid_log.append(info)
        if info[2] > self._best_prec1:
            self._best_prec1 = info[2]
            self._best_epoch = epoch

    def save(self):
        fname = pjoin('../logs', self._model_name + '.log')
        with open(fname, 'wb') as f:
            f.write("model_name: {}\n".format(self._model_name))
            f.write("batch_size: {}\n".format(self._batch_size))
            f.write("num_epochs: {}\n".format(self._epochs))

            f.write('train info: \n')
            for info in self._train_log:
                f.write("{0:d} {1:d} {2:.4f} {3:.4f}\n".format(
                    info[0], info[1], info[2], info[3]))

            f.write('valid info: \n')
            for info in self._valid_log:
                f.write("{0:d} {1:.4f} {2:.4f}\n".format(
                    info[0], info[1], info[2]))
