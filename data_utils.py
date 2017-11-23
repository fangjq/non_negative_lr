# -*- coding: utf-8 -*- #
# Author: Jiaquan Fang
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pandas as pd
from tensorflow.contrib.data import Dataset
import tensorflow as tf


def get_dataset(features, targets=None,
                shuffle=True, n_epochs=1, batch_size=None):
    if targets is not None:
        dataset = Dataset.from_tensor_slices((tf.constant(features, tf.float32),
                                              tf.constant(targets, tf.float32)))
    else:
        dataset = Dataset.from_tensor_slices(tf.constant(features, tf.float32))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=64, seed=27)
    if n_epochs > 1:
        dataset = dataset.repeat(n_epochs)
    if batch_size is None:
        dataset = dataset.batch(features.shape[0])
    else:
        dataset = dataset.batch(batch_size)

    return dataset


class CSVDataHelper(object):

    def __init__(self, filename, sep=",", target_col="label"):
        data_df = pd.read_csv(filename, sep=sep, index_col=0)
        feature_cols = [col for col in data_df if col != target_col]
        self._features = data_df[feature_cols].as_matrix()
        self._targets = data_df[target_col].as_matrix()
        self._num_examples = self._features.shape[0]
        self._num_features = self._features.shape[1]

        self._index_in_epoch = 0
        self._epochs_completed = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_features(self):
        return self._num_features

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def features(self):
        return self._features

    @property
    def targets(self):
        return self._targets

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._features = self._features[perm0]
            self._targets = self._targets[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            targets_rest_part = self._targets[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._features = self._features[perm]
                self._targets = self._targets[perm]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            features_new_part = self._features[start:end]
            targets_new_part = self._targets[start:end]
            return (np.concatenate((features_rest_part, features_new_part),
                                   axis=0),
                    np.concatenate((targets_rest_part, targets_new_part),
                                   axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._targets[start:end]
