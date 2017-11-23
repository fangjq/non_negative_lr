# -*- coding: utf-8 -*- #
# Author: Jiaquan Fang
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from collections import namedtuple


ModelSpec = namedtuple("ModelSpec", ["train_op", "clip", "loss",
                                     "global_step", "W", "b",
                                     "score"])


def build_model(X, y, l1_reg=0.0, l2_reg=0.0, lr=0.01):
    with tf.variable_scope("params"):
        W = tf.get_variable("W", shape=(X.shape[1], 1),
                            initializer=tf.zeros_initializer(dtype=tf.float32))
        b = tf.get_variable("b", shape=(1),
                            initializer=tf.zeros_initializer(dtype=tf.float32))

    with tf.variable_scope("outputs"):
        logits = tf.matmul(X, W) + b
        score = tf.sigmoid(logits)

    with tf.variable_scope("loss"):
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels=y, logits=logits, name="xent")
        loss = tf.reduce_mean(xent)

    with tf.variable_scope("train"):
        clip = W.assign(tf.maximum(0., W))
        optimizer = tf.train.FtrlOptimizer(lr,
                                           l1_regularization_strength=l1_reg,
                                           l2_regularization_strength=l2_reg)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step)

    return ModelSpec(train_op=train_op, clip=clip, loss=loss,
                     global_step=global_step, W=W, b=b,
                     score=score)
