# -*- coding: utf-8 -*- #
# Author: Jiaquan Fang
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import argparse
from data_utils import get_dataset
from model import build_model
import time
import os
from sklearn import metrics
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path",
                        default="data/train_set.npz")
    parser.add_argument("--dev-data-path",
                        default="data/test_set.npz")
    parser.add_argument("--evaluate-every", default=500)
    parser.add_argument("--model-dir", default="./model_dir")
    parser.add_argument("--n-epochs", default=30)
    parser.add_argument("--batch-size", default=100)
    args = parser.parse_args()

    train_data = np.load(args.train_data_path)
    dev_data = np.load(args.dev_data_path)
    X_train = train_data["features"]
    y_train = train_data["labels"].reshape(-1, 1)
    X_dev = dev_data["features"]
    y_dev = dev_data["labels"].reshape(-1, 1)

    train_set = get_dataset(X_train, y_train,
                            n_epochs=args.n_epochs,
                            batch_size=args.batch_size)
    dev_set = get_dataset(X_dev, y_dev,
                          shuffle=False)

    data_iter = tf.contrib.data.Iterator.from_structure(train_set.output_types,
                                                        train_set.output_shapes)
    train_init = data_iter.make_initializer(train_set)
    dev_init = data_iter.make_initializer(dev_set)

    # Initialize model path
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(args.model_dir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Build model
    batch_X, batch_y = data_iter.get_next()
    model_spec = build_model(batch_X, batch_y, l1_reg=1.0, lr=0.5)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    with tf.Session() as sess:
        init.run()
        train_init.run()
        while True:
            try:
                _, _, loss, step = sess.run([model_spec.train_op,
                                             model_spec.clip,
                                             model_spec.loss,
                                             model_spec.global_step])
                if step % args.evaluate_every == 0:
                    print("loss: {}".format(loss))
            except tf.errors.OutOfRangeError:
                print("Finished training")
                break

        dev_init.run()
        while True:
            try:
                score = sess.run(model_spec.score)
            except tf.errors.OutOfRangeError:
                print(metrics.roc_auc_score(y_dev, score))
                feature_cols = []
                with open("data/feature_cols.txt", "r") as f:
                    for line in f:
                        feature_cols.append(line.strip("\n"))
                W = sess.run(model_spec.W)
                b = sess.run(model_spec.b)
                params = pd.Series(W.reshape(-1), index=feature_cols)
                params = params[params != 0]
                params["intercept"] = b[0]
                params.to_csv("params.csv")
                break
