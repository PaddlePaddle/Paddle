#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import tensorflow as tf
import paddle.v2 as paddle

DTYPE = tf.float32


def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=5, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    args = parser.parse_args()
    return args


def run_benchmark(args):
    def weight_variable(dtype, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=dtype)
        return tf.Variable(initial)

    def bias_variable(dtype, shape):
        initial = tf.constant(0.1, shape=shape, dtype=dtype)
        return tf.Variable(initial)

    device = '/cpu:0' if args.device == 'CPU' else '/device:GPU:0'
    with tf.device(device):
        images = tf.placeholder(DTYPE, shape=(None, 28, 28, 1))
        labels = tf.placeholder(tf.int64, shape=(None, ))

        # conv1, relu, pool1
        conv1_weights = weight_variable(DTYPE, [5, 5, 1, 20])
        conv1_bias = bias_variable(DTYPE, [20])
        conv1 = tf.nn.conv2d(
            images, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # conv2, relu, pool2
        conv2_weights = weight_variable(DTYPE, [5, 5, 20, 50])
        conv2_bias = bias_variable(DTYPE, [50])
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        # FC 
        pool_shape = pool2.get_shape().as_list()
        hidden_dim = reduce(lambda a, b: a * b, pool_shape[1:], 1)
        reshape = tf.reshape(pool2, shape=(tf.shape(pool2)[0], hidden_dim))
        fc_weights = weight_variable(DTYPE, [hidden_dim, 10])
        fc_bias = bias_variable(DTYPE, [10])
        logits = tf.matmul(reshape, fc_weights) + fc_bias

        # Get prediction
        prediction = tf.nn.softmax(logits)

        # Loss 
        one_hot_labels = tf.one_hot(labels, depth=10)
        cost = -tf.reduce_sum(tf.log(prediction) * one_hot_labels, [1])
        avg_cost = tf.reduce_mean(cost)

        # Get accuracy
        correct = tf.equal(tf.argmax(prediction, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # metrics, g_accuracy
        with tf.variable_scope("reset_metrics_accuracy_scope") as scope:
            g_accuracy = tf.metrics.accuracy(
                labels, tf.argmax(
                    prediction, axis=1))
            vars = tf.contrib.framework.get_variables(
                scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            g_accuracy_reset_op = tf.variables_initializer(vars)

        # Optimizer 
        opt = tf.train.AdamOptimizer(
            learning_rate=0.001, beta1=0.9, beta2=0.999)
        train_op = opt.minimize(avg_cost)
        # train_op = tf.train.AdamOptimizer(1e-4).minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=args.batch_size)

    def eval_test():
        sess.run(g_accuracy_reset_op)
        for batch_id, data in enumerate(test_reader()):
            images_data = np.array(
                map(lambda x: np.transpose(x[0].reshape([1, 28, 28]), axes=[1,2,0]), data)).astype("float32")
            labels_data = np.array(map(lambda x: x[1], data)).astype("int64")

            loss, acc, g_acc = sess.run(
                [avg_cost, accuracy, g_accuracy],
                feed_dict={images: images_data,
                           labels: labels_data})
        return g_acc[1]

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        for pass_id in range(args.pass_num):
            sess.run(g_accuracy_reset_op)

            pass_start = time.time()
            for batch_id, data in enumerate(train_reader()):
                images_data = np.array(
                    map(lambda x: np.transpose(x[0].reshape([1, 28, 28]), axes=[1,2,0]), data)).astype("float32")
                labels_data = np.array(map(lambda x: x[1], data)).astype(
                    "int64")

                start = time.time()
                _, loss, acc, g_acc = sess.run(
                    [train_op, avg_cost, accuracy, g_accuracy],
                    feed_dict={images: images_data,
                               labels: labels_data})
                end = time.time()

                print("pass=%d, batch=%d, loss=%f, error=%f, elapse=%f" %
                      (pass_id, batch_id, loss, 1 - acc, (end - start) / 1000))

            pass_end = time.time()
            test_avg_acc = eval_test()

            print(
                "pass=%d, training_avg_accuracy=%f, test_avg_acc=%f, elapse=%f"
                % (pass_id, g_acc[1], test_avg_acc,
                   (pass_end - pass_start) / 1000))


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    run_benchmark(args)
