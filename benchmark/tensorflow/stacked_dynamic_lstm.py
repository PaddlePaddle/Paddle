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

import numpy as np
import argparse
import time
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser("LSTM model benchmark.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--stacked_num',
        type=int,
        default=5,
        help='Number of lstm layers to stack. (default: %(default)d)')
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=512,
        help='Dimension of embedding table. (default: %(default)d)')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512,
        help='Hidden size of lstm unit. (default: %(default)d)')
    parser.add_argument(
        '--pass_num',
        type=int,
        default=10,
        help='Epoch number to train. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0002,
        help='Learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def dynamic_lstm_model(dict_size,
                       embedding_dim,
                       hidden_dim,
                       stacked_num,
                       class_num=2,
                       is_train=True):
    word_idx = tf.placeholder(tf.int64, shape=[None, None])
    sequence_length = tf.placeholder(tf.int64, shape=[None, ])

    embedding_weights = tf.get_variable('word_embeddings',
                                        [dict_size, embedding_dim])
    embedding = tf.nn.embedding_lookup(embedding_weights, word_idx)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=hidden_dim, use_peepholes=False)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * stacked_num)

    # final_state [LSTMTuple(c, h), LSTMTuple(c, h) ...] total stacked_num LSTMTuples
    _, final_state = tf.nn.dynamic_rnn(
        cell=stacked_cell,
        inputs=embedding,
        dtype=tf.float32,
        sequence_length=sequence_length)

    w = tf.Variable(
        tf.truncated_normal([hidden_dim, class_num]), dtype=tf.float32)
    bias = tf.Variable(
        tf.constant(
            value=0.0, shape=[class_num], dtype=tf.float32))
    prediction = tf.matmul(final_state[-1][1], w) + bias

    if not is_train:
        return (word_idx, sequence_length), tf.nn.softmax(prediction)

    label = tf.placeholder(tf.int64, shape=[None, ])
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(label, 2), logits=prediction)
    avg_loss = tf.reduce_mean(loss)

    correct_count = tf.equal(tf.argmax(prediction, 1), label)
    acc = tf.reduce_mean(tf.cast(correct_count, tf.float32))

    with tf.variable_scope("reset_metrics_accuracy_scope") as scope:
        g_acc = tf.metrics.accuracy(label, tf.argmax(prediction, axis=1))
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)

    return (word_idx, sequence_length, label), avg_loss, acc, g_acc, reset_op


def padding_data(data, padding_size, value):
    data = data + [value] * padding_size
    return data[:padding_size]


def train(args):
    word_dict = paddle.dataset.imdb.word_dict()
    dict_size = len(word_dict)

    feeding_list, avg_loss, acc, g_acc, reset_op = dynamic_lstm_model(
        dict_size, args.embedding_dim, args.hidden_dim, args.stacked_num)

    adam_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = adam_optimizer.minimize(avg_loss)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=25000),
        batch_size=args.batch_size)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.test(word_dict), buf_size=25000),
        batch_size=args.batch_size)

    def do_validation(sess):
        sess.run(reset_op)
        for batch_id, data in enumerate(test_reader()):
            word_idx = map(lambda x: x[0], data)
            sequence_length = np.array(
                [len(seq) for seq in word_idx]).astype('int64')
            maxlen = np.max(sequence_length)
            word_idx = [padding_data(seq, maxlen, 0) for seq in word_idx]
            word_idx = np.array(word_idx).astype('int64')
            label = np.array(map(lambda x: x[1], data)).astype('int64')

            _, loss, fetch_acc, fetch_g_acc = sess.run(
                [train_op, avg_loss, acc, g_acc],
                feed_dict={
                    feeding_list[0]: word_idx,
                    feeding_list[1]: sequence_length,
                    feeding_list[2]: label
                })

        return fetch_g_acc[1]

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_l)
        sess.run(init_g)

        for pass_id in xrange(args.pass_num):
            # clear accuracy local variable 
            sess.run(reset_op)
            pass_start_time = time.time()
            words_seen = 0

            for batch_id, data in enumerate(train_reader()):
                word_idx = map(lambda x: x[0], data)
                sequence_length = np.array(
                    [len(seq) for seq in word_idx]).astype('int64')
                words_seen += np.sum(sequence_length)
                maxlen = np.max(sequence_length)
                word_idx = [padding_data(seq, maxlen, 0) for seq in word_idx]
                word_idx = np.array(word_idx).astype('int64')
                label = np.array(map(lambda x: x[1], data)).astype('int64')

                _, loss, fetch_acc, fetch_g_acc = sess.run(
                    [train_op, avg_loss, acc, g_acc],
                    feed_dict={
                        feeding_list[0]: word_idx,
                        feeding_list[1]: sequence_length,
                        feeding_list[2]: label
                    })

                print("pass_id=%d, batch_id=%d, loss: %f, acc: %f, avg_acc: %f"
                      % (pass_id, batch_id, loss, fetch_acc, fetch_g_acc[1]))

            pass_end_time = time.time()
            time_consumed = pass_end_time - pass_start_time
            words_per_sec = words_seen / time_consumed
            test_acc = do_validation(sess)
            print("pass_id=%d, test_acc: %f, words/s: %f, sec/pass: %f" %
                  (pass_id, test_acc, words_per_sec, time_consumed))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)

    if args.infer_only:
        pass
    else:
        train(args)
