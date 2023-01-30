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

<<<<<<< HEAD
import os
import re
import string
import tarfile

from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
import paddle.fluid as fluid
=======
from __future__ import print_function

import numpy as np
import argparse
import time
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import unittest
from multiprocessing import Process
import os
import signal
import six
import tarfile
import string
import re
from functools import reduce
from test_dist_base import TestDistRunnerBase, runtime_main
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

DTYPE = "float32"
VOCAB_URL = 'http://paddle-dist-ce-data.bj.bcebos.com/imdb.vocab'
VOCAB_MD5 = '23c86a0533c0151b6f12fa52b106dcc2'
DATA_URL = 'http://paddle-dist-ce-data.bj.bcebos.com/text_classification.tar.gz'
DATA_MD5 = '29ebfc94f11aea9362bbb7f5e9d86b8a'


# Load dictionary.
def load_vocab(filename):
    vocab = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab


def get_worddict(dict_path):
    word_dict = load_vocab(dict_path)
    word_dict["<unk>"] = len(word_dict)
    dict_dim = len(word_dict)
    return word_dict, dict_dim


<<<<<<< HEAD
def conv_net(
    input,
    dict_dim,
    emb_dim=128,
    window_size=3,
    num_filters=128,
    fc0_dim=96,
    class_dim=2,
):
=======
def conv_net(input,
             dict_dim,
             emb_dim=128,
             window_size=3,
             num_filters=128,
             fc0_dim=96,
             class_dim=2):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    emb = fluid.layers.embedding(
        input=input,
        size=[dict_dim, emb_dim],
        is_sparse=False,
<<<<<<< HEAD
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)
        ),
    )
=======
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=num_filters,
        filter_size=window_size,
        act="tanh",
        pool_type="max",
<<<<<<< HEAD
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)
        ),
    )

    fc_0 = paddle.static.nn.fc(
        x=[conv_3],
        size=fc0_dim,
        weight_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)
        ),
    )

    prediction = paddle.static.nn.fc(
        x=[fc_0],
        size=class_dim,
        activation="softmax",
        weight_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)
        ),
    )
=======
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))

    fc_0 = fluid.layers.fc(
        input=[conv_3],
        size=fc0_dim,
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))

    prediction = fluid.layers.fc(
        input=[fc_0],
        size=class_dim,
        act="softmax",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return prediction


def inference_network(dict_dim):
<<<<<<< HEAD
    data = paddle.static.data(
        name="words", shape=[-1, 1], dtype="int64", lod_level=1
    )
=======
    data = fluid.layers.data(name="words",
                             shape=[1],
                             dtype="int64",
                             lod_level=1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    out = conv_net(data, dict_dim)
    return out


def get_reader(word_dict, batch_size):
    # The training data set.
    train_reader = paddle.batch(train(word_dict), batch_size=batch_size)

    # The testing data set.
    test_reader = paddle.batch(test(word_dict), batch_size=batch_size)

    return train_reader, test_reader


def get_optimizer(learning_rate):
    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    return optimizer


class TestDistTextClassification2x2(TestDistRunnerBase):
<<<<<<< HEAD
    def get_model(self, batch_size=2):
        vocab = os.path.join(
            paddle.dataset.common.DATA_HOME, "text_classification", "imdb.vocab"
        )
        word_dict, dict_dim = get_worddict(vocab)

        # Input data
        data = paddle.static.data(
            name="words", shape=[-1, 1], dtype="int64", lod_level=1
        )
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

        # Train program
        predict = conv_net(data, dict_dim)
        cost = paddle.nn.functional.cross_entropy(
            input=predict, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)
        acc = paddle.static.accuracy(input=predict, label=label)
=======

    def get_model(self, batch_size=2):
        vocab = os.path.join(paddle.dataset.common.DATA_HOME,
                             "text_classification", "imdb.vocab")
        word_dict, dict_dim = get_worddict(vocab)

        # Input data
        data = fluid.layers.data(name="words",
                                 shape=[1],
                                 dtype="int64",
                                 lod_level=1)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        # Train program
        predict = conv_net(data, dict_dim)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = paddle.mean(x=cost)
        acc = fluid.layers.accuracy(input=predict, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        inference_program = fluid.default_main_program().clone()

        # Optimization
        opt = get_optimizer(learning_rate=0.001)
        opt.minimize(avg_cost)

        # Reader
        train_reader, test_reader = get_reader(word_dict, batch_size)

<<<<<<< HEAD
        return (
            inference_program,
            avg_cost,
            train_reader,
            test_reader,
            acc,
            predict,
        )
=======
        return inference_program, avg_cost, train_reader, test_reader, acc, predict
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def tokenize(pattern):
    """
    Read files that match the given pattern.  Tokenize and yield each file.
    """

    with tarfile.open(
<<<<<<< HEAD
        paddle.dataset.common.download(
            DATA_URL, 'text_classification', DATA_MD5
        )
    ) as tarf:
=======
            paddle.dataset.common.download(DATA_URL, 'text_classification',
                                           DATA_MD5)) as tarf:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # Note that we should use tarfile.next(), which does
        # sequential access of member files, other than
        # tarfile.extractfile, which does random access and might
        # destroy hard disks.
        tf = tarf.next()
<<<<<<< HEAD
        while tf is not None:
            if bool(pattern.match(tf.name)):
                # newline and punctuations removal and ad-hoc tokenization.
                yield tarf.extractfile(tf).read().rstrip(b'\n\r').translate(
                    None, string.punctuation.encode('latin-1')
                ).lower().split()
=======
        while tf != None:
            if bool(pattern.match(tf.name)):
                # newline and punctuations removal and ad-hoc tokenization.
                yield tarf.extractfile(tf).read().rstrip(
                    six.b("\n\r")).translate(None, six.b(
                        string.punctuation)).lower().split()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            tf = tarf.next()


def reader_creator(pos_pattern, neg_pattern, word_idx):
    UNK = word_idx['<unk>']
    INS = []

    def load(pattern, out, label):
        for doc in tokenize(pattern):
            out.append(([word_idx.get(w, UNK) for w in doc], label))

    load(pos_pattern, INS, 0)
    load(neg_pattern, INS, 1)

    def reader():
        for doc, label in INS:
            yield doc, label

    return reader


def train(word_idx):
    """
    IMDB training set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    sequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Training reader creator
    :rtype: callable
    """
<<<<<<< HEAD
    return reader_creator(
        re.compile(r"train/pos/.*\.txt$"),
        re.compile(r"train/neg/.*\.txt$"),
        word_idx,
    )
=======
    return reader_creator(re.compile(r"train/pos/.*\.txt$"),
                          re.compile(r"train/neg/.*\.txt$"), word_idx)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def test(word_idx):
    """
    IMDB test set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    sequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Test reader creator
    :rtype: callable
    """
<<<<<<< HEAD
    return reader_creator(
        re.compile(r"test/pos/.*\.txt$"),
        re.compile(r"test/neg/.*\.txt$"),
        word_idx,
    )
=======
    return reader_creator(re.compile(r"test/pos/.*\.txt$"),
                          re.compile(r"test/neg/.*\.txt$"), word_idx)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    paddle.dataset.common.download(VOCAB_URL, 'text_classification', VOCAB_MD5)
    paddle.dataset.common.download(DATA_URL, 'text_classification', DATA_MD5)
    runtime_main(TestDistTextClassification2x2)
