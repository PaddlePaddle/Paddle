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
from functools import reduce
from test_dist_base import TestDistRunnerBase, runtime_main

DTYPE = "float32"
paddle.dataset.imdb.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def conv_net(input,
             dict_dim,
             emb_dim=128,
             window_size=3,
             num_filters=128,
             fc0_dim=96,
             class_dim=2):
    emb = fluid.layers.embedding(
        input=input, size=[dict_dim, emb_dim], is_sparse=False)

    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=num_filters,
        filter_size=window_size,
        act="tanh",
        pool_type="max")

    fc_0 = fluid.layers.fc(input=[conv_3], size=fc0_dim)
    prediction = fluid.layers.fc(input=[fc_0], size=class_dim, act="softmax")
    return prediction


def inference_network(dict_dim):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    out = conv_net(data, dict_dim)
    return out


def get_reader(word_dict, batch_size):
    # The training data set.
    train_reader = paddle.batch(
        paddle.dataset.imdb.train(word_dict), batch_size=batch_size)

    # The testing data set.
    test_reader = paddle.batch(
        paddle.dataset.imdb.test(word_dict), batch_size=batch_size)

    return train_reader, test_reader


def get_optimizer(learning_rate):
    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    return optimizer


class TestDistTextClassification2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2):
        word_dict = paddle.dataset.imdb.word_dict()
        dict_dim = word_dict["<unk>"]

        # Input data
        data = fluid.layers.data(
            name="words", shape=[1], dtype="int64", lod_level=1)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        # Train program
        predict = conv_net(data, dict_dim)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=predict, label=label)

        inference_program = fluid.default_main_program().clone()
        # Optimization
        opt = get_optimizer(learning_rate=0.001)
        opt.minimize(avg_cost)

        # Reader
        train_reader, test_reader = get_reader(word_dict, batch_size)

        return inference_program, avg_cost, train_reader, test_reader, acc, predict


if __name__ == "__main__":
    runtime_main(TestDistTextClassification2x2)
