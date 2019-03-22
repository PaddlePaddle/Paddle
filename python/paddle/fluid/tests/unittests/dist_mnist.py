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
paddle.dataset.mnist.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))

    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=SIZE,
        act="softmax",
        param_attr=fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)))
    return predict


class TestDistMnist2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False):
        # Input data
        images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        # Train program
        predict = cnn_model(images)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        # Evaluator
        batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
        batch_acc = fluid.layers.accuracy(
            input=predict, label=label, total=batch_size_tensor)

        inference_program = fluid.default_main_program().clone()
        # Optimization
        # TODO(typhoonzero): fix distributed adam optimizer
        # opt = fluid.optimizer.AdamOptimizer(
        #     learning_rate=0.001, beta1=0.9, beta2=0.999)
        if not use_dgc:
            opt = fluid.optimizer.Momentum(learning_rate=self.lr, momentum=0.9)
        else:
            opt = fluid.optimizer.DGCMomentumOptimizer(
                learning_rate=self.lr, momentum=0.9, rampup_begin_step=0)

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        opt.minimize(avg_cost)
        return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict


if __name__ == "__main__":
    runtime_main(TestDistMnist2x2)
