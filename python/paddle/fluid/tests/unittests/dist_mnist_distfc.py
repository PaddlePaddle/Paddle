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
import six
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
from paddle.fluid.layers import dist_algo
from test_dist_base import TestDistRunnerBase, runtime_main
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.collective import DistFCConfig

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def cnn_model(data, label, loss_type, rank_id, nranks):
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
    emb = fluid.layers.fc(input=conv_pool_2,
                          size=50,
                          act=None,
                          param_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.Constant(
                                  value=0.01)))

    SIZE = 10
    if loss_type == "softmax":
        out = fluid.layers.fc(
            input=emb,
            size=SIZE,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)))
        loss = fluid.layers.softmax_with_cross_entropy(logits=out, label=label)
        loss = fluid.layers.mean(x=loss)
    elif loss_type == "dist_softmax":
        #out = fluid.layers.fc(
        #    input=emb,
        #    size=SIZE,
        #    act=None,
        #    param_attr=fluid.param_attr.ParamAttr(
        #        initializer=fluid.initializer.Constant(value=0.01)))
        #loss = fluid.layers.softmax_with_cross_entropy(logits=out, label=label)
        #loss = fluid.layers.mean(x=loss)

        loss = dist_algo._distributed_softmax_classify(
            x=emb, label=label, class_num=SIZE, nranks=nranks, rank_id=rank_id)
        #param_attr=fluid.param_attr.ParamAttr(
        #    initializer=fluid.initializer.Constant(value=0.01)))
    elif loss_type == "dist_arcface":
        loss = dist_algo._distributed_arcface_classify(
            x=emb,
            label=label,
            class_num=SIZE,
            nranks=nranks,
            rank_id=rank_id,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)))
        loss = loss[0]
    else:
        raise ValueError("Unknown loss type: {}.".format(loss_type))

    return loss


class TestDistMnist2x2DistFC(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        loss_type = self._distfc_loss_type
        # Train program
        avg_cost = cnn_model(images, label, loss_type, self.worker_index,
                             self.worker_num)
        #avg_cost = fluid.layers.mean(x=cost)

        # Evaluator
        #batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
        #batch_acc = fluid.layers.accuracy(
        #    input=predict, label=label, total=batch_size_tensor)
        predict = None
        batch_acc = None

        inference_program = fluid.default_main_program().clone()
        # Optimization
        # TODO(typhoonzero): fix distributed adam optimizer
        # opt = fluid.optimizer.AdamOptimizer(
        #     learning_rate=0.001, beta1=0.9, beta2=0.999)
        opt = fluid.optimizer.Momentum(learning_rate=self.lr, momentum=0.9)

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)

        if dist_strategy:
            dist_strategy.use_dist_fc = True
            dist_opt = fleet.distributed_optimizer(
                optimizer=opt, strategy=dist_strategy)
            dist_opt.minimize(avg_cost)
        else:
            opt.minimize(avg_cost)

        return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict


if __name__ == "__main__":
    runtime_main(TestDistMnist2x2DistFC)
