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
import sys

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
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.collective import DistFCConfig
from paddle.fluid import unique_name
from paddle.fluid.layers import dist_algo

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
        loss = dist_algo._distributed_softmax_classify(
            x=emb, label=label, class_num=SIZE, nranks=nranks, rank_id=rank_id)
    elif loss_type == "arcface":
        input_norm = fluid.layers.sqrt(
            fluid.layers.reduce_sum(
                fluid.layers.square(emb), dim=1))
        input = fluid.layers.elementwise_div(emb, input_norm, axis=0)

        weight = fluid.layers.create_parameter(
            shape=[input.shape[1], SIZE],
            dtype='float32',
            name=unique_name.generate('final_fc_w'),
            attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)))

        weight_norm = fluid.layers.sqrt(
            fluid.layers.reduce_sum(
                fluid.layers.square(weight), dim=0))
        weight = fluid.layers.elementwise_div(weight, weight_norm, axis=1)
        cos = fluid.layers.mul(input, weight)

        theta = fluid.layers.acos(cos)
        margin_cos = fluid.layers.cos(theta + 0.5)
        one_hot = fluid.layers.one_hot(label, SIZE)
        diff = (margin_cos - cos) * one_hot
        target_cos = cos + diff
        logit = fluid.layers.scale(target_cos, scale=64.)

        loss, prob = fluid.layers.softmax_with_cross_entropy(
            logits=logit, label=label, return_softmax=True)
        loss = fluid.layers.mean(x=loss)
        one_hot.stop_gradient = True
    elif loss_type == "dist_arcface":
        loss = dist_algo._distributed_arcface_classify(
            x=emb,
            label=label,
            class_num=SIZE,
            nranks=nranks,
            rank_id=rank_id,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)))
    else:
        raise ValueError("Unknown loss type: {}.".format(loss_type))

    return loss


class TestDistMnist2x2DistFC(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)

        loss_type = self._distfc_loss_type
        # Train program
        avg_cost = cnn_model(images, label, loss_type, self.worker_index,
                             self.worker_num)
        # Evaluator
        predict = 1.0
        batch_acc = 1.0

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
            dist_strategy.mode = "collective"
            dist_strategy.collective_mode = "grad_allreduce"
            dist_strategy.dist_fc_config = DistFCConfig(batch_size)
            dist_opt = fleet.distributed_optimizer(
                optimizer=opt, strategy=dist_strategy)
            dist_opt.minimize(avg_cost)
        else:
            opt.minimize(avg_cost)
        return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict


if __name__ == "__main__":
    runtime_main(TestDistMnist2x2DistFC)
