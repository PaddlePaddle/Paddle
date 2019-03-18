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
import sys
import signal
from test_dist_base import TestDistRunnerBase, runtime_main
from dist_se_resnext import train_parameters, SE_ResNeXt

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def test_merge_reader(repeat_batch_size=8):
    orig_reader = paddle.dataset.flowers.test(use_xmap=False)
    record_batch = []
    b = 0
    for d in orig_reader():
        if b >= repeat_batch_size:
            break
        record_batch.append(d)
        b += 1
    while True:
        for d in record_batch:
            yield d


class DistSeResneXt2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2):
        # Input data
        image = fluid.layers.data(
            name="data", shape=[3, 224, 224], dtype='float32')
        label = fluid.layers.data(name="int64", shape=[1], dtype='int64')

        # Train program
        model = SE_ResNeXt(layers=50)
        out = model.net(input=image, class_dim=102)
        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

        # Evaluator
        test_program = fluid.default_main_program().clone(for_test=True)

        # Optimization
        total_images = 6149  # flowers
        epochs = [30, 60, 90]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in epochs]
        base_lr = 0.1
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
        optimizer.minimize(avg_cost)

        # Reader
        train_reader = paddle.batch(test_merge_reader, batch_size=batch_size)
        test_reader = paddle.batch(test_merge_reader, batch_size=batch_size)

        return test_program, avg_cost, train_reader, test_reader, acc_top1, out


if __name__ == "__main__":
    runtime_main(DistSeResneXt2x2)
