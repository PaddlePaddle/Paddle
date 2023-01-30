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
from functools import reduce

from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
import paddle.distributed.fleet as fleet
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
from functools import reduce
from test_dist_base import TestDistRunnerBase, runtime_main
import paddle.distributed.fleet as fleet
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()

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
<<<<<<< HEAD
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)
        ),
    )
=======
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
<<<<<<< HEAD
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.01)
        ),
    )
=======
        param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.01)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
<<<<<<< HEAD
    scale = (2.0 / (param_shape[0] ** 2 * SIZE)) ** 0.5

    with fluid.device_guard("gpu:1"):
        predict = paddle.static.nn.fc(
            x=conv_pool_2,
            size=SIZE,
            activation="softmax",
            weight_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)
            ),
        )
        # To cover @RENAMED@GRADIENT
        predict2 = paddle.static.nn.fc(
            x=conv_pool_1,
            size=SIZE,
            activation="softmax",
            weight_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)
            ),
        )
=======
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    with fluid.device_guard("gpu:1"):
        predict = fluid.layers.fc(
            input=conv_pool_2,
            size=SIZE,
            act="softmax",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)))
        # To cover @RENAMED@GRADIENT
        predict2 = fluid.layers.fc(
            input=conv_pool_1,
            size=SIZE,
            act="softmax",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        predict += predict2
    return predict


class TestDistMnist2x2(TestDistRunnerBase):
<<<<<<< HEAD
    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        with fluid.device_guard("gpu:0"):
            images = paddle.static.data(
                name='pixel', shape=[-1, 1, 28, 28], dtype=DTYPE
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
=======

    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        with fluid.device_guard("gpu:0"):
            images = fluid.layers.data(name='pixel',
                                       shape=[1, 28, 28],
                                       dtype=DTYPE)
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            if dist_strategy:
                data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[images, label],
                    capacity=64,
                    use_double_buffer=False,
<<<<<<< HEAD
                    iterable=False,
                )
            # Train program
            predict = cnn_model(images)
        with fluid.device_guard("gpu:1"):
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
=======
                    iterable=False)
            # Train program
            predict = cnn_model(images)
        with fluid.device_guard("gpu:1"):
            cost = fluid.layers.cross_entropy(input=predict, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            avg_cost = paddle.mean(x=cost)

        # Evaluator
        with fluid.device_guard("gpu:1"):
<<<<<<< HEAD
            batch_size_tensor = paddle.tensor.create_tensor(dtype='int64')
            batch_acc = paddle.static.accuracy(
                input=predict, label=label, total=batch_size_tensor
            )
=======
            batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
            batch_acc = fluid.layers.accuracy(input=predict,
                                              label=label,
                                              total=batch_size_tensor)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        inference_program = fluid.default_main_program().clone()
        base_lr = self.lr
        passes = [30, 60, 80, 90]
        steps_per_pass = 10
        bd = [steps_per_pass * p for p in passes]
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        lr_val = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
        opt = fluid.optimizer.Momentum(
            learning_rate=lr_val,
            momentum=0.9,
<<<<<<< HEAD
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        )
=======
            grad_clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        acc_steps = 2  # accumulated steps for pipeline
        if dist_strategy:
            # Reader
<<<<<<< HEAD
            train_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size
            )
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size
            )
=======
            train_reader = paddle.batch(paddle.dataset.mnist.test(),
                                        batch_size=batch_size)
            test_reader = paddle.batch(paddle.dataset.mnist.test(),
                                       batch_size=batch_size)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            fleet.init(is_collective=True)
            strategy = fleet.DistributedStrategy()
            strategy.pipeline = True
            strategy.amp = True
            strategy.pipeline_configs = {
                'micro_batch_size': batch_size,
                'schedule_mode': 'F-then-B',
<<<<<<< HEAD
                'accumulate_steps': acc_steps,
            }
            dist_opt = fleet.distributed_optimizer(
                optimizer=opt, strategy=strategy
            )
=======
                'accumulate_steps': acc_steps
            }
            dist_opt = fleet.distributed_optimizer(optimizer=opt,
                                                   strategy=strategy)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dist_opt.minimize(avg_cost)
        else:
            opt.minimize(avg_cost)
            # Reader
<<<<<<< HEAD
            train_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size * acc_steps
            )
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size * acc_steps
            )

        if dist_strategy:
            return (
                inference_program,
                avg_cost,
                train_reader,
                test_reader,
                batch_acc,
                predict,
                data_loader,
            )
        else:
            return (
                inference_program,
                avg_cost,
                train_reader,
                test_reader,
                batch_acc,
                predict,
            )
=======
            train_reader = paddle.batch(paddle.dataset.mnist.test(),
                                        batch_size=batch_size * acc_steps)
            test_reader = paddle.batch(paddle.dataset.mnist.test(),
                                       batch_size=batch_size * acc_steps)

        if dist_strategy:
            return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict, data_loader
        else:
            return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    runtime_main(TestDistMnist2x2)
