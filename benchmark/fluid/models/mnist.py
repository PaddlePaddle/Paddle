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
import cProfile
import os

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

SEED = 1
DTYPE = "float32"

# random seed must set before configuring the network.
# fluid.default_startup_program().random_seed = SEED


def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    # TODO(dzhwinter) : refine the initializer and random seed settting
    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=SIZE,
        act="softmax",
        param_attr=fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=scale)))
    return predict


def get_model(args, is_train, main_prog, startup_prog):
    # NOTE: mnist is small, we don't implement data sharding yet.
    opt = None
    data_file_handle = None
    with fluid.program_guard(main_prog, startup_prog):
        if args.use_reader_op:
            filelist = [
                os.path.join(args.data_path, f)
                for f in os.listdir(args.data_path)
            ]
            data_file_handle = fluid.layers.open_files(
                filenames=filelist,
                shapes=[[-1, 1, 28, 28], (-1, 1)],
                lod_levels=[0, 0],
                dtypes=["float32", "int64"],
                thread_num=1,
                pass_num=1)
            data_file = fluid.layers.double_buffer(
                fluid.layers.batch(
                    data_file_handle, batch_size=args.batch_size))
        with fluid.unique_name.guard():
            if args.use_reader_op:
                input, label = fluid.layers.read_file(data_file)
            else:
                images = fluid.layers.data(
                    name='pixel', shape=[1, 28, 28], dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')

            predict = cnn_model(images)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            # Evaluator
            batch_acc = fluid.layers.accuracy(input=predict, label=label)
            # Optimization
            if is_train:
                opt = fluid.optimizer.AdamOptimizer(
                    learning_rate=0.001, beta1=0.9, beta2=0.999)
                opt.minimize(avg_cost)
                if args.memory_optimize:
                    fluid.memory_optimize(main_prog)

    # Reader
    if is_train:
        reader = paddle.dataset.mnist.train()
    else:
        reader = paddle.dataset.mnist.test()
    batched_reader = paddle.batch(
        reader, batch_size=args.batch_size * args.gpus)
    return avg_cost, opt, [batch_acc], batched_reader, data_file_handle
