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


def get_model(args):
    if args.use_reader_op:
        filelist = [
            os.path.join(args.data_path, f) for f in os.listdir(args.data_path)
        ]
        data_file = fluid.layers.open_files(
            filenames=filelist,
            shapes=[[-1, 1, 28, 28], (-1, 1)],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            thread_num=args.gpus,
            pass_num=args.pass_num)
        data_file = fluid.layers.double_buffer(
            fluid.layers.batch(
                data_file, batch_size=args.batch_size))
        images, label = fluid.layers.read_file(data_file)
    else:
        images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if args.device == 'CPU' and args.cpus > 1:
        places = fluid.layers.get_places(args.cpus)
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            predict = cnn_model(pd.read_input(images))
            label = pd.read_input(label)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            batch_acc = fluid.layers.accuracy(input=predict, label=label)

            pd.write_output(avg_cost)
            pd.write_output(batch_acc)

        avg_cost, batch_acc = pd()
        avg_cost = fluid.layers.mean(avg_cost)
        batch_acc = fluid.layers.mean(batch_acc)
    else:
        # Train program
        predict = cnn_model(images)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        # Evaluator
        batch_acc = fluid.layers.accuracy(input=predict, label=label)

    # inference program
    inference_program = fluid.default_main_program().clone()

    # Optimization
    opt = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=args.batch_size * args.gpus)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=args.batch_size)
    return avg_cost, inference_program, opt, train_reader, test_reader, batch_acc
