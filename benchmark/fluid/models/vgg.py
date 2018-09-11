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
"""VGG16 benchmark in Fluid"""
from __future__ import print_function

import sys
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import argparse
import functools
import os


def vgg16_bn_drop(input, is_train=True):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu', is_test=not is_train)
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


def get_model(args, is_train, main_prog, startup_prog):
    if args.data_set == "cifar10":
        classdim = 10
        if args.data_format == 'NCHW':
            data_shape = [3, 32, 32]
        else:
            data_shape = [32, 32, 3]
    else:
        classdim = 102
        if args.data_format == 'NCHW':
            data_shape = [3, 224, 224]
        else:
            data_shape = [224, 224, 3]
    filelist = [
        os.path.join(args.data_path, f) for f in os.listdir(args.data_path)
    ]
    with fluid.program_guard(main_prog, startup_prog):
        if args.use_reader_op:
            data_file_handle = fluid.layers.open_files(
                filenames=filelist,
                shapes=[[-1] + data_shape, (-1, 1)],
                lod_levels=[0, 0],
                dtypes=["float32", "int64"],
                thread_num=1,
                pass_num=1)
            data_file = fluid.layers.double_buffer(
                fluid.layers.batch(
                    data_file_handle, batch_size=args.batch_size))
        with fluid.unique_name.guard():
            if args.use_reader_op:
                images, label = fluid.layers.read_file(data_file)
            else:
                images = fluid.layers.data(
                    name='data', shape=data_shape, dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')
            # Train program
            net = vgg16_bn_drop(images, is_train=is_train)
            predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            # Evaluator
            batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
            batch_acc = fluid.layers.accuracy(
                input=predict, label=label, total=batch_size_tensor)
            # Optimization
            if is_train:
                optimizer = fluid.optimizer.Adam(
                    learning_rate=args.learning_rate)
                optimizer.minimize(avg_cost)

    # data reader
    if is_train:
        reader = paddle.dataset.cifar.train10() \
            if args.data_set == 'cifar10' else paddle.dataset.flowers.train()
    else:
        reader = paddle.dataset.cifar.test10() \
            if args.data_set == 'cifar10' else paddle.dataset.flowers.test()

    batched_reader = paddle.batch(
        paddle.reader.shuffle(
            reader, buf_size=5120),
        batch_size=args.batch_size * args.gpus)

    return avg_cost, optimizer, [batch_acc], batched_reader, data_file_handle
