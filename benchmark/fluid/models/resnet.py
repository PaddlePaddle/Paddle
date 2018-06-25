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

import functools
import numpy as np
import time
import os

import cProfile, pstats, StringIO

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
from recordio_converter import imagenet_train, imagenet_test


def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv1, act=act)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]  # if args.data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_out, stride):
    short = shortcut(input, ch_out, stride)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def bottleneck(input, ch_out, stride):
    short = shortcut(input, ch_out * 4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu')


def layer_warp(block_func, input, ch_out, count, stride):
    res_out = block_func(input, ch_out, stride)
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1)
    return res_out


def resnet_imagenet(input, class_dim, depth=50, data_format='NCHW'):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(input, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = fluid.layers.pool2d(
        input=conv1, pool_type='avg', pool_size=3, pool_stride=2)
    res1 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res2 = layer_warp(block_func, res1, 128, stages[1], 2)
    res3 = layer_warp(block_func, res2, 256, stages[2], 2)
    res4 = layer_warp(block_func, res3, 512, stages[3], 2)
    pool2 = fluid.layers.pool2d(
        input=res4,
        pool_size=7,
        pool_type='avg',
        pool_stride=1,
        global_pooling=True)
    out = fluid.layers.fc(input=pool2, size=class_dim, act='softmax')
    return out


def resnet_cifar10(input, class_dim, depth=32, data_format='NCHW'):
    assert (depth - 2) % 6 == 0

    n = (depth - 2) // 6

    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    out = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return out


def get_model(args):
    model = resnet_cifar10
    if args.data_set == "cifar10":
        class_dim = 10
        if args.data_format == 'NCHW':
            dshape = [3, 32, 32]
        else:
            dshape = [32, 32, 3]
        model = resnet_cifar10
        train_reader = paddle.dataset.cifar.train10()
        test_reader = paddle.dataset.cifar.test10()
    elif args.data_set == "flowers":
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
        model = resnet_imagenet
        train_reader = paddle.dataset.flowers.train()
        test_reader = paddle.dataset.flowers.test()
    elif args.data_set == "imagenet":
        class_dim = 1000
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
        model = resnet_imagenet
        if not args.data_path:
            raise Exception(
                "Must specify --data_path when training with imagenet")
        train_reader = imagenet_train(args.data_path)
        test_reader = imagenet_test(args.data_path)

    if args.use_reader_op:
        filelist = [
            os.path.join(args.data_path, f) for f in os.listdir(args.data_path)
        ]
        data_file = fluid.layers.open_files(
            filenames=filelist,
            shapes=[[-1] + dshape, (-1, 1)],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            thread_num=args.gpus,
            pass_num=args.pass_num)
        data_file = fluid.layers.double_buffer(
            fluid.layers.batch(
                data_file, batch_size=args.batch_size))
        input, label = fluid.layers.read_file(data_file)
    else:
        input = fluid.layers.data(name='data', shape=dshape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if args.device == 'CPU' and args.cpus > 1:
        places = fluid.layers.get_places(args.cpus)
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            predict = model(pd.read_input(input), class_dim)
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
        predict = model(input, class_dim)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        batch_acc = fluid.layers.accuracy(input=predict, label=label)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            target_vars=[batch_acc])

    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)

    batched_train_reader = paddle.batch(
        paddle.reader.shuffle(
            train_reader, buf_size=5120),
        batch_size=args.batch_size * args.gpus,
        drop_last=True)
    batched_test_reader = paddle.batch(
        train_reader, batch_size=args.batch_size, drop_last=True)

    return avg_cost, inference_program, optimizer, batched_train_reader,\
                   batched_test_reader, batch_acc
