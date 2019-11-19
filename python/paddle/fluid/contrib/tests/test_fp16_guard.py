#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import contextlib
import math
import sys
import numpy
import unittest
import os
import numpy as np

import paddle.fluid.contrib.mixed_precision as mixed_precision
from paddle.fluid.transpiler.details import program_to_code


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def around_is_precision(train_program, startup_prog):
    classdim = 10
    data_shape = [3, 32, 32]
    with fluid.program_guard(train_program, startup_prog):
        with mixed_precision.precision_guard():
            images = fluid.layers.data(
                name='pixel', shape=data_shape, dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            net = resnet_cifar10(images, 32)

            logits = fluid.layers.fc(input=net, size=classdim, act="softmax")
            cost, predict = fluid.layers.softmax_with_cross_entropy(
                logits, label, return_softmax=True)
            avg_cost = fluid.layers.mean(cost)
            acc = fluid.layers.accuracy(input=predict, label=label)

            optimizer = fluid.optimizer.Lamb(learning_rate=0.001)
            mp_optimizer = fluid.contrib.mixed_precision.decorate(
                optimizer=optimizer,
                init_loss_scaling=8.0,
                use_dynamic_loss_scaling=True,
                decorate_type=mixed_precision.DecorateType.user_defined)

        return avg_cost, mp_optimizer


def around_is_half(train_program, startup_prog):
    classdim = 10
    data_shape = [3, 32, 32]

    with mixed_precision.half_precision_guard():
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        net = resnet_cifar10(images, 32)

        logits = fluid.layers.fc(input=net, size=classdim, act="softmax")
        cost, predict = fluid.layers.softmax_with_cross_entropy(
            logits, label, return_softmax=True)
        avg_cost = fluid.layers.mean(cost)

        optimizer = fluid.optimizer.Lamb(learning_rate=0.001)
        mp_optimizer = fluid.contrib.mixed_precision.decorate(
            optimizer=optimizer,
            init_loss_scaling=8.0,
            use_dynamic_loss_scaling=True,
            decorate_type=mixed_precision.DecorateType.use_defined)

    return avg_cost, mp_optimizer


def aroud_is_user_defined(train_program, startup_prog):
    classdim = 10
    data_shape = [3, 32, 32]

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    with mixed_precision.half_precision_guard():
        net = resnet_cifar10(images, 32)

    with mixed_precision.precision_guard():
        logits = fluid.layers.fc(input=net, size=classdim, act="softmax")
        cost, predict = fluid.layers.softmax_with_cross_entropy(
            logits, label, return_softmax=True)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=predict, label=label)

    optimizer = fluid.optimizer.Lamb(learning_rate=0.001)
    mp_optimizer = fluid.contrib.mixed_precision.decorate(
        optimizer=optimizer,
        init_loss_scaling=8.0,
        use_dynamic_loss_scaling=True,
        decorate_type=mixed_precision.DecorateType.user_defined)

    return avg_cost, mp_optimizer


class TestAmpGuard(unittest.TestCase):
    def test_decorate_half(self):
        train_program = fluid.Program()
        startup_prog = fluid.Program()
        train_program.random_seed = 123
        startup_prog.random_seed = 456

        with fluid.program_guard(train_program, startup_prog):
            avg_cost, optimizer = around_is_half(train_program, startup_prog)
            optimizer.minimize(avg_cost)
            loss_scaling = optimizer.get_loss_scaling()
            scaled_loss = optimizer.get_scaled_loss()
        print("test_decorate_half:")
        program_to_code(train_program)


if __name__ == '__main__':
    unittest.main()
