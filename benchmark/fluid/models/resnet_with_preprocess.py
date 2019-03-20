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
# from recordio_converter import imagenet_train, imagenet_test
from imagenet_reader import train_raw, val


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  is_train=True):
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv1, act=act, is_test=not is_train)


def shortcut(input, ch_out, stride, is_train=True):
    ch_in = input.shape[1]  # if args.data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out:
        return conv_bn_layer(
            input, ch_out, 1, stride, 0, None, is_train=is_train)
    else:
        return input


def basicblock(input, ch_out, stride, is_train=True):
    short = shortcut(input, ch_out, stride, is_train=is_train)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1, is_train=is_train)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None, is_train=is_train)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def bottleneck(input, ch_out, stride, is_train=True):
    short = shortcut(input, ch_out * 4, stride, is_train=is_train)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0, is_train=is_train)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, is_train=is_train)
    conv3 = conv_bn_layer(
        conv2, ch_out * 4, 1, 1, 0, act=None, is_train=is_train)
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu')


def layer_warp(block_func, input, ch_out, count, stride):
    res_out = block_func(input, ch_out, stride)
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1)
    return res_out


def resnet_imagenet(input,
                    class_dim,
                    depth=50,
                    data_format='NCHW',
                    is_train=True):

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


def _model_reader_dshape_classdim(args, is_train):
    model = resnet_cifar10
    reader = None
    if args.data_set == "cifar10":
        class_dim = 10
        if args.data_format == 'NCHW':
            dshape = [3, 32, 32]
        else:
            dshape = [32, 32, 3]
        model = resnet_cifar10
        if is_train:
            reader = paddle.dataset.cifar.train10()
        else:
            reader = paddle.dataset.cifar.test10()
    elif args.data_set == "flowers":
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
        model = resnet_imagenet
        if is_train:
            reader = paddle.dataset.flowers.train()
        else:
            reader = paddle.dataset.flowers.test()
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
        if not args.use_reader_op:
            if is_train:
                reader = train_raw()
            else:
                reader = val()
        else:
            if is_train:
                reader = train_raw()
            else:
                reader = val(xmap=False)
    return model, reader, dshape, class_dim


def get_model(args, is_train, main_prog, startup_prog):
    model, reader, dshape, class_dim = _model_reader_dshape_classdim(args,
                                                                     is_train)

    pyreader = None
    trainer_count = int(os.getenv("PADDLE_TRAINERS"))
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            if args.use_reader_op:
                pyreader = fluid.layers.py_reader(
                    capacity=args.batch_size * args.gpus,
                    shapes=([-1] + dshape, (-1, 1)),
                    dtypes=('uint8', 'int64'),
                    name="train_reader" if is_train else "test_reader",
                    use_double_buffer=True)
                input, label = fluid.layers.read_file(pyreader)
            else:
                input = fluid.layers.data(
                    name='data', shape=dshape, dtype='uint8')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')

            # add imagenet preprocessors
            random_crop = fluid.layers.random_crop(input, dshape)
            casted = fluid.layers.cast(random_crop, 'float32')
            # input is HWC
            trans = fluid.layers.transpose(casted, [0, 3, 1, 2]) / 255.0
            img_mean = fluid.layers.tensor.assign(
                np.array([0.485, 0.456, 0.406]).astype('float32').reshape((3, 1,
                                                                           1)))
            img_std = fluid.layers.tensor.assign(
                np.array([0.229, 0.224, 0.225]).astype('float32').reshape((3, 1,
                                                                           1)))
            h1 = fluid.layers.elementwise_sub(trans, img_mean, axis=1)
            h2 = fluid.layers.elementwise_div(h1, img_std, axis=1)

            # pre_out = (trans - img_mean) / img_std

            predict = model(h2, class_dim, is_train=is_train)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=predict, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=predict, label=label, k=5)

            # configure optimize
            optimizer = None
            if is_train:
                total_images = 1281167 / trainer_count

                step = int(total_images / args.batch_size + 1)
                epochs = [30, 60, 80, 90]
                bd = [step * e for e in epochs]
                base_lr = args.learning_rate
                lr = []
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=base_lr,
                    #learning_rate=fluid.layers.piecewise_decay(
                    #    boundaries=bd, values=lr),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
                optimizer.minimize(avg_cost)

                if args.memory_optimize:
                    fluid.memory_optimize(main_prog)

    # config readers
    if not args.use_reader_op:
        batched_reader = paddle.batch(
            reader if args.no_random else paddle.reader.shuffle(
                reader, buf_size=5120),
            batch_size=args.batch_size * args.gpus,
            drop_last=True)
    else:
        batched_reader = None
        pyreader.decorate_paddle_reader(
            paddle.batch(
                # reader if args.no_random else paddle.reader.shuffle(
                #     reader, buf_size=5120),
                reader,
                batch_size=args.batch_size))

    return avg_cost, optimizer, [batch_acc1,
                                 batch_acc5], batched_reader, pyreader
