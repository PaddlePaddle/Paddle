#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.fluid as fluid

SEED = 1
DTYPE = "float32"

# random seed must set before configuring the network.
# fluid.default_startup_program().random_seed = SEED


def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
    parser.add_argument(
        '--label_size', type=int, default=10, help='The label size.')
    parser.add_argument(
        '--batch_size', type=int, default=10, help='The minibatch size.')
    parser.add_argument(
        '--iterations', type=int, default=5, help='The number of minibatches.')
    parser.add_argument(
        '--use_nccl',
        default=False,
        action='store_true',
        help='If set, use nccl')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def program_summary(program):
    print("--------------------")
    for block in program.blocks:
        for op in block.ops:
            outputs = [[x + ":"] + op.output(x) for x in op.output_names]
            inputs = [[x + ":"] + op.input(x) for x in op.input_names]
            print(block.idx, op.type, inputs, "|", outputs)


def vgg16_bn_drop(input):
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
    fc1 = fluid.layers.fc(input=drop, size=4096, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=4096, act=None)
    return fc2


def vgg16_bn(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=False,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    fc1 = fluid.layers.fc(input=conv5, size=4096, act=None)
    fc2 = fluid.layers.fc(input=fc1, size=4096, act=None)
    return fc2


def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False)
    # return conv1
    return fluid.layers.batch_norm(input=conv1, act=act)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
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


def run_benchmark(args):
    # Train program
    images = fluid.layers.fill_constant(
        shape=(args.batch_size, 3, 200, 200), dtype='float32', value=0.1)
    idx = fluid.layers.fill_constant(shape=[4, 1], dtype='int64', value=1)
    emb = fluid.layers.embedding(input=idx, size=[2, 3])
    avg_cost = fluid.layers.mean(emb)
    # predict = vgg16_bn_drop(images)
    # predict = resnet_imagenet(images, class_dim=1000)

    # avg_cost = fluid.layers.mean(x=predict)

    # fluid.layers.Print(idx, summarize=7)
    # fluid.layers.Print(emb, summarize=7)
    # fluid.layers.Print(avg_cost)

    # Optimization
    # Note the flag append_all_reduce=True
    opt = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
    opt.minimize(avg_cost)

    # program_summary(fluid.default_main_program())
    act_places = []
    for each in [
            fluid.CUDAPlace(i)
            for i in range(fluid.core.get_cuda_device_count())
    ]:
        p = fluid.core.Place()
        p.set_place(each)
        act_places.append(p)

    exe = fluid.core.ParallelExecutor(act_places,
                                      set([
                                          p.name
                                          for p in fluid.default_main_program()
                                          .global_block().iter_parameters()
                                      ]))

    # Parameter initialization
    exe.init(fluid.default_startup_program().desc, 0, True, True)
    for iter_id in range(0, args.iterations):
        start = time.time()
        exe.run(fluid.default_main_program().desc, 0, True, True)
        end = time.time()
        print("iter=%d, elapse=%f" % (iter_id, (end - start)))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    run_benchmark(args)
