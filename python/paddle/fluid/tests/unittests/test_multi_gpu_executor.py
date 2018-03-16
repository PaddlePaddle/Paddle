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


def run_benchmark(args):
    # Train program
    images = fluid.layers.fill_constant(
        shape=(args.batch_size, 3, 200, 200), dtype='float32', value=0.01)
    predict = vgg16_bn_drop(images)
    label = fluid.layers.fill_constant(
        shape=(args.batch_size, 1), dtype='int64', value=0)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Optimization
    # Note the flag append_all_reduce=True
    opt = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
    opt.minimize(avg_cost)

    # program_summary(fluid.default_main_program())
    act_places = []
    for each in [fluid.CUDAPlace(0), fluid.CUDAPlace(1)]:
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
    exe.run(fluid.default_startup_program().desc, 0, True, True)

    for iter_id in range(0, args.iterations):
        start = time.time()
        exe.run(fluid.default_main_program().desc, 0, True, True)
        end = time.time()
        print("iter=%d, elapse=%f" % (iter_id, (end - start)))
        time.sleep(1)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    run_benchmark(args)
