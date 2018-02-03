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
import collections
import time

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.profiler as profiler
import pdb

SEED = 1
DTYPE = "float32"

# random seed must set before configuring the network.
# fluid.default_startup_program().random_seed = SEED


def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
    parser.add_argument(
        '--label_size', type=int, default=10, help='The label size.')
    parser.add_argument(
        '--batch_size', type=int, default=512, help='The minibatch size.')
    parser.add_argument(
        '--iterations', type=int, default=5, help='The number of minibatches.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    parser.add_argument(
        '--use_data_parallel',
        default=False,
        action='store_true',
        help='If set, use data parallel')
    args = parser.parse_args()
    return args


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def parallel_do_transpiler(init_program, program):
    nccl_communicator = init_program.block(0).create_var(name='nccl_com')
    init_program.block(0).append_op(
        type='ncclInit', outputs={'Communicator': nccl_communicator})
    nccl_communicator = program.block(0).create_var(name='nccl_com')

    def insert_allreduce_op(block, param_names):
        print(param_names)
        new_ops = collections.deque()
        for i in range(0, len(block.ops)):
            op = block.ops[i]
            new_ops.append(op)
            for o_param in op.output_names:
                for o_argu in op.output(o_param):
                    if o_argu in param_names:
                        block.append_op(
                            type='ncclAllReduce',
                            inputs={
                                'X': [block.var(o_argu)],
                                'Communicator': nccl_communicator
                            },
                            outputs={'Out': [block.create_var()]})
                        new_ops.append(block.ops[-1])
        block.ops = new_ops

    block = program.block(0)
    for op in block.ops:
        print(block.idx, op.type)
        if op.type == 'parallel_do_grad':
            param_names = set(op.output('parameters@GRAD'))
            sub_block = block.program.block(op.block_attr("sub_block"))
            insert_allreduce_op(sub_block, param_names)


def cnn_model(data, args):
    conv_pool_1 = fluid.layers.conv2d(
        input=data,
        stride=2,
        num_filters=20,
        filter_size=5,
        act=None,
        use_cudnn=True)
    # after_conv = conv_pool_1
    conv_pool_2 = fluid.layers.conv2d(
        input=conv_pool_1,
        stride=2,
        num_filters=50,
        filter_size=5,
        act=None,
        use_cudnn=True)
    conv_pool_3 = fluid.layers.conv2d(
        input=conv_pool_2,
        stride=2,
        num_filters=50,
        filter_size=5,
        act=None,
        use_cudnn=True)
    conv_pool_4 = fluid.layers.conv2d(
        input=conv_pool_3,
        stride=2,
        num_filters=50,
        filter_size=5,
        act=None,
        use_cudnn=True)
    conv_pool_5 = fluid.layers.conv2d(
        input=conv_pool_4,
        stride=2,
        num_filters=50,
        filter_size=5,
        act=None,
        use_cudnn=True)
    after_conv = conv_pool_5

    hidden = fluid.layers.fc(input=after_conv,
                             size=args.label_size,
                             act="softmax")
    predict = fluid.layers.fc(input=hidden, size=args.label_size, act="softmax")
    return predict


def run_benchmark(model, args):
    img_data = np.ones((args.batch_size, 1, 200, 200), dtype='float32')
    y_data = np.random.randint(
        low=0, high=10, size=(args.batch_size, 1), dtype='int64')
    images = fluid.layers.data(name='pixel', shape=[1, 200, 200], dtype=DTYPE)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    if args.use_data_parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            pd.read_input(images)
            predict = model(images, args)
            pd.write_output(predict)
        predict = pd()
    else:
        predict = model(images, args)

    # pdb.set_trace()
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    fluid.backward.append_backward(avg_cost)

    # # Optimization
    # opt = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
    # opt.minimize(avg_cost)

    parallel_do_transpiler(fluid.default_startup_program(),
                           fluid.default_main_program())
    print("---------")
    for block in fluid.default_main_program().blocks:
        for op in block.ops:
            print(block.idx, op.type)
    # print(fluid.default_main_program())
    time.sleep(2)

    # Initialize executor
    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # Parameter initialization
    exe.run(fluid.default_startup_program())

    for iter_id in range(0, args.iterations):
        start = time.time()

        outs = exe.run(fluid.default_main_program(),
                       feed={"pixel": img_data,
                             "label": y_data},
                       fetch_list=[avg_cost])
        loss = np.array(outs[0])

        end = time.time()
        print("iter=%d, error=%f, elapse=%f" % (iter_id, loss, (end - start)))
        time.sleep(1)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    run_benchmark(cnn_model, args)
