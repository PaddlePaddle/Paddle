# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import ast
import os
import random

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.distributed import fleet
from paddle.static import Executor, Program, program_guard


def get_program(args):
    main, startup = Program(), Program()
    main.random_seed = 10
    startup.random_seed = 10
    with base.unique_name.guard():
        with program_guard(main, startup):
            data = paddle.static.data(
                name='input',
                shape=args.dshape,
                dtype=args.dtype,
            )
            data.desc.set_need_check_feed(False)
            conv = paddle.static.nn.conv2d(
                input=data,
                num_filters=32,
                filter_size=1,
                param_attr=base.ParamAttr(name='conv2d_weight'),
                bias_attr=False,
                use_cudnn=args.use_cudnn,
            )
            bn = paddle.static.nn.batch_norm(
                conv,
                param_attr=base.ParamAttr(name='bn_scale'),
                bias_attr=base.ParamAttr(name='bn_bias'),
                moving_mean_name='bn_moving_mean',
                moving_variance_name='bn_moving_variance',
                data_layout=args.layout,
                is_test=args.only_forward,
            )
            if core.is_compiled_with_rocm():
                bn = paddle.cast(bn, 'float32')
            else:
                bn = paddle.cast(bn, 'float64')
            sigmoid = paddle.nn.functional.sigmoid(bn)
            out = paddle.sum(sigmoid)
            if not args.only_forward:
                sgd_opt = paddle.optimizer.SGD(learning_rate=0.0)
                opt = fleet.distributed_optimizer(sgd_opt)
                opt.minimize(out)
    return main, startup, [out, conv, bn]


def train(args):
    build_strategy = base.BuildStrategy()
    build_strategy.sync_batch_norm = True
    build_strategy.enable_inplace = False
    build_strategy.memory_optimize = False

    distributed_strategy = fleet.DistributedStrategy()
    distributed_strategy.build_strategy = build_strategy
    distributed_strategy.without_graph_optimization = True
    distributed_strategy.fuse_all_reduce_ops = True
    distributed_strategy.fuse_grad_size_in_num = 8

    fleet.init(is_collective=True, strategy=distributed_strategy)
    main, startup, outs = get_program(args)
    exe = Executor()
    exe.run(startup)

    for nm in args.fetch_list:
        fv = base.framework._get_var(str(nm), program=main)
        fv.persistable = True

    fetch_list = [v.name for v in outs] + args.fetch_list

    rank = paddle.distributed.get_rank()
    filepath = os.path.join(
        args.data_dir,
        'input_{}_{}_{}_{}.npy'.format(
            rank, args.only_forward, str(args.dtype), args.layout
        ),
    )
    data = np.load(filepath)

    comp_prog = base.compiler.CompiledProgram(
        main, build_strategy=build_strategy
    )
    sync_bn_fetches = exe.run(
        program=comp_prog, feed={'input': data}, fetch_list=fetch_list
    )

    for i in range(0, len(sync_bn_fetches)):
        file_path = os.path.join(
            args.data_dir,
            'output_{}_{}_{}_{}.npy'.format(
                rank, args.only_forward, str(args.dtype), i
            ),
        )
        np.save(file_path, sync_bn_fetches[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dshape', type=str, required=True)
    parser.add_argument('--dtype', type=str, required=True)
    parser.add_argument('--layout', type=str, required=True)
    parser.add_argument('--fetch_list', type=str, required=True)
    parser.add_argument('--use_cudnn', action='store_true')
    parser.add_argument('--only_forward', action='store_true')

    args = parser.parse_args()
    args.dshape = ast.literal_eval(args.dshape)
    args.fetch_list = ast.literal_eval(args.fetch_list)

    paddle.enable_static()

    paddle.seed(0)
    np.random.seed(0)
    random.seed(0)

    train(args)
