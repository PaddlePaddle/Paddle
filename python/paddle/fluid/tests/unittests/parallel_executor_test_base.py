# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import multiprocessing
import os
import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler
import time
import numpy as np
import math
import sys

__all__ = ['TestParallelExecutorBase']


class TestParallelExecutorBase(unittest.TestCase):
    def check_network_convergence(self,
                                  method,
                                  use_cuda=True,
                                  memory_opt=True,
                                  iter=50,
                                  batch_size=None,
                                  allow_op_delay=False,
                                  feed_dict=None,
                                  seed=None,
                                  use_parallel_executor=True,
                                  use_reduce=False,
                                  use_ir_memory_optimize=True,
                                  enable_inplace=True,
                                  fuse_elewise_add_act_ops=False,
                                  fuse_all_reduce_ops=False,
                                  fuse_relu_depthwise_conv=False,
                                  optimizer=fluid.optimizer.Adam,
                                  use_fast_executor=False,
                                  enable_sequential_execution=False):
        def run_executor(exe, binary, feed, fetch_list):
            res = exe.run(binary, feed=feed, fetch_list=fetch_list)
            return res

        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = 1  # Fix random seed
        main.random_seed = 1
        with fluid.program_guard(main, startup):
            if seed is not None:
                startup.random_seed = seed
                main.random_seed = seed

            loss = method(use_feed=feed_dict is not None)
            if optimizer:
                optimizer().minimize(loss)

            if memory_opt:
                fluid.memory_optimize(main)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.allow_op_delay = allow_op_delay
        if use_fast_executor:
            exec_strategy.use_experimental_executor = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce \
            if use_reduce else fluid.BuildStrategy.ReduceStrategy.AllReduce
        build_strategy.fuse_elewise_add_act_ops = fuse_elewise_add_act_ops
        build_strategy.fuse_relu_depthwise_conv = fuse_relu_depthwise_conv
        build_strategy.memory_optimize = False if memory_opt else use_ir_memory_optimize
        build_strategy.fuse_all_reduce_ops = fuse_all_reduce_ops
        # python memory optimization is conflict with inplace pass.
        # Use ir graph memory optimization after inplace pass is the correct way.
        build_strategy.enable_inplace = False if memory_opt else enable_inplace
        build_strategy.enable_sequential_execution = enable_sequential_execution

        if use_cuda and core.is_compiled_with_cuda():
            build_strategy.remove_unnecessary_lock = True
        if use_parallel_executor:
            binary = compiler.CompiledProgram(main).with_data_parallel(
                loss_name=loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
        else:
            binary = compiler.CompiledProgram(main)

        if batch_size is not None:
            batch_size *= fluid.core.get_cuda_device_count(
            ) if use_cuda else int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        begin = time.time()
        first_loss, = run_executor(
            exe=exe, binary=binary, feed=feed_dict, fetch_list=[loss.name])

        for i in range(iter):
            run_executor(exe=exe, binary=binary, feed=feed_dict, fetch_list=[])

        last_loss, = run_executor(
            exe=exe, binary=binary, feed=feed_dict, fetch_list=[loss.name])
        end = time.time()

        if batch_size is not None:
            print("%.4f Instance per second" % (
                (batch_size * iter + 2) / (end - begin)))

        avg_last_loss_val = np.array(last_loss).mean()
        avg_first_loss_val = np.array(first_loss).mean()
        if math.isnan(float(avg_last_loss_val)) or math.isnan(
                float(avg_first_loss_val)):
            sys.exit("got NaN loss, training failed.")

        print(first_loss, last_loss)
        # self.assertGreater(first_loss[0], last_loss[0])
        return first_loss, last_loss
