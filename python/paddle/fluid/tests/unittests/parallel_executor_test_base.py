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

import multiprocessing
import os
import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler
import time
import numpy as np
import math
import sys
from feed_data_reader import FeedDataReader

__all__ = ['TestParallelExecutorBase']
DeviceType = core.DeviceType


class TestParallelExecutorBase(unittest.TestCase):

    @classmethod
    def check_network_convergence(cls,
                                  method,
                                  use_device=DeviceType.CUDA,
                                  iter=5,
                                  batch_size=None,
                                  feed_dict=None,
                                  feed_data_reader=None,
                                  get_data_from_feeder=None,
                                  use_parallel_executor=True,
                                  use_reduce=False,
                                  use_ir_memory_optimize=False,
                                  enable_inplace=True,
                                  fuse_elewise_add_act_ops=False,
                                  fuse_all_optimizer_ops=False,
                                  fuse_all_reduce_ops=False,
                                  fuse_relu_depthwise_conv=False,
                                  optimizer=fluid.optimizer.Adam,
                                  use_fast_executor=False,
                                  enable_sequential_execution=False):

        def run_executor(exe, binary, feed, fetch_list):
            if feed_data_reader is None:
                res = exe.run(binary, feed=feed, fetch_list=fetch_list)
            else:
                res = exe.run(binary,
                              feed=feed_data_reader.get_next(exe, binary),
                              fetch_list=fetch_list)
            return res

        if feed_data_reader is not None:
            assert isinstance(
                feed_data_reader, FeedDataReader
            ), "feed_data_reader must be type of FeedDataReader"

        paddle.seed(0)
        paddle.framework.random._manual_program_seed(0)
        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            feed_dict, loss = cls.build_model(feed_dict, get_data_from_feeder,
                                              main, method, optimizer)

        place = fluid.CUDAPlace(
            0) if use_device == DeviceType.CUDA else fluid.XPUPlace(
                0) if use_device == DeviceType.XPU else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)

        build_strategy, exec_strategy = cls.set_strategy(
            enable_inplace, enable_sequential_execution, fuse_all_optimizer_ops,
            fuse_all_reduce_ops, fuse_elewise_add_act_ops,
            fuse_relu_depthwise_conv, use_fast_executor, use_ir_memory_optimize,
            use_reduce, use_device)

        if use_parallel_executor:
            binary = compiler.CompiledProgram(main).with_data_parallel(
                loss_name=loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
        else:
            binary = main

        if batch_size is not None:
            batch_size *= fluid.core.get_cuda_device_count(
            ) if use_device == DeviceType.CUDA else fluid.core.get_xpu_device_count(
            ) if use_device == DeviceType.XPU else int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        area_below_loss = 0
        begin = time.time()
        first_loss, = run_executor(exe=exe,
                                   binary=binary,
                                   feed=feed_dict,
                                   fetch_list=[loss.name])
        area_below_loss += 0.5 * first_loss.mean()
        for _ in range(iter):
            mid_loss = run_executor(exe=exe,
                                    binary=binary,
                                    feed=feed_dict,
                                    fetch_list=[loss.name])
            area_below_loss += mid_loss[0].mean()
        last_loss, = run_executor(exe=exe,
                                  binary=binary,
                                  feed=feed_dict,
                                  fetch_list=[loss.name])
        area_below_loss += 0.5 * last_loss.mean()
        end = time.time()

        if batch_size is not None:
            print("%.4f Instance per second" % ((batch_size * iter + 2) /
                                                (end - begin)))

        avg_last_loss_val = np.array(last_loss).mean()
        avg_first_loss_val = np.array(first_loss).mean()
        if math.isnan(float(avg_last_loss_val)) or math.isnan(
                float(avg_first_loss_val)):
            sys.exit("got NaN loss, training failed.")

        print(first_loss, last_loss, area_below_loss)
        # self.assertGreater(first_loss[0], last_loss[0])
        return first_loss, last_loss, area_below_loss

    @classmethod
    def check_pass_conflict(cls,
                            method,
                            use_device=DeviceType.CUDA,
                            feed_dict=None,
                            get_data_from_feeder=None,
                            use_reduce=False,
                            use_ir_memory_optimize=True,
                            enable_inplace=True,
                            fuse_elewise_add_act_ops=False,
                            fuse_all_optimizer_ops=False,
                            fuse_all_reduce_ops=False,
                            fuse_relu_depthwise_conv=False,
                            optimizer=fluid.optimizer.Adam,
                            use_fast_executor=True,
                            enable_sequential_execution=False):

        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            feed_dict, loss = cls.build_model(feed_dict, get_data_from_feeder,
                                              main, method, optimizer)

        place = fluid.CUDAPlace(
            0) if use_device == DeviceType.CUDA else fluid.XPUPlace(
                0) if use_device == DeviceType.XPU else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)

        build_strategy, exec_strategy = cls.set_strategy(
            enable_inplace, enable_sequential_execution, fuse_all_optimizer_ops,
            fuse_all_reduce_ops, fuse_elewise_add_act_ops,
            fuse_relu_depthwise_conv, use_fast_executor, use_ir_memory_optimize,
            use_reduce, use_device)

        binary = compiler.CompiledProgram(main).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        exe.run(binary, feed=feed_dict, fetch_list=[loss.name])

    @classmethod
    def set_strategy(cls, enable_inplace, enable_sequential_execution,
                     fuse_all_optimizer_ops, fuse_all_reduce_ops,
                     fuse_elewise_add_act_ops, fuse_relu_depthwise_conv,
                     use_fast_executor, use_ir_memory_optimize, use_reduce,
                     use_device):
        exec_strategy = fluid.ExecutionStrategy()
        if use_fast_executor:
            exec_strategy.use_experimental_executor = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce \
            if use_reduce else fluid.BuildStrategy.ReduceStrategy.AllReduce
        build_strategy.fuse_elewise_add_act_ops = fuse_elewise_add_act_ops
        build_strategy.fuse_relu_depthwise_conv = fuse_relu_depthwise_conv
        build_strategy.fuse_all_optimizer_ops = fuse_all_optimizer_ops
        build_strategy.fuse_all_reduce_ops = fuse_all_reduce_ops
        build_strategy.memory_optimize = use_ir_memory_optimize
        build_strategy.enable_inplace = enable_inplace
        build_strategy.enable_sequential_execution = enable_sequential_execution

        if use_device == DeviceType.CUDA and core.is_compiled_with_cuda():
            build_strategy.remove_unnecessary_lock = True
        if use_device == DeviceType.XPU and core.is_compiled_with_xpu():
            build_strategy.fuse_elewise_add_act_ops = False
            build_strategy.fuse_relu_depthwise_conv = False
            build_strategy.fuse_all_optimizer_ops = False
            build_strategy.memory_optimize = False
            build_strategy.enable_inplace = False
            build_strategy.enable_sequential_execution = False

        return build_strategy, exec_strategy

    @classmethod
    def build_model(cls, feed_dict, get_data_from_feeder, main, method,
                    optimizer):
        loss = method(use_feed=feed_dict is not None)
        # NOTE(zjl): memory_optimize/inplace pass would not require
        # that loss.persistable = True.
        # We set loss.persistable = False here to verify our memory
        # optimization strategies intentionally.
        loss.persistable = False
        if optimizer:
            optimizer().minimize(loss)

        if get_data_from_feeder is not None:
            assert feed_dict is None
            feed_dict = get_data_from_feeder()
        return feed_dict, loss
