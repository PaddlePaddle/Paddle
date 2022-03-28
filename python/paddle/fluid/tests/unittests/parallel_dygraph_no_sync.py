# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import contextlib
import unittest
import numpy as np
import six
import pickle
import random

import paddle
import paddle.fluid as fluid
import paddle.distributed as dist
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid import core
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.framework import _test_eager_guard
from test_dist_base import print_to_err, print_to_out, runtime_main, TestParallelDyGraphRunnerBase

seed = 90
RUN_STEP = 20
batch_size = 4
batch_num = 1000


class SimpleNet(fluid.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net_a = Linear(input_dim=10, output_dim=20)
        self.net_b = Linear(input_dim=20, output_dim=5)
        self.net_c = Linear(input_dim=5, output_dim=10)

    def forward(self, x):
        x = self.net_a(x)
        x = self.net_b(x)
        x = self.net_c(x)
        return x


class TestNoSync(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SimpleNet()
        train_reader = paddle.batch(
            fake_sample_reader(), batch_size=batch_size, drop_last=True)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                         parameters=model.parameters())
        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        x_data = np.array([x for x in batch])
        x_data = x_data.reshape((-1, 10))
        x = paddle.to_tensor(x_data)
        out = model(x)
        loss = out.sum() / len(batch)
        return loss

    def run_trainer(self, args):
        if args.eager_mode:
            self.run_trainer_in_eager_mode(args)
        else:
            self.run_trainer_func(args)

    def run_trainer_with_spawn(self, args):
        if args.eager_mode:
            return self.run_trainer_with_spawn_in_eager_mode(args)
        else:
            return self.run_trainer_with_spawn_func(args)

    def run_trainer_func(self, args):
        if fluid.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(device_id)
        else:
            assert ("Only support CUDAPlace for now.")

        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            np.random.seed(seed)
            random.seed(seed)
            model, train_reader, opt = self.get_model()

            if args.update_method == "nccl2":
                dist.init_parallel_env()
                print_to_err(
                    type(self).__name__,
                    "begin to prepare context in dygraph with nccl2")
                model = paddle.DataParallel(
                    model, find_unused_parameters=args.find_unused_parameters)
            print_to_err(type(self).__name__, "model built in dygraph")
            return self.model_train(args, model, opt, train_reader)

    def run_trainer_in_eager_mode(self, args):
        if fluid.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(device_id)
        else:
            assert ("Only support CUDAPlace for now.")

        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            np.random.seed(seed)
            random.seed(seed)

            with _test_eager_guard():
                model, train_reader, opt = self.get_model()
                if args.update_method == "nccl2":
                    dist.init_parallel_env()
                    print_to_err(
                        type(self).__name__,
                        "begin to prepare context in dygraph with nccl2")

                    nranks = ParallelEnv().nranks
                    rank = ParallelEnv().local_rank
                    is_master = True if rank == 0 else False
                    store = paddle.fluid.core.TCPStore(
                        "127.0.0.1", args.dist_port, is_master, nranks)
                    group = core.ProcessGroupNCCL(store, rank, nranks)
                    model = paddle.DataParallel(
                        model,
                        process_group=group,
                        find_unused_parameters=args.find_unused_parameters)
                print_to_err(type(self).__name__, "model built in dygraph")
                return self.model_train(args, model, opt, train_reader)

    def model_train(self, args, model, opt, train_reader):
        out_losses = []
        for step_id, data in enumerate(train_reader()):
            data = self._get_data(data, args)
            if step_id == RUN_STEP:
                break
            if step_id % 3 != 0:
                if args.update_method == "nccl2":
                    with model.no_sync():
                        loss = self.run_one_loop(model, opt, data)
                        loss.backward()
                else:
                    loss = self.run_one_loop(model, opt, data)
                    loss.backward()
            else:
                loss = self.run_one_loop(model, opt, data)
                loss.backward()
                opt.minimize(loss)
                print_to_err(
                    type(self).__name__,
                    "loss at step %d: %f" % (step_id, loss.numpy()))
                out_losses.append(loss.numpy())
                model.clear_gradients()
        print_to_out(out_losses)
        return out_losses


def fake_sample_reader():
    def __reader__():
        for i in range(batch_num):
            x_data = np.random.random_sample((10, )).astype('float32')
            yield x_data

    return __reader__


if __name__ == "__main__":
    runtime_main(TestNoSync)
