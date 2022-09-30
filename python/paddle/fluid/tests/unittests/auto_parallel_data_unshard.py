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

import unittest

import copy
import numpy as np
import random

import paddle
import paddle.nn as nn
import paddle.fluid.core as core
from paddle.distributed.fleet import auto
import paddle.nn.functional as F
from paddle.distributed import fleet

paddle.enable_static()
paddle.distributed.init_parallel_env()


class TestDataUnshard(unittest.TestCase):

    def test_dp2pp1mp1(self):

        def create_model(train_program, start_program):
            with paddle.static.program_guard(train_program, start_program):

                MESH_0 = auto.ProcessMesh([0, 1], dim_names=["x"])
                input = paddle.static.data(name='input', shape=[2, 8])
                label = paddle.static.data(name='label', shape=[2, 8])

                weight_attr = paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=0.02))
                linear0 = nn.Linear(8, 8, weight_attr)
                linear1 = nn.Linear(8, 8, weight_attr)

                auto.shard_tensor(input, MESH_0, ["x", None])
                auto.shard_tensor(label, MESH_0, ["x", None])
                auto.shard_tensor(linear0.weight, MESH_0, [None, None])
                auto.shard_tensor(linear1.weight, MESH_0, [None, None])

                linear0_out = linear0(input)
                gelu_out = F.gelu(linear0_out)
                linear1_out = linear1(gelu_out)
                error_cost = paddle.nn.functional.square_error_cost(
                    linear1_out, label)
                loss = paddle.mean(error_cost)
                return train_program, start_program, loss, input, label

        train_program = paddle.static.Program()
        start_program = paddle.static.Program()
        # serial program
        train_program, start_program, loss, input, label = create_model(
            train_program, start_program)

        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = paddle.fluid.optimizer.AdamOptimizer(learning_rate=0.00001,
                                                         beta1=0.9,
                                                         beta2=0.999,
                                                         epsilon=1e-08,
                                                         grad_clip=None)

        optimizer = fleet.distributed_optimizer(optimizer)
        _, _, distributed_startup_program, distributed_main_program = optimizer.minimize(
            loss, start_program)

        worker_index = paddle.distributed.get_rank()
        paddle.seed(worker_index + 2021)
        random.seed(worker_index + 2021)
        np.random.seed(worker_index + 2021)

        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(distributed_startup_program)

        input_data = np.array(range(2 * 8)).reshape([2, 8]).astype("float32")
        label_data = np.random.randint(0, 10, [2, 8]).astype("float32")

        fetchs = [loss.name, 'split@RESHARD.tmp_0'] if worker_index == 0 else [
            loss.name, 'split@RESHARD.tmp_1'
        ]
        loss_np, shard_data_np = exe.run(distributed_main_program,
                                         feed={
                                             "input": input_data,
                                             "label": label_data
                                         },
                                         fetch_list=fetchs)
        desired = input_data[worker_index].reshape(shard_data_np.shape)
        np.testing.assert_allclose(shard_data_np, desired)

    def dp1pp1mp2(self):

        def create_model(train_program, start_program):
            with paddle.static.program_guard(train_program, start_program):

                MESH_0 = auto.ProcessMesh([0, 1], dim_names=["x"])
                input = paddle.static.data(name='input', shape=[8, 8])
                label = paddle.static.data(name='label', shape=[8, 8])

                weight_attr = paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=0.02))
                linear0 = nn.Linear(8, 8, weight_attr)
                linear1 = nn.Linear(8, 8, weight_attr)

                auto.shard_tensor(input, MESH_0, [None, None])
                auto.shard_tensor(label, MESH_0, [None, None])
                auto.shard_tensor(linear0.weight, MESH_0, [None, "x"])
                auto.shard_tensor(linear1.weight, MESH_0, ["x", None])

                linear0_out = linear0(input)
                gelu_out = F.gelu(linear0_out)

                linear1_out = linear1(gelu_out)

                error_cost = paddle.nn.functional.square_error_cost(
                    linear1_out, label)
                loss = paddle.mean(error_cost)
                return train_program, start_program, loss, input, label

        train_program = paddle.static.Program()
        start_program = paddle.static.Program()
        # serial program
        train_program, start_program, loss, input, label = create_model(
            train_program, start_program)

        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        optimizer = paddle.fluid.optimizer.AdamOptimizer(learning_rate=0.00001,
                                                         beta1=0.9,
                                                         beta2=0.999,
                                                         epsilon=1e-08,
                                                         grad_clip=None)

        optimizer = fleet.distributed_optimizer(optimizer)
        _, _, distributed_startup_program, distributed_main_program = optimizer.minimize(
            loss, start_program)

        worker_index = paddle.distributed.get_rank()
        paddle.seed(worker_index + 2021)
        random.seed(worker_index + 2021)
        np.random.seed(worker_index + 2021)

        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(distributed_startup_program)

        input_data = np.array(range(8 * 8)).reshape([8, 8]).astype("float32")
        label_data = np.random.randint(0, 10, [8, 8]).astype("float32")

        fetchs = [loss.name, 'input']
        loss_np, shard_data_np = exe.run(distributed_main_program,
                                         feed={
                                             "input": input_data,
                                             "label": label_data
                                         },
                                         fetch_list=fetchs)

        desired = input_data.reshape(shard_data_np.shape)
        np.testing.assert_allclose(shard_data_np, desired)


if __name__ == "__main__":
    unittest.main()
