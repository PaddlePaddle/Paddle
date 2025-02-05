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

import os
import random
import shutil
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn, static, utils
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.utils import (
    load_checkpoint_into_program,
    save_distributed_checkpoint,
)
from paddle.distributed.fleet import auto

paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None
PP_MESH_0 = None
PP_MESH_1 = None


class MLPLayer(nn.Layer):
    def __init__(
        self, hidden_size=64, intermediate_size=4 * 64, initializer_range=0.02
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        np.random.seed(2021)
        arr = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(arr)
        )
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        if _global_parallel_strategy == "pp":
            auto.shard_tensor(self.linear0.weight, PP_MESH_0, [None, None])
            auto.shard_tensor(self.linear1.weight, PP_MESH_1, [None, None])
        elif _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.linear0.weight, _global_process_mesh, [None, "x"]
            )
            auto.shard_tensor(
                self.linear1.weight, _global_process_mesh, ["x", None]
            )
        elif _global_parallel_strategy == "dp":
            auto.shard_tensor(
                self.linear0.weight, _global_process_mesh, [None, None]
            )
            auto.shard_tensor(
                self.linear1.weight, _global_process_mesh, [None, None]
            )

        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(
        train_program, start_program
    ), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 64
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32'
        )
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32'
        )

        if _global_parallel_strategy == "pp":
            auto.shard_tensor(input, PP_MESH_0, [None, None])
            auto.shard_tensor(label, PP_MESH_1, [None, None])
        elif _global_parallel_strategy == "dp":
            auto.shard_tensor(input, _global_process_mesh, ["x", None])
        elif _global_parallel_strategy == "mp":
            auto.shard_tensor(input, _global_process_mesh, [None, None])

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02,
        )

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_distributed_program():
    train_program = static.Program()
    startup_program = static.Program()

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    fleet.init(is_collective=True, strategy=dist_strategy)

    loss, train_program, startup_program = mlp_forward(
        train_program, startup_program
    )

    optimizer = paddle.optimizer.SGD(learning_rate=0.01)
    optimizer = fleet.distributed_optimizer(optimizer)
    _, _, dist_startup_prog, dist_main_prog = optimizer.minimize(
        loss, startup_program
    )

    return dist_main_prog, dist_startup_prog, loss


class TestMLPSaveLoad(unittest.TestCase):
    def setUp(self):
        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)

    def test_mlp_dp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1], dim_names=["x"])

        dist_main_prog, dist_start_prog, loss = get_distributed_program()
        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)

        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')
        for step in range(20):
            if step == 10:
                path = f"./output_dp{paddle.distributed.get_rank()}"
                os.makedirs(path, exist_ok=True)
                save_distributed_checkpoint(dist_main_prog, path, path)

            res = exe.run(
                dist_main_prog,
                feed={
                    "input": input[step * 4 : (step + 1) * 4, :],
                    "label": label[step * 4 : (step + 1) * 4, :],
                },
                fetch_list=[loss],
            )

        last_res = res[0]
        ckpt_path = [
            "./output_dp0/model_state_rank0.pdmodel",
            "./output_dp1/model_state_rank1.pdmodel",
        ]
        dist_attr_path = [
            "./output_dp0/dist_attr_rank0.pdattr",
            "./output_dp1/dist_attr_rank1.pdattr",
        ]
        load_checkpoint_into_program(ckpt_path, dist_attr_path, dist_main_prog)
        for step in range(10, 20):
            res = exe.run(
                dist_main_prog,
                feed={
                    "input": input[step * 4 : (step + 1) * 4, :],
                    "label": label[step * 4 : (step + 1) * 4, :],
                },
                fetch_list=[loss],
            )

        self.assertEqual(last_res, res[0])
        shutil.rmtree(f"./output_dp{paddle.distributed.get_rank()}")

    def test_mlp_mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1], dim_names=["x"])

        dist_main_prog, dist_start_prog, loss = get_distributed_program()

        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)

        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')
        for step in range(20):
            if step == 10:
                path = f"./output_mp{paddle.distributed.get_rank()}"
                os.makedirs(path, exist_ok=True)
                save_distributed_checkpoint(dist_main_prog, path, path)

            res = exe.run(
                dist_main_prog,
                feed={
                    "input": input[step * 4 : (step + 1) * 4, :],
                    "label": label[step * 4 : (step + 1) * 4, :],
                },
                fetch_list=[loss],
            )

        last_res = res[0]
        ckpt_path = [
            "./output_mp0/model_state_rank0.pdmodel",
            "./output_mp1/model_state_rank1.pdmodel",
        ]
        dist_attr_path = [
            "./output_mp0/dist_attr_rank0.pdattr",
            "./output_mp1/dist_attr_rank1.pdattr",
        ]
        load_checkpoint_into_program(ckpt_path, dist_attr_path, dist_main_prog)
        for step in range(10, 20):
            res = exe.run(
                dist_main_prog,
                feed={
                    "input": input[step * 4 : (step + 1) * 4, :],
                    "label": label[step * 4 : (step + 1) * 4, :],
                },
                fetch_list=[loss],
            )

        self.assertEqual(last_res, res[0])
        shutil.rmtree(f"./output_mp{paddle.distributed.get_rank()}")

    def test_mlp_pp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "pp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1], dim_names=["x"])
        global PP_MESH_0
        PP_MESH_0 = auto.ProcessMesh(mesh=[0], dim_names=["x"])
        global PP_MESH_1
        PP_MESH_1 = auto.ProcessMesh(mesh=[1], dim_names=["x"])

        dist_main_prog, dist_start_prog, loss = get_distributed_program()

        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)

        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')
        for step in range(20):
            if step == 10:
                path = f"./output_pp{paddle.distributed.get_rank()}"
                os.makedirs(path, exist_ok=True)
                save_distributed_checkpoint(dist_main_prog, path, path)

            if paddle.distributed.get_rank() in [0]:
                res = exe.run(
                    dist_main_prog,
                    feed={
                        "input": input[step * 4 : (step + 1) * 4, :],
                        "label": label[step * 4 : (step + 1) * 4, :],
                    },
                )
            else:
                res = exe.run(
                    dist_main_prog,
                    feed={
                        "input": input[step * 4 : (step + 1) * 4, :],
                        "label": label[step * 4 : (step + 1) * 4, :],
                    },
                    fetch_list=[loss],
                )

        if paddle.distributed.get_rank() in [1]:
            last_res = res[0]

        ckpt_path = [
            "./output_pp0/model_state_rank0.pdmodel",
            "./output_pp1/model_state_rank1.pdmodel",
        ]
        dist_attr_path = [
            "./output_pp0/dist_attr_rank0.pdattr",
            "./output_pp1/dist_attr_rank1.pdattr",
        ]
        load_checkpoint_into_program(ckpt_path, dist_attr_path, dist_main_prog)
        for step in range(10, 20):
            if paddle.distributed.get_rank() in [0]:
                res = exe.run(
                    dist_main_prog,
                    feed={
                        "input": input[step * 4 : (step + 1) * 4, :],
                        "label": label[step * 4 : (step + 1) * 4, :],
                    },
                )
            else:
                res = exe.run(
                    dist_main_prog,
                    feed={
                        "input": input[step * 4 : (step + 1) * 4, :],
                        "label": label[step * 4 : (step + 1) * 4, :],
                    },
                    fetch_list=[loss],
                )

        if paddle.distributed.get_rank() in [1]:
            self.assertEqual(last_res, res[0])
        shutil.rmtree(f"./output_pp{paddle.distributed.get_rank()}")


if __name__ == "__main__":
    unittest.main()
