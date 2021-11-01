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

import unittest
import random
import numpy as np
import os
import shutil

import paddle
import paddle.nn as nn
import paddle.utils as utils
import paddle.static as static
import paddle.nn.functional as F
import paddle.distributed.auto_parallel as auto

from paddle.distributed import fleet
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.distributed.auto_parallel.utils import make_data_unshard
from paddle.distributed.auto_parallel.utils import save_distributed_checkpoint, load_distributed_checkpoint
from paddle.distributed.auto_parallel.reshard import reshard
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed.auto_parallel.process_group import get_all_process_groups

paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None
PP_MESH_0 = None
PP_MESH_1 = None


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=64,
                 intermediate_size=4 * 64,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        np.random.seed(2021)
        arr = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        weight_attr = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr))
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        if _global_parallel_strategy == "pp":
            auto.shard_tensor(
                self.linear0.weight,
                dist_attr={
                    "process_mesh": PP_MESH_0,
                    "dims_mapping": [-1, -1]
                })
            auto.shard_tensor(
                self.linear1.weight,
                dist_attr={
                    "process_mesh": PP_MESH_1,
                    "dims_mapping": [-1, -1]
                })
        elif _global_parallel_strategy == "mp":
            auto.shard_tensor(
                self.linear0.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, 0]
                })
            auto.shard_tensor(
                self.linear1.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [0, -1]
                })
        elif _global_parallel_strategy == "dp":
            auto.shard_tensor(
                self.linear0.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, -1]
                })
            auto.shard_tensor(
                self.linear1.weight,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, -1]
                })

        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(train_program,start_program), \
        utils.unique_name.guard():

        batch_size = 4
        hidden_size = 64
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32')
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32')

        if _global_parallel_strategy == "pp":
            auto.shard_tensor(
                input,
                dist_attr={
                    "process_mesh": PP_MESH_0,
                    "dims_mapping": [-1, -1]
                })
            auto.shard_tensor(
                label,
                dist_attr={
                    "process_mesh": PP_MESH_1,
                    "dims_mapping": [-1, -1]
                })
        elif _global_parallel_strategy == "dp":
            auto.shard_tensor(
                input,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [0, -1]
                })
        elif _global_parallel_strategy == "mp":
            auto.shard_tensor(
                input,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, -1]
                })

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02)

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_distributed_program():
    train_program = static.Program()
    startup_program = static.Program()
    dist_context = DistributedContext()
    rank_id = paddle.distributed.get_rank()

    loss, train_program, startup_program = mlp_forward(train_program,
                                                       startup_program)
    train_program = auto.complete_annotation(train_program, dist_context)

    optimizer = paddle.fluid.optimizer.SGDOptimizer(learning_rate=0.01)

    dist_strategy = fleet.DistributedStrategy()
    partitioner = Partitioner(dist_strategy, dist_context, rank_id)
    dist_main_program, dist_startup_program = partitioner.transpile_forward(
        train_program, startup_program)
    dist_params_grads = partitioner.apply_backward(
        loss, train_program, startup_program, dist_main_program,
        dist_startup_program)
    opt_ops = partitioner.apply_optimize(
        optimizer, dist_params_grads, dist_main_program, dist_startup_program)
    all_process_groups = get_all_process_groups()
    for process_group in all_process_groups:
        if rank_id not in process_group._ranks:
            continue
        process_group.instantiate()
    make_data_unshard(dist_main_program, dist_startup_program, dist_context)
    reshard(dist_main_program, dist_startup_program, rank_id, dist_context)
    return dist_main_program, dist_startup_program, loss


class TestMLPSaveLoad(unittest.TestCase):
    def setUp(self):
        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)

    def _rm_temp_dir(self, dirname):
        if paddle.distributed.get_rank() in [0]:
            shutil.rmtree(dirname)

    def test_mlp_dp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1])

        dist_main_prog, dist_start_prog, loss = get_distributed_program()
        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)

        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')
        for step in range(20):
            if step == 10:
                path = "./output"
                os.makedirs(path, exist_ok=True)
                save_distributed_checkpoint(dist_main_prog, path)

            res = exe.run(dist_main_prog,
                          feed={
                              "input": input[step * 4:(step + 1) * 4, :],
                              "label": label[step * 4:(step + 1) * 4, :]
                          },
                          fetch_list=[loss])

        last_res = res[0]
        ckpt_path = [
            "./output/model_state_rank0.pdmodel",
            "./output/model_state_rank1.pdmodel"
        ]
        load_distributed_checkpoint(ckpt_path, dist_main_prog)
        for step in range(10, 20):
            res = exe.run(dist_main_prog,
                          feed={
                              "input": input[step * 4:(step + 1) * 4, :],
                              "label": label[step * 4:(step + 1) * 4, :]
                          },
                          fetch_list=[loss])

        self.assertEqual(last_res, res[0])
        self._rm_temp_dir(path)

    def test_mlp_mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1])

        dist_main_prog, dist_start_prog, loss = get_distributed_program()

        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)

        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')
        for step in range(20):
            if step == 10:
                path = "./output"
                os.makedirs(path, exist_ok=True)
                save_distributed_checkpoint(dist_main_prog, path)

            res = exe.run(dist_main_prog,
                          feed={
                              "input": input[step * 4:(step + 1) * 4, :],
                              "label": label[step * 4:(step + 1) * 4, :]
                          },
                          fetch_list=[loss])

        last_res = res[0]
        ckpt_path = [
            "./output/model_state_rank0.pdmodel",
            "./output/model_state_rank1.pdmodel"
        ]
        load_distributed_checkpoint(ckpt_path, dist_main_prog)
        for step in range(10, 20):
            res = exe.run(dist_main_prog,
                          feed={
                              "input": input[step * 4:(step + 1) * 4, :],
                              "label": label[step * 4:(step + 1) * 4, :]
                          },
                          fetch_list=[loss])

        self.assertEqual(last_res, res[0])
        self._rm_temp_dir(path)

    def test_mlp_pp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "pp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1])
        global PP_MESH_0
        PP_MESH_0 = auto.ProcessMesh(mesh=[0])
        global PP_MESH_1
        PP_MESH_1 = auto.ProcessMesh(mesh=[1])

        dist_main_prog, dist_start_prog, loss = get_distributed_program()

        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)

        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')
        for step in range(20):
            if step == 10:
                path = "./output"
                os.makedirs(path, exist_ok=True)
                save_distributed_checkpoint(dist_main_prog, path)

            if paddle.distributed.get_rank() in [0]:
                res = exe.run(dist_main_prog,
                              feed={
                                  "input": input[step * 4:(step + 1) * 4, :],
                                  "label": label[step * 4:(step + 1) * 4, :]
                              })
            else:
                res = exe.run(dist_main_prog,
                              feed={
                                  "input": input[step * 4:(step + 1) * 4, :],
                                  "label": label[step * 4:(step + 1) * 4, :]
                              },
                              fetch_list=[loss])

        if paddle.distributed.get_rank() in [1]:
            last_res = res[0]

        ckpt_path = [
            "./output/model_state_rank0.pdmodel",
            "./output/model_state_rank1.pdmodel"
        ]
        load_distributed_checkpoint(ckpt_path, dist_main_prog)
        for step in range(10, 20):
            if paddle.distributed.get_rank() in [0]:
                res = exe.run(dist_main_prog,
                              feed={
                                  "input": input[step * 4:(step + 1) * 4, :],
                                  "label": label[step * 4:(step + 1) * 4, :]
                              })
            else:
                res = exe.run(dist_main_prog,
                              feed={
                                  "input": input[step * 4:(step + 1) * 4, :],
                                  "label": label[step * 4:(step + 1) * 4, :]
                              },
                              fetch_list=[loss])

        if paddle.distributed.get_rank() in [1]:
            self.assertEqual(last_res, res[0])
        self._rm_temp_dir(path)


if __name__ == "__main__":
    unittest.main()
