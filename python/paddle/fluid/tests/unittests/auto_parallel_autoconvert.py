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
from paddle.distributed.auto_parallel.utils import save_distributed_checkpoint, load_distributed_checkpoint, load_checkpoint_into_program
from paddle.distributed.auto_parallel.utils import get_dist_attr, merge_and_slice_parameter, load_parameter_into_program
from paddle.distributed.auto_parallel.reshard import HAS_SENT, HAS_RECV, HAS_ALLGATHER

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
        arr0 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        arr1 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        weight_attr0 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr0))
        weight_attr1 = paddle.ParamAttr(initializer=NumpyArrayInitializer(arr1))
        bias_attr = None
        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
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
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    fleet.init(is_collective=True, strategy=dist_strategy)
    loss, train_program, startup_program = mlp_forward(train_program,
                                                       startup_program)
    optimizer = paddle.fluid.optimizer.SGDOptimizer(learning_rate=0.01)
    optimizer = fleet.distributed_optimizer(optimizer)
    _, _, dist_startup_prog, dist_main_prog = optimizer.minimize(
        loss, startup_program)

    return dist_main_prog, dist_startup_prog, loss


class TestMLPAutoConvert(unittest.TestCase):
    def setUp(self):
        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)

    def tearDown(self):
        os.remove("./model_state_rank{}.pdmodel".format(
            str(paddle.distributed.get_rank())))
        os.remove("./dist_attr_rank{}.pdattr".format(
            str(paddle.distributed.get_rank())))

    def test_mlp_mp2pp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1])

        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')

        dist_main_prog, dist_start_prog, loss = get_distributed_program()
        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)

        for step in range(20):
            if step == 10:
                save_distributed_checkpoint(
                    dist_main_prog, ".", dist_attr_path=".")

            res = exe.run(dist_main_prog,
                          feed={
                              "input": input[step * 4:(step + 1) * 4, :],
                              "label": label[step * 4:(step + 1) * 4, :]
                          },
                          fetch_list=[loss])
        last_res = res[0]

        _global_parallel_strategy = "pp"
        _global_process_mesh = auto.ProcessMesh([0, 1])
        global PP_MESH_0
        PP_MESH_0 = auto.ProcessMesh(mesh=[0])
        global PP_MESH_1
        PP_MESH_1 = auto.ProcessMesh(mesh=[1])

        dist_main_prog_load, dist_start_prog_load, loss_load = get_distributed_program(
        )
        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog_load)

        ckpt_path = [
            "./model_state_rank0.pdmodel", "./model_state_rank1.pdmodel"
        ]
        dist_attr_path = [
            "./dist_attr_rank0.pdattr", "./dist_attr_rank1.pdattr"
        ]
        load_checkpoint_into_program(ckpt_path, dist_attr_path,
                                     dist_main_prog_load)
        for step in range(10, 20):
            if paddle.distributed.get_rank() in [0]:
                res = exe.run(dist_main_prog_load,
                              feed={
                                  "input": input[step * 4:(step + 1) * 4, :],
                                  "label": label[step * 4:(step + 1) * 4, :]
                              })
            else:
                res = exe.run(dist_main_prog_load,
                              feed={
                                  "input": input[step * 4:(step + 1) * 4, :],
                                  "label": label[step * 4:(step + 1) * 4, :]
                              },
                              fetch_list=[loss_load])
        if paddle.distributed.get_rank() in [1]:
            self.assertEqual(last_res, res[0])


class TestMLPAutoConvert2(unittest.TestCase):
    def setUp(self):
        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)
        HAS_SENT.clear()
        HAS_RECV.clear()
        HAS_ALLGATHER.clear()

    def tearDown(self):
        os.remove("./model_state_rank{}.pdmodel".format(
            str(paddle.distributed.get_rank())))
        os.remove("./dist_attr_rank{}.pdattr".format(
            str(paddle.distributed.get_rank())))

    def test_mlp_pp2mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "pp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1])
        global PP_MESH_0
        PP_MESH_0 = auto.ProcessMesh(mesh=[0])
        global PP_MESH_1
        PP_MESH_1 = auto.ProcessMesh(mesh=[1])
        input = np.random.random(size=(80, 64)).astype('float32')
        label = np.random.random(size=(80, 1)).astype('float32')

        dist_main_prog, dist_start_prog, loss = get_distributed_program()
        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog)
        for step in range(20):
            if step == 10:
                add_info = {"batch": step, "batch_size": 4}
                save_distributed_checkpoint(dist_main_prog, ".", ".", add_info)

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

        _global_parallel_strategy = "mp"
        _global_process_mesh = auto.ProcessMesh([0, 1])

        dist_main_prog_load, dist_start_prog_load, loss_load = get_distributed_program(
        )
        place = paddle.set_device("gpu")
        exe = paddle.static.Executor(place)
        exe.run(dist_start_prog_load)
        ckpt_path = [
            "./model_state_rank0.pdmodel", "./model_state_rank1.pdmodel"
        ]
        dist_attr_path = [
            "./dist_attr_rank0.pdattr", "./dist_attr_rank1.pdattr"
        ]
        param_dict, pre_dist_attr, add_info = load_distributed_checkpoint(
            ckpt_path, dist_attr_path)
        batch = add_info["batch"]
        batch_size = add_info["batch_size"]
        start_index = batch * batch_size
        input = input[start_index:, :]
        label = label[start_index:, :]
        cur_dist_attr = get_dist_attr(dist_main_prog_load)
        sliced_param_dict = merge_and_slice_parameter(param_dict, pre_dist_attr,
                                                      cur_dist_attr)
        load_parameter_into_program(sliced_param_dict, dist_main_prog_load)
        for step in range(10):
            res = exe.run(dist_main_prog_load,
                          feed={
                              "input": input[step * 4:(step + 1) * 4, :],
                              "label": label[step * 4:(step + 1) * 4, :]
                          },
                          fetch_list=[loss_load])
        if paddle.distributed.get_rank() in [1]:
            self.assertEqual(last_res, res[0])


class TestMLPAutoConvertInvalid(unittest.TestCase):
    def setUp(self):
        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)

    def test_input_invalid(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh([0, 1])
        dist_main_prog, _, _ = get_distributed_program()
        with self.assertRaises(TypeError):
            save_distributed_checkpoint(
                dist_main_prog, [""], [""], addition_info=[0])
        with self.assertRaises(ValueError):
            save_distributed_checkpoint(
                dist_main_prog, [""], [""], addition_info={"step": 0})
        with self.assertRaises(ValueError):
            save_distributed_checkpoint(
                dist_main_prog, [""], [""], addition_info={"batch": 0.0})
        with self.assertRaises(ValueError):
            load_checkpoint_into_program(["./model_state_rank.pdmodel"],
                                         ["./dist_attr_rank.pdattr"],
                                         dist_main_prog)
        with self.assertRaises(ValueError):
            load_distributed_checkpoint(["./model_state_rank.pdmodel"],
                                        ["./dist_attr_rank.pdattr"])
        with self.assertRaises(TypeError):
            load_distributed_checkpoint({
                "0": "./model_state_rank.pdmodel"
            }, {"1": "./dist_attr_rank.pdattr"})


if __name__ == "__main__":
    unittest.main()
