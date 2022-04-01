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

import os
if os.getenv("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.dist_context import get_default_distributed_context
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.reshard import Resharder
from paddle.distributed.auto_parallel.process_group import new_process_group

paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=initializer_range))
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
        else:
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
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
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
        else:
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


def get_dist_prog_with_parallelizer(train_program, startup_program,
                                    dist_context):
    global _global_process_mesh

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.amp = False
    dist_strategy.pipeline = False
    dist_strategy.recompute = False

    # init parallel optimizer
    dist_strategy.semi_auto = True
    fleet.init(is_collective=True, strategy=dist_strategy)

    loss, train_program, startup_program = mlp_forward(train_program,
                                                       startup_program)

    optimizer = paddle.fluid.optimizer.AdamOptimizer(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None)
    optimizer = fleet.distributed_optimizer(optimizer)

    _, _, distributed_startup_program, distributed_main_program = optimizer.minimize(
        loss, startup_program)

    return distributed_main_program, distributed_startup_program


def check_send_recv_result(dist_main_prog, rank_id):
    send_result = False
    recv_result = False
    ops = dist_main_prog.global_block().ops
    if rank_id == 0:
        for idx, op in enumerate(ops):
            if op.type == "send_v2" and "gelu_0.tmp_0" in op.input_arg_names:
                send_result = True
            if op.type == "recv_v2" and "gelu_0.tmp_0@GRAD" in op.output_arg_names[
                    0]:
                recv_result = True
    else:
        for idx, op in enumerate(ops):
            if op.type == "send_v2" and "gelu_0.tmp_0@GRAD" in op.input_arg_names:
                send_result = True
            if op.type == "recv_v2" and "gelu_0.tmp_0" in op.output_arg_names[
                    0]:
                recv_result = True

    return send_result and recv_result


class TestMLPReshard(unittest.TestCase):
    def test_mlp_serial(self):
        global _global_parallel_strategy
        _global_parallel_strategy = None
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0])

        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = get_default_distributed_context()
        rank_id = 0
        dist_main_prog, dist_startup_prog = get_dist_prog_with_parallelizer(
            train_program, startup_program, dist_context)
        # send and recv should not exist in serial scene.
        self.assertFalse(check_send_recv_result(dist_main_prog, rank_id))


if __name__ == "__main__":
    unittest.main()
