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

import paddle
import paddle.nn.functional as F
from paddle import nn, static, utils
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.cost import calc_time_by_cost_model
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
)
from paddle.distributed.auto_parallel.static.parallelizer import (
    AutoParallelizer,
)
from paddle.distributed.auto_parallel.static.partitioner import Partitioner
from paddle.distributed.auto_parallel.static.reshard import Resharder
from paddle.distributed.fleet import auto

paddle.enable_static()
_global_parallel_strategy = "dp_mp_pp"
_global_process_mesh = auto.ProcessMesh(
    [[[0, 1], [4, 5]], [[2, 3], [6, 7]]], dim_names=["x", "y", "z"]
)
PP_MESH_0 = auto.ProcessMesh([[0, 1], [4, 5]], dim_names=["x", "y"])
PP_MESH_1 = auto.ProcessMesh([[2, 3], [6, 7]], dim_names=["x", "y"])


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
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
        auto.shard_tensor(self.linear0.weight, PP_MESH_0, [None, "y"])
        auto.shard_tensor(self.linear1.weight, PP_MESH_1, ["y", None])

        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        param = paddle.create_parameter([1024, 4096], paddle.float32)
        auto.shard_tensor(param, PP_MESH_1, [None, "y"])
        out = paddle.matmul(out, param)

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(
        train_program, start_program
    ), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32'
        )
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32'
        )

        auto.shard_tensor(input, PP_MESH_0, ["x", None])
        auto.shard_tensor(label, PP_MESH_1, ["x", None])

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02,
        )

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_dist_prog(train_program, startup_program, dist_context, rank_id):
    global _global_process_mesh
    dist_context.process_mesh = _global_process_mesh
    loss, train_program, startup_program = mlp_forward(
        train_program, startup_program
    )

    fleet._user_defined_strategy = fleet.DistributedStrategy()
    fleet.user_defined_optimizer = paddle.optimizer.Adam()
    parallelizer = AutoParallelizer(fleet)
    parallelizer._dist_context = dist_context

    # serial forward & backward completion
    completer = Completer(dist_context)
    complete_train_program = completer.complete_forward_annotation(
        train_program
    )
    dist_context.block_state.parse_forward_blocks(complete_train_program)
    params_grads = parallelizer._generate_backward(
        complete_train_program,
        startup_program,
        loss,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    )

    # logical partition
    partitioner = Partitioner(dist_context, rank_id)
    (
        auto_parallel_main_prog,
        auto_parallel_startup_prog,
        dist_params_grads,
    ) = partitioner.partition(
        complete_train_program, startup_program, params_grads
    )

    partitioned_optimize_ops = parallelizer._apply_optimize(
        auto_parallel_main_prog, auto_parallel_startup_prog, dist_params_grads
    )

    return (
        auto_parallel_main_prog,
        auto_parallel_startup_prog,
        dist_params_grads,
    )


class TestCostInterface(unittest.TestCase):
    def test_cost_interface(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 2
        dist_main_prog, dist_startup_prog, dist_params_grads = get_dist_prog(
            train_program, startup_program, dist_context, rank_id
        )

        resharder = Resharder(
            dist_main_prog,
            dist_startup_prog,
            rank_id,
            dist_context,
            dist_params_grads,
        )
        resharder.reshard()
        cluster = Cluster()
        cluster.gen_default_config_cluster(node_count=1, device_count=8)
        for op in dist_main_prog.global_block().ops:
            time = calc_time_by_cost_model(op, cluster)
            assert time > -1


if __name__ == "__main__":
    unittest.main()
