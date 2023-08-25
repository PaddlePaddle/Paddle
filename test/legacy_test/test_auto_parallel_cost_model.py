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
from paddle.base import core
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.cost_model import estimate_cost
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
PP_MESH_0 = auto.ProcessMesh([[0, 1], [4, 5]], dim_names=["x", "y"])
PP_MESH_1 = auto.ProcessMesh([[2, 3], [6, 7]], dim_names=["x", "y"])
NUM_RANKS = 8
STAGE_0_CNT = 5
STAGE_1_CNT = 10
pp_cfg = [[0, 1, 4, 5], [2, 3, 6, 7]]

device = "gpu" if core.is_compiled_with_cuda() else "cpu"


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=256,
        intermediate_size=4 * 256,
        initializer_range=0.02,
        is_distributed=True,
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

        self.is_distributed = is_distributed

    def forward(self, input):
        if self.is_distributed:
            auto.shard_tensor(self.linear0.weight, PP_MESH_0, [None, None])
            auto.shard_tensor(self.linear1.weight, PP_MESH_1, ["y", None])

        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def get_single_node_data():
    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    loss, train_program, startup_program = mlp_forward(
        train_program, startup_program, is_distributed=False
    )

    cost_model = core.CostModel()
    cost_data = cost_model.profile_measure(
        train_program, startup_program, device, ["time"]
    )

    op_name2cost = [{}, {}]
    for idx, op in enumerate(train_program.blocks[0].ops):
        if idx <= STAGE_0_CNT:
            op_name2cost[0][op.type] = cost_data.get_op_time_ms(idx)
        elif idx <= STAGE_1_CNT:
            op_name2cost[1][op.type] = cost_data.get_op_time_ms(idx)
    return op_name2cost


def mlp_forward(train_program, start_program, is_distributed=True):
    with static.program_guard(
        train_program, start_program
    ), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 256
        sequence_len = 128
        if is_distributed:
            input = static.data(
                name="input", shape=[batch_size, hidden_size], dtype='float32'
            )
            label = static.data(
                name="label", shape=[batch_size, 1], dtype='float32'
            )
        else:
            input = paddle.ones(
                name="input", shape=[batch_size, hidden_size], dtype='float32'
            )
            label = paddle.ones(
                name="label", shape=[batch_size, 1], dtype='float32'
            )

        if is_distributed:
            auto.shard_tensor(input, PP_MESH_0, ["x", None])
            auto.shard_tensor(label, PP_MESH_1, ["x", None])

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02,
            is_distributed=is_distributed,
        )

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_dist_prog(train_program, startup_program, dist_context, rank_id):
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


def check_runtime_estimation(cost):
    return cost.runtime > 0


def check_memory_estimation(cost):
    for i in range(NUM_RANKS):
        if cost.static_mem[i] <= 0 or cost.peak_mem[i] <= 0:
            return False
        if cost.static_mem[i] > cost.peak_mem[i]:
            return False
    return True


def check_empty_program_runtime(cost):
    return cost.runtime == 0


def check_empty_program_memory(cost):
    for mem in cost.peak_mem:
        if mem > 1:
            return False
    for mem in cost.static_mem:
        if mem > 1:
            return False
    return True


class TestCostModel(unittest.TestCase):
    def test_empty_program_cost_model(self):
        empty_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        standalone_cost_data = [{}]
        empty_pp_cfg = None
        cluster = None
        cost = estimate_cost(
            [empty_program],
            cluster=cluster,
            pipeline_config=empty_pp_cfg,
            standalone_cost_data=standalone_cost_data,
            batch_size=1,
        )

        self.assertTrue(check_empty_program_runtime(cost))
        self.assertTrue(check_empty_program_memory(cost))

    def test_auto_parallel_cost_model(self):
        standalone_cost_data = get_single_node_data()
        dist_program = []
        for rank_id in range(NUM_RANKS):
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            dist_context = DistributedContext()
            (
                distributed_program,
                dist_startup_prog,
                dist_params_grads,
            ) = get_dist_prog(
                train_program, startup_program, dist_context, rank_id
            )
            resharder = Resharder(
                distributed_program,
                dist_startup_prog,
                rank_id,
                dist_context,
                dist_params_grads,
            )
            resharder.reshard()
            dist_program.append(distributed_program)
        cluster = None
        cost = estimate_cost(
            dist_program,
            cluster=cluster,
            pipeline_config=pp_cfg,
            standalone_cost_data=standalone_cost_data,
            batch_size=4,
        )
        self.assertTrue(check_runtime_estimation(cost))
        self.assertTrue(check_memory_estimation(cost))


if __name__ == "__main__":
    unittest.main()
