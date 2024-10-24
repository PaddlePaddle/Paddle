# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import paddle

sys.path.append("../legacy_test")

import paddle.nn.functional as F
from paddle import nn, static, utils
from paddle.base import ParamAttr
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
)
from paddle.distributed.auto_parallel.static.operators.common import (
    is_data_parallel_reduce_op,
    is_data_parallel_scale_op,
)
from paddle.distributed.auto_parallel.static.parallelizer_v2 import Parallelizer
from paddle.distributed.auto_parallel.static.planner_v2 import Planner
from paddle.distributed.auto_parallel.strategy import Strategy
from paddle.distributed.fleet import auto

paddle.enable_static()
BATCH_SIZE = 4
SEQ_LEN = 512
HIDDEN_SIZE = 1024
MESH_0 = auto.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"])


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        dropout_ratio=0.1,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = ParamAttr(
            initializer=paddle.nn.initializer.Normal(
                mean=0.0, std=initializer_range
            )
        )
        bias_attr = ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0)
        )

        self.norm = paddle.nn.LayerNorm(d_model, epsilon=1e-5)
        self.linear0 = paddle.nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = paddle.nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
        self.dropout = paddle.nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        return out


class HybridParallelNet(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
    ):
        super().__init__()
        self.mlp0 = MLPLayer(hidden_size, hidden_size * 4)
        self.mlp1 = MLPLayer(hidden_size, hidden_size * 4)
        self.mlp2 = MLPLayer(hidden_size, hidden_size * 4)

    def forward(self, input):
        # prune dp
        auto.shard_tensor(input, MESH_0, ["x", None, None])
        activation0 = self.mlp0(input)
        auto.shard_tensor(activation0, MESH_0, ["x", None, None])
        activation1 = F.gelu(activation0, approximate=True)

        # prune sp
        auto.shard_tensor(activation1, MESH_0, [None, "y", None])
        activation2 = self.mlp1(activation1)
        auto.shard_tensor(activation2, MESH_0, [None, "y", None])
        activation3 = F.gelu(activation2, approximate=True)

        # dp_sp
        auto.shard_tensor(activation3, MESH_0, ["x", "y", None])
        out = self.mlp2(activation3)

        return out


def get_hybrid_parallel_model(train_program, start_program):
    with static.program_guard(
        train_program, start_program
    ), utils.unique_name.guard():
        batch_size = BATCH_SIZE
        hidden_size = HIDDEN_SIZE
        sequence_len = SEQ_LEN

        input = static.data(
            name="input",
            shape=[batch_size, sequence_len, hidden_size],
            dtype='float32',
        )
        network = HybridParallelNet(hidden_size=HIDDEN_SIZE)

        predict = network(input)
        error_cost = paddle.sum(predict)

    return error_cost, train_program, start_program


def get_dist_prog(rank=2):
    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    loss, train_program, startup_program = get_hybrid_parallel_model(
        train_program, startup_program
    )
    opt = paddle.optimizer.AdamW(learning_rate=0.00001)
    strategy = Strategy()
    strategy.auto_mode = "semi"
    dist_context = DistributedContext(
        train_program, startup_program, opt, loss, strategy=strategy
    )
    planner = Planner("train", dist_context)
    planner.plan()

    parallelizer = Parallelizer(
        "train",
        planner.completer,
        dist_context,
    )
    parallelizer.parallel(rank=rank)

    return (
        dist_context.dist_main_programs[rank],
        dist_context.dist_startup_programs[rank],
    )


class TestGradSync(unittest.TestCase):
    def test_decoder_dp_sp(self):
        dist_main_prog, dist_startup_prog = get_dist_prog()

        ops = dist_main_prog.global_block().ops
        allreduce_count = 0
        scale_count = 0
        # Linear, Linear, LN
        dp_sync_indices = [
            0,
            2,
            4,
            6,
            8,
            9,
            18,
            19,
            20,
            21,
            22,
            23,
        ]  # check data parallel sync
        sp_sync_indices = [
            1,
            3,
            5,
            7,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
        ]  # check sp parallel sync
        dp_ring_id = None
        sp_ring_id = None
        dp_scale = 0.5
        sp_scale = 0.25

        for op in ops:
            if is_data_parallel_reduce_op(op):
                if allreduce_count in dp_sync_indices:
                    if dp_ring_id is None:
                        dp_ring_id = int(op.attr("ring_id"))
                    else:
                        assert dp_ring_id == int(
                            op.attr("ring_id")
                        ), "gradient synchronization of dp use different communication group [{}] and [{}]".format(
                            dp_ring_id, int(op.attr("ring_id"))
                        )
                elif allreduce_count in sp_sync_indices:
                    if sp_ring_id is None:
                        sp_ring_id = int(op.attr("ring_id"))
                    else:
                        assert sp_ring_id == int(
                            op.attr("ring_id")
                        ), "gradient synchronization of sp use different communication group [{}] and [{}]".format(
                            sp_ring_id, int(op.attr("ring_id"))
                        )
                else:
                    raise AssertionError(
                        f"encounter redundant gradient synchronization: [{op}]"
                    )
                allreduce_count += 1

            elif is_data_parallel_scale_op(op):
                if scale_count in dp_sync_indices:
                    assert dp_scale == float(
                        op.attr("scale")
                    ), "gradient synchronization of dp use different scale [{}] and [{}]".format(
                        dp_scale, int(op.attr("scale"))
                    )
                elif scale_count in sp_sync_indices:
                    assert sp_scale == float(
                        op.attr("scale")
                    ), "gradient synchronization of sp use different scale [{}] and [{}]".format(
                        sp_scale, int(op.attr("scale"))
                    )
                else:
                    raise AssertionError(
                        f"encounter redundant gradient synchronization: [{op}]"
                    )

                scale_count += 1

        assert scale_count == 24
        assert allreduce_count == 24


if __name__ == "__main__":
    unittest.main()
