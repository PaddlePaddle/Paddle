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
from paddle.distributed.auto_parallel.static.parallelizer_v2 import Parallelizer
from paddle.distributed.auto_parallel.static.planner_v2 import Planner
from paddle.distributed.auto_parallel.strategy import Strategy
from paddle.distributed.fleet import auto

paddle.enable_static()
BATCH_SIZE = 4
SEQ_LEN = 512
HIDDEN_SIZE = 1024
MESH_0 = auto.ProcessMesh([0, 1, 2, 3], dim_names=["x"])


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
        bias_attr = False

        self.norm0 = paddle.nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm0.bias.stop_gradient = True
        self.norm1 = paddle.nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm1.bias.stop_gradient = True
        self.linear0 = paddle.nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        auto.shard_tensor(self.linear0.weight, MESH_0, [None, "x"])
        self.linear1 = paddle.nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
        auto.shard_tensor(self.linear1.weight, MESH_0, ["x", None])
        self.dropout = paddle.nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        # sp region
        auto.shard_tensor(input, MESH_0, ["x", None, None])
        out = self.norm0(input)
        auto.shard_tensor(input, MESH_0, ["x", None, None])
        out = F.gelu(out, approximate=True)

        # tp region
        auto.shard_tensor(out, MESH_0, [None, None, None])
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        auto.shard_tensor(out, MESH_0, [None, None, None])

        # sp region
        out = self.dropout(out)
        auto.shard_tensor(out, MESH_0, ["x", None, None])
        out = F.gelu(out, approximate=True)
        out = self.norm1(out)

        return out


class HybridParallelNet(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
    ):
        super().__init__()
        self.mlp0 = MLPLayer(hidden_size, hidden_size * 4)
        self.mlp1 = MLPLayer(hidden_size, hidden_size * 4)

    def forward(self, input):
        out = self.mlp0(input)
        out = self.mlp1(out)

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
    strategy.sp_optimization.enable = True
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

        with open("test_static_sequence_parallel.txt", "w+") as f:
            f.write(str(dist_main_prog))

        ops = dist_main_prog.global_block().ops
        sp_ring_id = None
        allgather_count = 0
        reducescatter_count = 0
        allreduce_count = 0

        for op in ops:
            # check sequence parallel allgather
            if op.type == "all_gather":
                assert (
                    int(op.attr("nranks")) == 4
                ), "sequence parallel allgather error with nranks [{}]".format(
                    op.attr("nranks")
                )
                if sp_ring_id is None:
                    sp_ring_id = int(op.attr("ring_id"))
                else:
                    assert sp_ring_id == int(
                        op.attr("ring_id")
                    ), "sequence parallel allgather error with ring_id [{}]".format(
                        op.attr("ring_id")
                    )
                allgather_count += 1

            # check sequence parallel reducescatter
            elif op.type == "reduce_scatter":
                assert (
                    int(op.attr("nranks")) == 4
                ), "sequence parallel reducescatter error with nranks [{}]".format(
                    op.attr("nranks")
                )
                assert sp_ring_id == int(
                    op.attr("ring_id")
                ), "sequence parallel reducescatter error with ring_id [{}]".format(
                    op.attr("ring_id")
                )
                reducescatter_count += 1

            # check sequence parallel grad sync
            elif op.type == "c_allreduce_sum":
                assert (
                    "layer_norm" in op.output_arg_names[0]
                ), f"sequence parallel reducescatter error grad sync var [{op.output_arg_names[0]}]"
                assert sp_ring_id == int(
                    op.attr("ring_id")
                ), "sequence parallel reducescatter error with ring_id [{}]".format(
                    op.attr("ring_id")
                )
                allreduce_count += 1

        assert (
            allgather_count == 4
        ), f"sequence parallel should have 4 allgather, but got [{allgather_count}]"
        assert (
            reducescatter_count == 4
        ), f"sequence parallel should have 4 allgather, but got [{reducescatter_count}]"
        assert (
            allreduce_count == 4
        ), f"sequence parallel should have 4 allgather, but got [{allreduce_count}]"


if __name__ == "__main__":
    unittest.main()
