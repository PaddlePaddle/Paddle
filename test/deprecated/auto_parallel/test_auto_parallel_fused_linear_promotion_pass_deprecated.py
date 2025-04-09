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

sys.path.append("../../legacy_test")

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
        enable_sp=False,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = ParamAttr(
            initializer=paddle.nn.initializer.Normal(
                mean=0.0, std=initializer_range
            )
        )
        self.enable_sp = enable_sp
        bias_attr = True

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
        if self.enable_sp:
            # sp region
            auto.shard_tensor(input, MESH_0, ["x", None, None])
            out = self.norm0(input)
            auto.shard_tensor(input, MESH_0, ["x", None, None])
            out = F.gelu(out, approximate=True)
        else:
            out = self.norm0(input)
            out = F.gelu(out, approximate=True)

        # tp region
        auto.shard_tensor(out, MESH_0, [None, None, None])
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        auto.shard_tensor(out, MESH_0, [None, None, None])

        if self.enable_sp:
            # sp region
            out = self.dropout(out)
            auto.shard_tensor(out, MESH_0, ["x", None, None])
            out = F.gelu(out, approximate=True)
            out = self.norm1(out)
        else:
            out = self.dropout(out)
            out = F.gelu(out, approximate=True)
            out = self.norm1(out)

        return out


class HybridParallelNet(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        enable_sp=False,
    ):
        super().__init__()
        self.mlp0 = MLPLayer(hidden_size, hidden_size * 4, enable_sp=enable_sp)
        self.mlp1 = MLPLayer(hidden_size, hidden_size * 4, enable_sp=enable_sp)

    def forward(self, input):
        out = self.mlp0(input)
        out = self.mlp1(out)

        return out


def get_hybrid_parallel_model(train_program, start_program, enable_sp=False):
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
        network = HybridParallelNet(
            hidden_size=HIDDEN_SIZE, enable_sp=enable_sp
        )

        predict = network(input)
        error_cost = paddle.sum(predict)

    return error_cost, train_program, start_program


def get_dist_prog(rank=0, enable_fused_linear_promotion=False, enable_sp=False):
    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    loss, train_program, startup_program = get_hybrid_parallel_model(
        train_program, startup_program, enable_sp=enable_sp
    )
    opt = paddle.optimizer.AdamW(learning_rate=0.00001)
    strategy = Strategy()
    strategy.auto_mode = "semi"
    strategy.fused_passes.enable = True
    strategy.sp_optimization.enable = enable_sp
    strategy.fused_linear_promotion.enable = enable_fused_linear_promotion
    strategy.fused_passes.fused_passes_list = ["fuse_gemm_epilogue"]
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


class TestFusedLinerPromotion(unittest.TestCase):
    def test_fused_linear_promotion_mp(self):
        dist_main_prog, _ = get_dist_prog(
            rank=0, enable_fused_linear_promotion=False, enable_sp=False
        )
        ops_without_promotion = dist_main_prog.global_block().ops
        origin_fused_gemm_epilogue_ops = [
            op
            for op in ops_without_promotion
            if op.type == "fused_gemm_epilogue"
        ]

        dist_main_prog_pro, _ = get_dist_prog(
            rank=0, enable_fused_linear_promotion=True, enable_sp=False
        )
        ops_with_promotion = dist_main_prog_pro.global_block().ops
        fused_gemm_epilogue_ops = [
            op for op in ops_with_promotion if op.type == "fused_gemm_epilogue"
        ]
        self.assertEqual(
            len(fused_gemm_epilogue_ops),
            len(origin_fused_gemm_epilogue_ops) + 2,
        )


if __name__ == "__main__":
    unittest.main()
