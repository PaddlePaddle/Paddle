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
import unittest

import paddle
from paddle.distributed import fleet

paddle.enable_static()


class TensorParallelNet(paddle.base.dygraph.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = paddle.nn.Embedding(hidden_size, hidden_size)
        self.col_linear = fleet.meta_parallel.ColumnParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            weight_attr=None,
            has_bias=True,
            gather_output=False,
            # name="test_column_linear",
        )
        self.row_linear = fleet.meta_parallel.RowParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            has_bias=True,
            input_is_parallel=True,
            # name="test_row_linear",
        )
        self.layer_norm = paddle.nn.LayerNorm(hidden_size)

    def forward(self, x):
        out = self.embedding(x)
        out = self.col_linear(out)
        out = self.row_linear(out)
        output = self.layer_norm(out)
        return output


def filter_fn(param, pos_emb=True, layer_norm=True, bias=True):
    """
    Layer filter function for tensor parallelism transformer.

    In tensor parallelism of transformer like model, there is 4 kind of param
    that are supposed to be the same in all tensor parallel peers:
        * position embedding
        * scale of layer norm
        * bias of layer norm
        * bias of row parallel linear

    set corresponding input args to select specific layers.
    NOTE  adopting the param name pattern for different transformer blocks.
    """
    p_name = param.name
    if pos_emb and p_name.startswith("embedding"):
        return True

    elif layer_norm and p_name.startswith("layer_norm"):
        return True

    elif bias and ".b_" in p_name and (param.is_distributed is False):
        return True

    else:
        return False


class TestFleetMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = (
            "127.0.0.1:36001,127.0.0.1:36002"
        )

    def test_tensor_parallel_extra_sync(self):
        from paddle.distributed import fleet

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.tensor_parallel = True
        strategy.tensor_parallel_configs = {"tensor_parallel_degree": 2}
        fleet.init(is_collective=True, strategy=strategy)

        main_program, startup_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, startup_program):
            hidden_size = 512
            input_x = paddle.static.data(
                name="x", shape=[-1, hidden_size], dtype='int64'
            )
            model_a = TensorParallelNet(hidden_size)
            y = model_a(input_x)
            loss = paddle.mean(y)

        optimizer = paddle.optimizer.Adam(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(loss)
        ref_ops = [
            'lookup_table_v2',
            'c_identity',
            'matmul_v2',
            'elementwise_add',
            'matmul_v2',
            'all_reduce',
            'elementwise_add',
            'layer_norm',
            'reduce_mean',
            'fill_constant',
            'reduce_mean_grad',
            'layer_norm_grad',
            'elementwise_add_grad',
            'c_identity',
            'matmul_v2_grad',
            'elementwise_add_grad',
            'matmul_v2_grad',
            'all_reduce',
            'lookup_table_v2_grad',
            'adam',
            'adam',
            'adam',
            'broadcast',
            'adam',
            'broadcast',
            'adam',
            'broadcast',
            'adam',
            'broadcast',
            'adam',
        ]
        paddle.distributed.fleet.utils.tensor_parallel_utils.add_extra_synchronization(
            main_program, params_filter_fn=filter_fn
        )
        ops = [op.type for op in main_program.global_block().ops]
        self.assertTrue(ops == ref_ops)


if __name__ == "__main__":
    unittest.main()
