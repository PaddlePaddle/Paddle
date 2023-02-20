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
import paddle.distributed.fleet as fleet

paddle.enable_static()

class TensorParallelNet(paddle.fluid.dygraph.Layer):
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
        out = self.col_linear(x)
        out = self.row_linear(out)
        output = self.layer_norm(out)
        return output



class TestFleetMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"
        ] = "127.0.0.1:36001,127.0.0.1:36002"

    def test_tensor_parallel_extra_sync(self):
        import paddle.distributed.fleet as fleet
        import paddle.distributed.fleet.base.role_maker as role_maker


        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.tensor_parallel = True
        strategy.tensor_parallel_configs = {"tensor_parallel_degree": 2}
        fleet.init(is_collective=True, strategy=strategy)

        main_program, startup_program = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            hidden_size = 512
            input_x = paddle.static.data(name="x", shape=[-1, hidden_size], dtype='float32')
            model_a = TensorParallelNet(hidden_size)
            y = model_a(input_x)
            loss = paddle.mean(y)


        optimizer = paddle.fluid.optimizer.Adam(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(loss)

        print(main_program)
        print(startup_program)

        # paddle.distributed.fleet.meta_optimizers.utils.tensor_parallel_utils.add_extra_synchronization(main_program)

if __name__ == "__main__":
    unittest.main()
