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

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.process_group import new_process_group
from paddle.distributed.passes import PassManager, new_pass

paddle.enable_static()


class MPLinearNet(paddle.nn.Layer):
    def __init__(self, in_features, out_features, mp_degree=1):
        super().__init__()
        self.in_linear = paddle.nn.Linear(in_features, in_features)
        self.col_parallel_linear = fleet.meta_parallel.ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            weight_attr=None,
            has_bias=True,
            gather_output=False,
            name="test_column_linear",
        )
        self.row_parallel_linear = fleet.meta_parallel.RowParallelLinear(
            in_features=out_features,
            out_features=out_features,
            has_bias=True,
            input_is_parallel=True,
            name="test_row_linear",
        )
        self.out_linear = paddle.nn.Linear(out_features, out_features)

    def forward(self, x):
        x = self.in_linear(x)
        x = self.col_parallel_linear(x)
        x = self.row_parallel_linear(x)
        x = self.out_linear(x)
        return x


class TestDistTraning(unittest.TestCase):
    def setUp(self):
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"
        ] = "127.0.0.1:36001,127.0.0.1:36002"

        strategy = fleet.DistributedStrategy()
        strategy.tensor_parallel = True
        strategy.tensor_parallel_configs = {"tensor_parallel_degree": 2}

        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)
        new_process_group([0, 1])

    def get_value(self, use_pass=False):
        batch_size = 2
        in_features = 768
        out_features = 3072

        np.random.seed(1234)
        x_data = np.random.rand(batch_size, in_features, in_features).astype(
            'float32'
        )

        main_prog = paddle.static.Program()
        main_prog.random_seed = 1234
        startup_prog = paddle.static.Program()
        startup_prog.random_seed = 1234

        with paddle.static.program_guard(main_prog, startup_prog):
            data = paddle.static.data(
                name="x",
                shape=[batch_size, in_features, in_features],
                dtype='float32',
            )
            net = MPLinearNet(
                in_features=in_features,
                out_features=out_features,
            )
            out = net(data)
            loss = paddle.mean(out)
            sgd_optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.001)
            dist_opt = fleet.distributed_optimizer(sgd_optimizer, self.strategy)
            dist_opt.minimize(loss)

        if use_pass:
            pass_manager = PassManager([new_pass("fused_linear_with_mp_scale")])
            pass_manager.apply([main_prog], [startup_prog])
            ops = main_prog.global_block().ops
            assert 'scale' in [op.type for op in ops]
            assert 'fused_gemm_epilogue' in [op.type for op in ops]
            assert 'fused_gemm_epilogue_grad' in [op.type for op in ops]

        exe = paddle.static.Executor()
        exe.run(startup_prog)

        for i in range(2):
            ret_loss = exe.run(
                main_prog, feed={"x": x_data}, fetch_list=[loss.name]
            )

        return ret_loss

    def test_pass(self):
        ret_loss = self.get_value()
        ret_loss_fused = self.get_value(use_pass=True)
        # print("ret_loss: ", ret_loss)
        # print("ret_loss_fused: ", ret_loss_fused)
        assert np.allclose(ret_loss, ret_loss_fused, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
