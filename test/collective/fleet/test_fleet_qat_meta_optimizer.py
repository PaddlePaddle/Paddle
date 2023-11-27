# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import base, nn
from paddle.distributed import fleet

paddle.enable_static()
fleet.init(is_collective=True)


class SimpleNet(nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.linear3 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestFleetWithQAT(unittest.TestCase):
    def setUp(self):
        self.input_size = 4096
        self.output_size = 4096
        self.batch_size = 8

    def setup_strategy(self, strategy):
        strategy.qat = True

    def generate_program(self, strategy):
        train_prog, startup_prog = base.Program(), base.Program()
        with base.program_guard(train_prog, startup_prog):
            input_x = paddle.static.data(
                name='X',
                shape=[self.batch_size, self.input_size],
                dtype='float32',
            )
            input_y = paddle.static.data(
                name='Y',
                shape=[self.batch_size, self.output_size],
                dtype='float32',
            )
            model = SimpleNet(self.input_size, self.output_size)
            mse = paddle.nn.MSELoss()
            out = model(input_x)
            loss = mse(out, input_y)
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy
            )
            optimizer.minimize(loss)
        return train_prog, startup_prog, input_x, input_y, optimizer

    def execute_program(self, train_prog, startup_prog, input_x, input_y):
        place = (
            base.CUDAPlace(0)
            if paddle.base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        feeder = base.DataFeeder(feed_list=[input_x, input_y], place=place)
        exe.run(startup_prog)
        data = (
            np.random.randn(self.batch_size, self.input_size),
            np.random.randn(self.batch_size, self.output_size),
        )
        exe.run(train_prog, feed=feeder.feed([data]))

    def valid_program(self, train_prog, eval_prog):
        ops_type = [op.type for op in train_prog.block(0).ops]
        self.assertEqual(
            ops_type.count('matmul_v2'), 3
        )  # SimpleNet has 3 linear layers
        self.assertEqual(ops_type.count('quantize_linear'), 6)
        # There are three linear layers and each layer has this op in weight.
        self.assertEqual(
            ops_type.count('dequantize_linear'), 6
        )  # Dequantize Op will follow quantize op (fake quantize), so the number is same.

    def test_fleet_with_qat(self):
        dist_strategy = paddle.distributed.fleet.DistributedStrategy()
        self.setup_strategy(dist_strategy)
        (
            train_prog,
            startup_prog,
            input_x,
            input_y,
            optimizer,
        ) = self.generate_program(dist_strategy)
        place = (
            base.CUDAPlace(0)
            if paddle.base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        eval_prog = train_prog.clone(for_test=True)
        optimizer.qat_init(
            place, scope=paddle.static.global_scope(), test_program=eval_prog
        )
        self.execute_program(train_prog, startup_prog, input_x, input_y)
        self.valid_program(train_prog, eval_prog)


class TestFleetWithAMPQAT(TestFleetWithQAT):
    def setup_strategy(self, strategy):
        strategy.qat = True
        strategy.amp = True

    def valid_program(self, train_prog, eval_prog):
        ops_type = [op.type for op in train_prog.block(0).ops]
        self.assertEqual(
            ops_type.count('matmul_v2'), 3
        )  # SimpleNet has 3 linear layers
        self.assertEqual(ops_type.count('quantize_linear'), 6)
        # There are three linear layers and each layer has this op in weight.
        self.assertEqual(
            ops_type.count('dequantize_linear'), 6
        )  # Dequantize Op will follow quantize op (fake quantize), so the number is same.


if __name__ == "__main__":
    unittest.main()
