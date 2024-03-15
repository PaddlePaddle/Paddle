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

import unittest

import numpy as np

import paddle
from paddle import _pir_ops, nn
from paddle.autograd.ir_backward import grad

paddle.enable_static()


class Net(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z1 = _pir_ops.add(x, y)
        z2 = _pir_ops.multiply(x, y)
        z3 = _pir_ops.subtract(z1, z2)
        z4 = _pir_ops.scale(z3, -1, 0, True)
        res = _pir_ops.divide(z3, z4)
        return res


class SimbolNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z1 = x + y
        z2 = x * y
        z3 = z1 - z2
        z4 = -z3
        res = z3 / z4
        return res


class CompareNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z1 = _pir_ops.less_equal(x, y)
        z2 = _pir_ops.greater_equal(x, y)
        z3 = _pir_ops.less_than(x, y)
        z4 = _pir_ops.greater_than(x, y)
        return z1, z2, z3, z4


class SimbolCompareNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z1 = x <= y
        z2 = x >= y
        z3 = x < y
        z4 = x > y
        return z1, z2, z3, z4


class TestValueSymbol(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 1024, 1024]
        self.shape_y = [2, 1024, 1024]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")

    def base_net(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            net = Net()
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            res = net(x, y)
            gradients = grad(res, (x, y))

            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[res, gradients[0], gradients[1]],
            )
            ops = [op.name() for op in main_program.global_block().ops]
        return outs, ops

    def symbol_net(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            net = SimbolNet()
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            res = net(x, y)
            gradients = grad(res, (x, y))

            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[res, gradients[0], gradients[1]],
            )
            ops = [op.name() for op in main_program.global_block().ops]
        return outs, ops

    def test_symbol_overload(self):
        res_ref, ops_ref = self.base_net()
        res, ops = self.symbol_net()
        for ref, actual in zip(res_ref, res):
            np.testing.assert_equal(ref, actual)
        self.assertEqual(ops_ref, ops)


class TestValueCompareSymbol(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 1024, 1024]
        self.shape_y = [2, 1024, 1024]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")

    def base_net(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            net = CompareNet()
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')

            res = net(x, y)

            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[res],
            )
            ops = [op.name() for op in main_program.global_block().ops]
        return outs, ops

    def symbol_net(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            net = SimbolCompareNet()
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')

            res = net(x, y)

            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[res],
            )
            ops = [op.name() for op in main_program.global_block().ops]
        return outs, ops

    def test_compare_symbol_overload(self):
        res_ref, ops_ref = self.base_net()
        res, ops = self.symbol_net()
        for ref, actual in zip(res_ref, res):
            np.testing.assert_equal(ref, actual)
        self.assertEqual(ops_ref, ops)


if __name__ == "__main__":
    unittest.main()
