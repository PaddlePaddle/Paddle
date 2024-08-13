#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from numpy import linalg as LA
from op_test import OpTest

import paddle
import paddle.distributed as dist
from paddle import _C_ops, _legacy_C_ops


def test_squared_l2_norm(x):
    return _C_ops.squared_l2_norm(x)


class TestSquaredL2NormF16Op(unittest.TestCase):
    def init_test_case(self):
        X = np.random.uniform(-0.1, 0.1, (8, 5, 10)).astype('float32')
        return X

    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np)

        x.stop_gradient = False
        y = test_squared_l2_norm(x)
        x_g = paddle.grad(y, [x])

        paddle.enable_static()
        return y, x_g

    def test_main(self):
        x_np = self.init_test_case()
        y_np_1, x_g_np_1 = self.check_main(x_np, 'float32')
        y_np_2, x_g_np_2 = self.check_main(x_np, 'float16')

        def assert_equal(x, y):
            np.testing.assert_allclose(x, y, rtol=1e-05, atol=0.0)

        assert_equal(y_np_1, y_np_2)
        assert_equal(x_g_np_1, x_g_np_2)


class TestSquaredL2NormF16Op1(TestSquaredL2NormF16Op):
    def init_test_case(self):
        X = np.random.uniform(-2.0, 2.0, (30, 10)).astype('float32')
        return X


class TestSquaredL2NormF16Op2(TestSquaredL2NormF16Op):
    def init_test_case(self):
        X = np.random.uniform(-5.0, 5.0, (20, 10, 20)).astype('float32')
        return X


class TestL2LossOp(OpTest):
    """Test squared_l2_norm"""

    def config(self):
        self.x_shape = (13, 19)
        self.check_auto_parallel = False

    def setUp(self):
        self.config()
        self.python_api = test_squared_l2_norm
        self.public_python_api = test_squared_l2_norm
        self.op_type = "squared_l2_norm"
        self.prim_op_type = "comp"
        self.max_relative_error = 0.05

        X = np.random.uniform(-1, 1, self.x_shape).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.array([np.square(LA.norm(X))])}

    def test_check_output(self):
        self.check_output(check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=self.max_relative_error,
            check_auto_parallel=self.check_auto_parallel,
        )


class TestSquaredL2NormAutoParallel_1(TestL2LossOp):
    def config(self):
        self.x_shape = (14, 18)
        self.check_auto_parallel = True
        self.placements = {
            'X': [dist.Replicate()],
        }


class TestSquaredL2NormAutoParallel_2(TestL2LossOp):
    def config(self):
        self.x_shape = (14, 18)
        self.check_auto_parallel = True
        self.placements = {
            'X': [dist.Shard(0)],
        }


class TestSquaredL2NormAutoParallel_3(TestL2LossOp):
    def config(self):
        self.x_shape = (14, 18)
        self.check_auto_parallel = True
        self.placements = {
            'X': [dist.Shard(1)],
        }


class TestL2LossDeterministic(unittest.TestCase):
    def check_place(self, place):
        with paddle.base.dygraph.guard(place):
            x_np = np.random.rand(5, 11, 13).astype('float32')
            x = paddle.to_tensor(x_np)
            y1 = _legacy_C_ops.squared_l2_norm(x)
            y2 = _legacy_C_ops.squared_l2_norm(x)
            np.testing.assert_array_equal(y1.numpy(), y2.numpy())

    def test_main(self):
        self.check_place(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.check_place(paddle.CUDAPlace(0))


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
