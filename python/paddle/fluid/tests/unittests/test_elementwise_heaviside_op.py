#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((13, 17)).astype("float64")
        y = np.random.random((13, 17)).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))

class TestElementwiseHeavisideOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((100, 5)).astype(np.float64)
        y = np.random.random((100, 1)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseHeavisideOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_heaviside"
        x = np.random.random((5, 100)).astype(np.float64)
        y = np.random.random((1, 100)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out':np.heaviside(self.inputs['X'], self.inputs['Y'])}


class TestHeavisideAPI_float64(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.random((13, 17)).astype("float64")
        self.y_np = np.random.random((13, 17)).astype("float64")
        self.out_np = np.heaviside(self.x_np, self.y_np)
        self.dtype = "float64"

    def test_static(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            x = paddle.fluid.data(
                name=f"x_{self.dtype}", shape=[13, 17], dtype=self.dtype)
            y = paddle.fluid.data(
                name=f"y_{self.dtype}", shape=[13, 17], dtype=self.dtype)
            out = paddle.heaviside(x, y)

            exe = paddle.static.Executor(place=place)
            res = exe.run(fluid.default_main_program(),
                            feed={f"x_{self.dtype}": self.x_np, f"y_{self.dtype}": self.y_np},
                            fetch_list=out,
                            use_prune=True)

            self.assertTrue(np.allclose(res, self.out_np))

    def test_dygraph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
            result = paddle.heaviside(paddle.to_tensor(self.x_np), paddle.to_tensor(self.y_np))

            self.assertTrue(np.allclose(result.numpy(), self.out_np))


class TestHeavisideAPI_float32(TestHeavisideAPI_float64):
    def setUp(self):
        self.x_np = np.random.random((13, 17)).astype("float32")
        self.y_np = np.random.random((13, 17)).astype("float32")
        self.out_np = np.heaviside(self.x_np, self.y_np)
        self.dtype = "float32"


class TestHeavisideAPI_int64(TestHeavisideAPI_float64):
    def setUp(self):
        self.x_np = np.random.random((13, 17)).astype("int64")
        self.y_np = np.random.random((13, 17)).astype("int64")
        self.out_np = np.heaviside(self.x_np, self.y_np)
        self.dtype = "int64"


class TestHeavisideAPI_int32(TestHeavisideAPI_float64):
    def setUp(self):
        self.x_np = np.random.random((13, 17)).astype("int32")
        self.y_np = np.random.random((13, 17)).astype("int32")
        self.out_np = np.heaviside(self.x_np, self.y_np)
        self.dtype = "int32"

class TestHeavisideError(unittest.TestCase):
    def test_input(self):
        def test_input_x():
            with paddle.fluid.dygraph.guard():
                paddle.heaviside(1, paddle.randn([100]))
        self.assertRaises(ValueError, test_input_x)

        def test_input_y():
            with paddle.fluid.dygraph.guard():
                paddle.heaviside(paddle.randn([100]), 1)
        self.assertRaises(ValueError, test_input_y)

        def test_input_xy():
            with paddle.fluid.dygraph.guard():
                paddle.heaviside(paddle.randn([100], 'float32'),
                                 paddle.randn([100], 'float64'))
        self.assertRaises(ValueError, test_input_xy)

if __name__ == '__main__':
    unittest.main()
