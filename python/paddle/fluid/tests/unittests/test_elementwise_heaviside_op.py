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
from op_test import OpTest
import paddle


def Heaviside_grad(x, y, dout):
    tmp = np.zeros(x.shape).astype("float16")
    dx = np.multiply(tmp, dout)
    dy = np.multiply(np.equal(x, 0), dout).astype("float16")
    return dx, dy


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
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))


class TestHeavisideBroadcast(unittest.TestCase):

    def setUp(self):
        self.input_1 = np.random.rand(2, 100, 13, 17).astype("float32")
        self.input_2 = np.random.rand(100, 13, 17).astype("float32")
        self.input_3 = np.random.rand(100, 13, 1).astype("float32")
        self.input_4 = np.random.rand(13, 17).astype("float32")
        self.input_5 = np.random.rand(1).astype("float32")

        self.np_expected1 = np.heaviside(self.input_1, self.input_2)
        self.np_expected2 = np.heaviside(self.input_2, self.input_3)
        self.np_expected3 = np.heaviside(self.input_2, self.input_4)
        self.np_expected4 = np.heaviside(self.input_4, self.input_5)

    def test_broadcast(self):
        paddle.disable_static()
        self.tensor_1 = paddle.to_tensor(self.input_1)
        self.tensor_2 = paddle.to_tensor(self.input_2)
        self.tensor_3 = paddle.to_tensor(self.input_3)
        self.tensor_4 = paddle.to_tensor(self.input_4)
        self.tensor_5 = paddle.to_tensor(self.input_5)

        res = paddle.heaviside(self.tensor_1, self.tensor_2)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        res = paddle.heaviside(self.tensor_2, self.tensor_3)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        res = paddle.heaviside(self.tensor_2, self.tensor_4)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        res = paddle.heaviside(self.tensor_4, self.tensor_5)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected4, rtol=1e-05)


class TestHeavisideAPI_float64(unittest.TestCase):

    def setUp(self):
        self.x_np = np.random.random((13, 17)).astype("float64")
        self.y_np = np.random.random((13, 17)).astype("float64")
        self.out_np = np.heaviside(self.x_np, self.y_np)
        self.dtype = "float64"

    def test_static(self):
        for use_cuda in ([False, True]
                         if paddle.device.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            prog = paddle.static.Program()
            with paddle.static.program_guard(prog):
                x = paddle.static.data(name=f"x_{self.dtype}",
                                       shape=[13, 17],
                                       dtype=self.dtype)
                y = paddle.static.data(name=f"y_{self.dtype}",
                                       shape=[13, 17],
                                       dtype=self.dtype)
                out = paddle.heaviside(x, y)

            exe = paddle.static.Executor(place=place)
            res, = exe.run(prog,
                           feed={
                               f"x_{self.dtype}": self.x_np,
                               f"y_{self.dtype}": self.y_np
                           },
                           fetch_list=out,
                           use_prune=True)

            np.testing.assert_allclose(res, self.out_np, rtol=1e-05)

    def test_dygraph(self):
        for use_cuda in ([False, True]
                         if paddle.device.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
            result = paddle.heaviside(paddle.to_tensor(self.x_np),
                                      paddle.to_tensor(self.y_np))

            np.testing.assert_allclose(result.numpy(), self.out_np, rtol=1e-05)


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


class TestHeavisideAPI_float16(OpTest):

    def setUp(self):
        self.dtype = np.float16
        self.op_type = "elementwise_heaviside"
        self.python_api = paddle.heaviside
        self.inputs = {
            'X': np.random.uniform(1, 2, [20, 5]).astype("float16"),
            'Y': np.random.uniform(1, 2, [20, 5]).astype("float16")
        }
        self.outputs = {'Out': np.heaviside(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        user_defined_grads=Heaviside_grad(
                            self.inputs['X'], self.inputs['Y'],
                            1 / self.inputs['X'].size),
                        check_eager=True)


class TestHeavisideError(unittest.TestCase):

    def test_input(self):
        paddle.disable_static()

        def test_input_x():
            paddle.heaviside(1, paddle.randn([100]))

        self.assertRaises(ValueError, test_input_x)

        def test_input_y():
            paddle.heaviside(paddle.randn([100]), 1)

        self.assertRaises(ValueError, test_input_y)

        def test_input_xy():
            paddle.heaviside(paddle.randn([100], 'float32'),
                             paddle.randn([100], 'float64'))

        self.assertRaises(ValueError, test_input_xy)


if __name__ == '__main__':
    unittest.main()
