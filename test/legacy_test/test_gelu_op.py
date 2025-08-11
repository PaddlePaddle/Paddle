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
from scipy.special import erf

import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base, nn


def gelu(x, approximate):
    if approximate == "tanh":
        approximate = True
    if approximate == "none":
        approximate = False
    if approximate:
        y_ref = (
            0.5
            * x
            * (
                1.0
                + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
            )
        )
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


class TestGeluOp(unittest.TestCase):
    def _test_case1_cpu(self, approximate):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)

        place = base.CPUPlace()
        with dg.guard(place) as g:
            x_var = paddle.to_tensor(x)
            y_var1 = F.gelu(x_var, approximate)
            y_test1 = y_var1.numpy()

            func = nn.GELU(approximate)
            y_var2 = func(x_var)
            y_test2 = y_var2.numpy()
        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)

    def _test_case1_gpu(self, approximate):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)

        place = base.CUDAPlace(0)
        with dg.guard(place) as g:
            x_var = paddle.to_tensor(x)
            y_var1 = F.gelu(x_var, approximate)
            y_test1 = y_var1.numpy()

            func = nn.GELU(approximate)
            y_var2 = func(x_var)
            y_test2 = y_var2.numpy()
        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)

    def test_cases(self):
        for approximate in [True, False, "none", "tanh"]:
            self._test_case1_cpu(approximate)
            if base.is_compiled_with_cuda():
                self._test_case1_gpu(approximate)

    def test_fast_math(self):
        if not paddle.is_compiled_with_cuda():
            return

        def use_fast_math(enabled):
            paddle.set_flags({'FLAGS_use_fast_math': enabled})

        shape = [11, 17, 8]
        x_np = np.random.uniform(-1, 1, size=shape).astype(np.float16)
        y_g_np = np.random.uniform(-1, 1, size=shape).astype(np.float16)

        def run_gelu_op(approximate):
            with dg.guard():
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                y = F.gelu(x, approximate=approximate)
                x_grad = paddle.grad([y], [x], [paddle.to_tensor(y_g_np)])[0]
                return y.numpy(), x_grad.numpy()

        def run_gelu_class(approximate):
            with dg.guard():
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                func = nn.GELU(approximate=approximate)
                y = func(x)
                x_grad = paddle.grad([y], [x], [paddle.to_tensor(y_g_np)])[0]
                return y.numpy(), x_grad.numpy()

        use_fast_math(True)
        y_fast_math1, x_g_fast_math1 = run_gelu_op(True)
        y_fast_math2, x_g_fast_math2 = run_gelu_class(True)
        use_fast_math(False)

        y_ref1, x_g_ref1 = run_gelu_op(True)
        y_ref2, x_g_ref2 = run_gelu_class(True)
        np.testing.assert_allclose(
            y_ref1, y_fast_math1, rtol=1e-05, atol=0.0005
        )

        np.testing.assert_allclose(
            x_g_ref1, x_g_fast_math1, rtol=1e-05, atol=0.0005
        )

        np.testing.assert_allclose(
            y_ref2, y_fast_math2, rtol=1e-05, atol=0.0005
        )

        np.testing.assert_allclose(
            x_g_ref2, x_g_fast_math2, rtol=1e-05, atol=0.0005
        )


class TestGeluOp_ZeroSize(unittest.TestCase):
    def _test_case1_cpu(self, approximate):
        x = np.random.uniform(-1, 1, size=(0, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)

        place = base.CPUPlace()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            x_var1.stop_gradient = False
            x_var2.stop_gradient = False

            y_var1 = F.gelu(x_var1, approximate)
            y_test1 = y_var1.numpy()

            func = nn.GELU(approximate)
            y_var2 = func(x_var2)
            y_test2 = y_var2.numpy()

            loss1 = paddle.sum(y_var1)
            loss1.backward()

            loss2 = paddle.sum(y_var2)
            loss2.backward()
        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(x_var1.grad.shape, x_var1.shape)

        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(x_var2.grad.shape, x_var2.shape)

    def _test_case1_gpu(self, approximate):
        x = np.random.uniform(-1, 1, size=(0, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)

        place = base.CUDAPlace(0)
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            x_var1.stop_gradient = False
            x_var2.stop_gradient = False

            y_var1 = F.gelu(x_var1, approximate)
            y_test1 = y_var1.numpy()

            func = nn.GELU(approximate)
            y_var2 = func(x_var2)
            y_test2 = y_var2.numpy()

            loss1 = paddle.sum(y_var1)
            loss1.backward()

            loss2 = paddle.sum(y_var2)
            loss2.backward()
        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(x_var1.grad.shape, x_var1.shape)

        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(x_var2.grad.shape, x_var2.shape)

    def test_cases(self):
        for approximate in [True, False, "none", "tanh"]:
            self._test_case1_cpu(approximate)
            if base.is_compiled_with_cuda():
                self._test_case1_gpu(approximate)


class TestGeluError(unittest.TestCase):

    def setUp(self):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        self.x = paddle.to_tensor(x)

    def test_gelu_op_error(self):

        def test_type_error1():
            y = F.gelu(self.x, "tan")

        def test_type_error2():
            y = F.gelu(self.x, 1234)

        self.assertRaises(TypeError, test_type_error1)
        self.assertRaises(TypeError, test_type_error2)

    def test_gelu_class_error(self):

        def test_type_error1():
            func = nn.GELU("tan")
            y = func(self.x)

        def test_type_error2():
            func = nn.GELU(1234)
            y = func(self.x)

        self.assertRaises(TypeError, test_type_error1)
        self.assertRaises(TypeError, test_type_error2)


if __name__ == '__main__':
    unittest.main()
