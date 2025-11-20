#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np
from op_test import OpTest, get_device_place, get_places, is_custom_device

import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base, nn


def silu(x):
    y_ref = x * (1 / (1 + np.exp(-x)))
    return y_ref.astype(x.dtype)


class TestSiluOpClass(unittest.TestCase):
    def _test_case1_cpu(self):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = silu(x)

        place = base.CPUPlace()
        with dg.guard(place) as g:
            x_var = paddle.to_tensor(x)
            y_var1 = F.silu(x_var)
            y_test1 = y_var1.numpy()

            func = nn.Silu()
            y_var2 = func(x_var)
            y_test2 = y_var2.numpy()
        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)

    def _test_case1_gpu(self):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = silu(x)

        place = get_device_place()
        with dg.guard(place) as g:
            x_var = paddle.to_tensor(x)
            y_var1 = F.silu(x_var)
            y_test1 = y_var1.numpy()

            func = nn.Silu()
            y_var2 = func(x_var)
            y_test2 = y_var2.numpy()
        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)

    def test_cases(self):
        self._test_case1_cpu()
        if base.is_compiled_with_cuda() or is_custom_device():
            self._test_case1_gpu()

    def test_fast_math(self):
        if not (paddle.is_compiled_with_cuda() or is_custom_device()):
            return

        def use_fast_math(enabled):
            paddle.set_flags({'FLAGS_use_fast_math': enabled})

        shape = [11, 17, 8]
        x_np = np.random.uniform(-1, 1, size=shape).astype(np.float16)
        y_g_np = np.random.uniform(-1, 1, size=shape).astype(np.float16)

        def run_silu_op():
            with dg.guard():
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                y = F.silu(x)
                x_grad = paddle.grad([y], [x], [paddle.to_tensor(y_g_np)])[0]
                return y.numpy(), x_grad.numpy()

        def run_silu_class():
            with dg.guard():
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                func = nn.Silu()
                y = func(x)
                x_grad = paddle.grad([y], [x], [paddle.to_tensor(y_g_np)])[0]
                return y.numpy(), x_grad.numpy()

        use_fast_math(True)
        y_fast_math1, x_g_fast_math1 = run_silu_op()
        y_fast_math2, x_g_fast_math2 = run_silu_class()
        use_fast_math(False)

        y_ref1, x_g_ref1 = run_silu_op()
        y_ref2, x_g_ref2 = run_silu_class()
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


class TestSiluOpClass_ZeroSize(unittest.TestCase):
    def _test_case1_cpu(self):
        x = np.random.uniform(-1, 1, size=(0, 17)).astype(np.float32)
        y_ref = silu(x)

        place = base.CPUPlace()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            x_var1.stop_gradient = False
            x_var2.stop_gradient = False

            y_var1 = F.silu(x_var1)
            y_test1 = y_var1.numpy()

            func = nn.Silu()
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

    def _test_case1_gpu(self):
        x = np.random.uniform(-1, 1, size=(0, 17)).astype(np.float32)
        y_ref = silu(x)

        place = get_device_place()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            x_var1.stop_gradient = False
            x_var2.stop_gradient = False

            y_var1 = F.silu(x_var1)
            y_test1 = y_var1.numpy()

            func = nn.Silu()
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
        self._test_case1_cpu()
        if base.is_compiled_with_cuda() or is_custom_device():
            self._test_case1_gpu()


class TestSiluOpClass_Inplace(unittest.TestCase):
    def _test_case1_cpu(self):
        x = np.random.uniform(-1, 1, size=(15, 17)).astype(np.float32)
        y_ref = silu(x)

        place = base.CPUPlace()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            y_var1 = F.silu(x_var1, True)
            y_test1 = y_var1.numpy()

            func = nn.Silu(True)
            y_var2 = func(x_var2)
            y_test2 = y_var2.numpy()

        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)

        np.testing.assert_allclose(
            y_ref, x_var1.numpy(), rtol=1e-05, atol=1e-08
        )
        np.testing.assert_allclose(
            y_ref, x_var2.numpy(), rtol=1e-05, atol=1e-08
        )

    def _test_case1_gpu(self):
        x = np.random.uniform(-1, 1, size=(15, 17)).astype(np.float32)
        y_ref = silu(x)

        place = get_device_place()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            y_var1 = F.silu(x_var1, True)
            y_test1 = y_var1.numpy()

            func = nn.Silu(True)
            y_var2 = func(x_var2)
            y_test2 = y_var2.numpy()

        np.testing.assert_allclose(y_ref, y_test1, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(y_ref, y_test2, rtol=1e-05, atol=1e-08)

        np.testing.assert_allclose(
            y_ref, x_var1.numpy(), rtol=1e-05, atol=1e-08
        )
        np.testing.assert_allclose(
            y_ref, x_var2.numpy(), rtol=1e-05, atol=1e-08
        )

    def test_cases(self):
        self._test_case1_cpu()
        if base.is_compiled_with_cuda() or is_custom_device():
            self._test_case1_gpu()


class TestSiluParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.random((10, 3, 4)).astype("float64")
        self.test_types = ["decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        if test_type == 'raw':
            result = F.silu(x, False)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator':
            result = F.silu(input=x, inplace=False)
            result.mean().backward()
            return result, x.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_all(self):
        out_std, grad_x_std = self.do_test('raw')
        for test_type in self.test_types:
            out, grad_x = self.do_test(test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-7)
            np.testing.assert_allclose(
                grad_x.numpy(), grad_x_std.numpy(), rtol=1e-7
            )


class TestSiluPrint(unittest.TestCase):
    def test_print(self):
        print(nn.Silu())
        print(nn.Silu(True))
        print(nn.Silu(False))
        print(nn.Silu(inplace=True))
        print(nn.Silu(inplace=False))


class SiluOpDefaultTest(OpTest):
    """the base class of other op testcases"""

    def setUp(self):
        self.initTestCase()
        self.python_api = F.silu

        self.op_type = "silu"
        self.inputs = {'X': self.X}

        self.target = copy.deepcopy(self.X)
        self.target = silu(self.target)
        self.outputs = {'Out': (self.target)}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_dtype(self):
        self.dtype = np.float64

    def initTestCase(self):
        self.init_dtype()
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.X = (
                np.random.uniform(-1, 1, [10, 10])
                + 1j * np.random.uniform(-1, 1, [10, 10])
            ).astype(self.dtype)


class SiluOpDefaultTestFP16(SiluOpDefaultTest):
    def init_dtype(self):
        self.dtype = np.float16


class SiluOpDefaultTestComplex_64(SiluOpDefaultTest):
    def init_dtype(self):
        self.dtype = np.complex64


class SiluOpDefaultTestComplex_128(SiluOpDefaultTest):
    def init_dtype(self):
        self.dtype = np.complex128


class TestSiluAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [10, 10]
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()
        self.x_feed = copy.deepcopy(self.x_np)

    def test_api_static(self):
        paddle.enable_static()

        def run(place, inplace):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape)
                out = F.silu(x, inplace)
                exe = paddle.static.Executor(self.place[0])
                res = exe.run(
                    feed={
                        'X': self.x_feed,
                    },
                    fetch_list=[out],
                )
            target = copy.deepcopy(self.x_np)
            out_ref = silu(target)

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place, True)
            run(place, False)

    def test_api_dygraph(self):
        def run(place, inplace):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            out = F.silu(x_tensor, inplace)

            target = copy.deepcopy(self.x_np)
            out_ref = silu(target)

            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place, True)
            run(place, False)


if __name__ == '__main__':
    unittest.main()
