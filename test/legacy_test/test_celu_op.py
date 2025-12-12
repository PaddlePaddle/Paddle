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
from op_test import get_device_place, get_places, is_custom_device

import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base, nn


def celu(x, alpha):
    y_ref = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return y_ref.astype(x.dtype)


class TestCELUOpClass_Inplace(unittest.TestCase):
    def _test_case1_cpu(self):
        x = np.random.uniform(-1, 1, size=(15, 17)).astype(np.float32)
        alpha = 1.0
        y_ref = celu(x, alpha)

        place = base.CPUPlace()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            y_var1 = F.celu(x_var1, alpha, True)
            y_test1 = y_var1.numpy()

            func = nn.CELU(alpha, True)
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
        alpha = 1.0
        y_ref = celu(x, alpha)

        place = get_device_place()
        with dg.guard(place) as g:
            x_var1 = paddle.to_tensor(x)
            x_var2 = paddle.to_tensor(x)

            y_var1 = F.celu(x_var1, alpha, True)
            y_test1 = y_var1.numpy()

            func = nn.CELU(alpha, True)
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


class TestCELUParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.random((10, 3, 4)).astype("float64")
        self.alpha = 1.0
        self.test_types = ["decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        if test_type == 'raw':
            result = F.celu(x, self.alpha, False)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator':
            result = F.celu(x=x, alpha=self.alpha, inplace=False)
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


class TestCELUAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [10, 10]
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.alpha = 1.0
        self.place = get_places()
        self.x_feed = copy.deepcopy(self.x_np)

    def test_api_static(self):
        paddle.enable_static()

        def run(place, inplace):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape)
                out = F.celu(x, self.alpha, inplace)
                exe = paddle.static.Executor(self.place[0])
                res = exe.run(
                    feed={
                        'X': self.x_feed,
                    },
                    fetch_list=[out],
                )
            target = copy.deepcopy(self.x_np)
            out_ref = celu(target, self.alpha)

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place, True)
            run(place, False)

    def test_api_dygraph(self):
        def run(place, inplace):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            out = F.celu(x_tensor, self.alpha, inplace)

            target = copy.deepcopy(self.x_np)
            out_ref = celu(target, self.alpha)

            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place, True)
            run(place, False)


if __name__ == '__main__':
    unittest.main()
