#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from scipy import special

import paddle
from paddle.base import core

np.random.seed(100)
paddle.seed(100)


def ref_polygamma(x, n):
    """
    The case where x = 0 differs from
    the current mainstream implementation,
    and requires specifying a special value point.
    """
    mask = x == 0
    if n == 0:
        out = special.psi(x)
        out[mask] = np.nan
    else:
        out = special.polygamma(n, x)
    return out


def ref_polygamma_grad(x, dout, n):
    """
    The case where x = 0 differs from
    the current mainstream implementation,
    and requires specifying a special value point.
    """
    mask = x == 0
    gradx = special.polygamma(n + 1, x)
    if n == 0:
        gradx[mask] = np.nan
    return dout * gradx


class TestPolygammaAPI(unittest.TestCase):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5]
    ORDER = 1

    def setUp(self):
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        def run(place):
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype=self.DTYPE
                )
                y = paddle.polygamma(x, self.ORDER)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x},
                    fetch_list=[y],
                )
                out_ref = ref_polygamma(self.x, self.ORDER)
                np.testing.assert_allclose(out_ref, res[0], rtol=1e-5)
            paddle.disable_static()

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.polygamma(x, self.ORDER)

            out_ref = ref_polygamma(self.x, self.ORDER)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            x = None
            self.assertRaises(ValueError, paddle.polygamma, x, self.ORDER)
            paddle.enable_static()

    def test_input_type_error(self):
        for place in self.place:
            paddle.disable_static(place)
            self.assertRaises(
                TypeError, paddle.polygamma, self.x, float(self.ORDER)
            )
            paddle.enable_static()

    def test_negative_order_error(self):
        for place in self.place:
            paddle.disable_static(place)
            self.assertRaises(ValueError, paddle.polygamma, self.x, -self.ORDER)
            paddle.enable_static()


class TestPolygammaFloat32Order1(TestPolygammaAPI):
    DTYPE = "float32"
    DATA = [2, 3, 5, 2.25, 7, 7.25]
    ORDER = 1


class TestPolygammaFloat32Order2(TestPolygammaAPI):
    DTYPE = "float32"
    DATA = [2, 3, 5, 2.25, 7, 7.25]
    ORDER = 2


class TestPolygammaFloat32Order3(TestPolygammaAPI):
    DTYPE = "float32"
    DATA = [2, 3, 5, 2.25, 7, 7.25]
    ORDER = 3


class TestPolygammaFloat64Order1(TestPolygammaAPI):
    DTYPE = "float64"
    DATA = [2, 3, 5, 2.25, 7, 7.25]
    ORDER = 1


class TestPolygammaFloat64Order2(TestPolygammaAPI):
    DTYPE = "float64"
    DATA = [2, 3, 5, 2.25, 7, 7.25]
    ORDER = 2


class TestPolygammaFloat64Order3(TestPolygammaAPI):
    DTYPE = "float64"
    DATA = [2, 3, 5, 2.25, 7, 7.25]
    ORDER = 3


class TestPolygammaNegativeInputOrder1(TestPolygammaAPI):
    DTYPE = "float64"
    DATA = [-2, 3, 5, 2.25, 7, 7.25]
    ORDER = 1


class TestPolygammaMultiDimOrder1(TestPolygammaAPI):
    DTYPE = "float64"
    DATA = [[-2, 3, 5, 2.25, 7, 7.25], [0, 1, 2, 3, 4, 5]]
    ORDER = 1


class TestPolygammaMultiDimOrder2(TestPolygammaAPI):
    DTYPE = "float64"
    DATA = [
        [[-2, 3, 5, 2.25, 7, 7.25], [0, 1, 2, 3, 4, 5]],
        [[6, 7, 8, 9, 1, 2], [0, 1, 2, 3, 4, 5]],
    ]
    ORDER = 2


class TestPolygammaOp(OpTest):
    def setUp(self) -> None:
        self.op_type = "polygamma"
        self.python_api = paddle.polygamma
        self.init_config()
        self.outputs = {"out": self.target}

    def init_config(self):
        self.dtype = np.float64
        self.order = 1
        rand_case = np.random.randn(100).astype(self.dtype)
        int_case = np.random.randint(low=1, high=100, size=100).astype(
            self.dtype
        )
        self.case = np.concatenate([rand_case, int_case])
        self.inputs = {'x': self.case}
        self.attrs = {'n': self.order}
        self.target = ref_polygamma(self.inputs['x'], self.order)

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['x'],
            'out',
            user_defined_grads=[
                ref_polygamma_grad(self.case, 1 / self.case.size, self.order)
            ],
            check_pir=True,
        )


if __name__ == "__main__":
    unittest.main()
