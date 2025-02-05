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


def output_i0e(x):
    return special.i0e(x)


def ref_i0e_grad(x, dout):
    eps = np.finfo(x.dtype).eps
    not_tiny = abs(x) > eps
    safe_x = np.where(not_tiny, x, eps)
    gradx = special.i1e(x) - np.sign(x) * output_i0e(safe_x)
    gradx = np.where(not_tiny, gradx, -1.0)
    return dout * gradx


class TestI0eAPI(unittest.TestCase):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5]

    def setUp(self):
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.out_ref = output_i0e(self.x)
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
                y = paddle.i0e(x)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": self.x},
                    fetch_list=[y],
                )
                np.testing.assert_allclose(self.out_ref, res[0], rtol=1e-5)
            paddle.disable_static()

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.i0e(x)

            out_ref = output_i0e(self.x)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            x = None
            self.assertRaises(ValueError, paddle.i0e, x)
            paddle.enable_static()


class TestI0eFloat32Zero2EightCase(TestI0eAPI):
    DTYPE = "float32"
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class TestI0eFloat32OverEightCase(TestI0eAPI):
    DTYPE = "float32"
    DATA = [9, 10, 11, 12]


class TestI0eFloat64Zero2EightCase(TestI0eAPI):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class TestI0eFloat64OverEightCase(TestI0eAPI):
    DTYPE = "float64"
    DATA = [9, 10, 11, 12]


class TestI0eOp(OpTest):
    def setUp(self) -> None:
        self.op_type = "i0e"
        self.python_api = paddle.i0e
        self.init_config()
        self.outputs = {"out": self.target}

    def init_config(self):
        self.dtype = np.float64
        zero_case = np.zeros(1).astype(self.dtype)
        rand_case = np.random.randn(100).astype(self.dtype)
        one2eight_case = np.random.uniform(low=1, high=8, size=100).astype(
            self.dtype
        )
        over_eight_case = np.random.uniform(low=9, high=15, size=100).astype(
            self.dtype
        )
        self.case = np.concatenate(
            [zero_case, rand_case, one2eight_case, over_eight_case]
        )
        self.inputs = {'x': self.case}
        self.target = output_i0e(self.inputs['x'])

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['x'],
            'out',
            user_defined_grads=[ref_i0e_grad(self.case, 1 / self.case.size)],
            check_pir=True,
        )


if __name__ == "__main__":
    unittest.main()
