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

import unittest

import numpy as np
from eager_op_test import OpTest
from scipy import special

import paddle
from paddle.fluid import core

np.random.seed(100)
paddle.seed(100)


def ref_polygamma(x, n):
    return special.polygamma(n, x)


def ref_polygamma_grad(x, dout, n):
    gradx = special.polygamma(n + 1, x)
    return dout * gradx


class TestPolygammaAPI(unittest.TestCase):
    DTYPE = "float64"
    DATA = [0, 1, 2, 3, 4, 5]
    ORDER = 0

    def setUp(self):
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.place = [paddle.CPUPlace()]
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


class TestPolygammaFloat32Zero2EightCase(TestPolygammaAPI):
    DTYPE = "float32"
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class TestPolygammaOp(OpTest):
    def setUp(self) -> None:
        self.op_type = "polygamma"
        self.python_api = paddle.polygamma
        self.init_config()
        self.outputs = {"out": self.target}

    def init_config(self):
        self.dtype = np.float64
        self.order = 0
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
        self.attrs = {'n': self.order}
        self.target = ref_polygamma(self.inputs['x'], self.order)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['x'],
            'out',
            user_defined_grads=[ref_polygamma_grad(self.case, 1 / self.case.size, self.order)],
        )


if __name__ == "__main__":
    unittest.main()
