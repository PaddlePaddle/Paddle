#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from numpy.random import random as rand

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg

paddle_apis = {
    "add": paddle.add,
    "sub": paddle.subtract,
    "mul": paddle.multiply,
    "div": paddle.divide,
}


class TestComplexElementwiseLayers(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def paddle_calc(self, x, y, op, place):
        with dg.guard(place):
            x_t = dg.to_variable(x)
            y_t = dg.to_variable(y)
            return paddle_apis[op](x_t, y_t).numpy()

    def assert_check(self, pd_result, np_result, place):
        np.testing.assert_allclose(
            pd_result,
            np_result,
            rtol=1e-05,
            err_msg='\nplace: {}\npaddle diff result:\n {}\nnumpy diff result:\n {}\n'.format(
                place,
                pd_result[~np.isclose(pd_result, np_result)],
                np_result[~np.isclose(pd_result, np_result)],
            ),
        )

    def compare_by_basic_api(self, x, y):
        for place in self._places:
            self.assert_check(
                self.paddle_calc(x, y, "add", place), x + y, place
            )
            self.assert_check(
                self.paddle_calc(x, y, "sub", place), x - y, place
            )
            self.assert_check(
                self.paddle_calc(x, y, "mul", place), x * y, place
            )
            self.assert_check(
                self.paddle_calc(x, y, "div", place), x / y, place
            )

    def compare_op_by_basic_api(self, x, y):
        for place in self._places:
            with dg.guard(place):
                var_x = dg.to_variable(x)
                var_y = dg.to_variable(y)
                self.assert_check((var_x + var_y).numpy(), x + y, place)
                self.assert_check((var_x - var_y).numpy(), x - y, place)
                self.assert_check((var_x * var_y).numpy(), x * y, place)
                self.assert_check((var_x / var_y).numpy(), x / y, place)

    def test_complex_xy(self):
        for dtype in self._dtypes:
            x = rand([2, 3, 4, 5]).astype(dtype) + 1j * rand(
                [2, 3, 4, 5]
            ).astype(dtype)
            y = rand([2, 3, 4, 5]).astype(dtype) + 1j * rand(
                [2, 3, 4, 5]
            ).astype(dtype)

            self.compare_by_basic_api(x, y)
            self.compare_op_by_basic_api(x, y)

    def test_complex_x_real_y(self):
        for dtype in self._dtypes:
            x = rand([2, 3, 4, 5]).astype(dtype) + 1j * rand(
                [2, 3, 4, 5]
            ).astype(dtype)
            y = rand([4, 5]).astype(dtype)

            # promote types cases
            self.compare_by_basic_api(x, y)
            self.compare_op_by_basic_api(x, y)

    def test_real_x_complex_y(self):
        for dtype in self._dtypes:
            x = rand([2, 3, 4, 5]).astype(dtype)
            y = rand([5]).astype(dtype) + 1j * rand([5]).astype(dtype)

            # promote types cases
            self.compare_by_basic_api(x, y)
            self.compare_op_by_basic_api(x, y)


if __name__ == '__main__':
    unittest.main()
