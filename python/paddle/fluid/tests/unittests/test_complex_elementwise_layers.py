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
from paddle import complex as cpx
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg

layers = {
    "add": cpx.elementwise_add,
    "sub": cpx.elementwise_sub,
    "mul": cpx.elementwise_mul,
    "div": cpx.elementwise_div,
}

fluid_layers = {
    "add": fluid.layers.elementwise_add,
    "sub": fluid.layers.elementwise_sub,
    "mul": fluid.layers.elementwise_mul,
    "div": fluid.layers.elementwise_div,
}


class TestComplexElementwiseLayers(unittest.TestCase):
    def setUp(self):
        self._dtype = "float64"
        self._places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(fluid.CUDAPlace(0))

    def calc(self, x, y, layer_type, place):
        with dg.guard(place):
            var_x = dg.to_variable(x)
            var_y = dg.to_variable(y)
            return layers[layer_type](var_x, var_y).numpy()

    def fuild_calc(self, x, y, layer_type, place):
        with dg.guard(place):
            var_x = fluid.core.VarBase(
                value=x,
                place=fluid.framework._current_expected_place(),
                persistable=False,
                zero_copy=None,
                name='')
            var_y = fluid.core.VarBase(
                value=y,
                place=fluid.framework._current_expected_place(),
                persistable=False,
                zero_copy=None,
                name='')
            return fluid_layers[layer_type](var_x, var_y).numpy()

    def compare(self, x, y):
        for place in self._places:
            self.assertTrue(np.allclose(self.calc(x, y, "add", place), x + y))
            self.assertTrue(np.allclose(self.calc(x, y, "sub", place), x - y))
            self.assertTrue(np.allclose(self.calc(x, y, "mul", place), x * y))
            self.assertTrue(np.allclose(self.calc(x, y, "div", place), x / y))

    def compare_1(self, x, y):
        for place in self._places:
            self.assertTrue(
                np.allclose(self.fuild_calc(x, y, "add", place), x + y))
            self.assertTrue(
                np.allclose(self.fuild_calc(x, y, "sub", place), x - y))
            self.assertTrue(
                np.allclose(self.fuild_calc(x, y, "mul", place), x * y))
            self.assertTrue(
                np.allclose(self.fuild_calc(x, y, "div", place), x / y))

    def compare_op(self, x, y):
        for place in self._places:
            with dg.guard(place):
                var_x = dg.to_variable(x)
                var_y = dg.to_variable(y)
                self.assertTrue(var_x + var_y, x + y)
                self.assertTrue(var_x - var_y, x - y)
                self.assertTrue(var_x * var_y, x * y)
                self.assertTrue(var_x / var_y, x / y)

    def compare_op_1(self, x, y):
        for place in self._places:
            with dg.guard(place):
                var_x = fluid.core.VarBase(
                    value=x,
                    place=fluid.framework._current_expected_place(),
                    persistable=False,
                    zero_copy=None,
                    name='')
                var_y = fluid.core.VarBase(
                    value=y,
                    place=fluid.framework._current_expected_place(),
                    persistable=False,
                    zero_copy=None,
                    name='')
                self.assertTrue(np.allclose((var_x + var_y).numpy(), x + y))
                self.assertTrue(np.allclose((var_x - var_y).numpy(), x - y))
                self.assertTrue(np.allclose((var_x * var_y).numpy(), x * y))
                self.assertTrue(np.allclose((var_x / var_y).numpy(), x / y))

    def test_complex_xy(self):
        x = rand([2, 3, 4, 5]).astype(self._dtype) + 1j * rand(
            [2, 3, 4, 5]).astype(self._dtype)
        y = rand([2, 3, 4, 5]).astype(self._dtype) + 1j * rand(
            [2, 3, 4, 5]).astype(self._dtype)
        self.compare(x, y)
        self.compare_op(x, y)
        self.compare_1(x, y)
        self.compare_op_1(x, y)

    def test_complex_x_real_y(self):
        x = rand([2, 3, 4, 5]).astype(self._dtype) + 1j * rand(
            [2, 3, 4, 5]).astype(self._dtype)
        y = rand([4, 5]).astype(self._dtype)
        self.compare(x, y)
        self.compare_op(x, y)

    def test_real_x_complex_y(self):
        x = rand([2, 3, 4, 5]).astype(self._dtype)
        y = rand([5]).astype(self._dtype) + 1j * rand([5]).astype(self._dtype)
        self.compare(x, y)
        self.compare_op(x, y)


if __name__ == '__main__':
    unittest.main()
