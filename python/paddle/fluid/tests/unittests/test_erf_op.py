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
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg


class TestErfOp(OpTest):

    def setUp(self):
        self.op_type = "erf"
        self.dtype = self._init_dtype()
        self.x_shape = [11, 17]
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        y_ref = erf(x).astype(self.dtype)
        self.inputs = {'X': x}
        self.outputs = {'Out': y_ref}

    def _init_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestErfLayer(unittest.TestCase):

    def _test_case(self, place):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float64)
        y_ref = erf(x)
        with dg.guard(place) as g:
            x_var = dg.to_variable(x)
            y_var = fluid.layers.erf(x_var)
            y_test = y_var.numpy()
        np.testing.assert_allclose(y_ref, y_test, rtol=1e-05)

    def test_case(self):
        self._test_case(fluid.CPUPlace())
        if fluid.is_compiled_with_cuda():
            self._test_case(fluid.CUDAPlace(0))

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = paddle.static.data('x', [3, 4])
            y = paddle.erf(x, name='erf')
            self.assertTrue('erf' in y.name)


if __name__ == '__main__':
    unittest.main()
