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
from eager_op_test import OpTest, convert_float_to_uint16
from scipy.special import erf

import paddle
import paddle.fluid.dygraph as dg
from paddle import fluid


class TestErfOp(OpTest):
    def setUp(self):
        self.op_type = "erf"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.erf
        self.python_api = paddle.erf
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
        self.check_grad(['X'], 'Out', check_prim=True)


class TestErfLayer(unittest.TestCase):
    def _test_case(self, place):
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float64)
        y_ref = erf(x)
        with dg.guard(place) as g:
            x_var = dg.to_variable(x)
            y_var = paddle.erf(x_var)
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


class TestErfFP16OP(OpTest):
    def setUp(self):
        self.op_type = "erf"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.erf
        self.python_api = paddle.erf
        self.dtype = np.float16
        self.x_shape = [11, 17]
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        y_ref = erf(x).astype(self.dtype)
        self.inputs = {'X': x}
        self.outputs = {'Out': y_ref}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True)


@unittest.skipIf(
    not paddle.fluid.core.is_compiled_with_cuda()
    or not paddle.fluid.core.is_bfloat16_supported(
        paddle.fluid.core.CUDAPlace(0)
    ),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestErfBF16OP(OpTest):
    def setUp(self):
        self.op_type = "erf"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.erf
        self.python_api = paddle.erf
        self.dtype = np.uint16
        self.x_shape = [11, 17]
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(np.float32)
        y_ref = erf(x).astype(np.float32)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(y_ref)}

    def test_check_output(self):
        place = paddle.fluid.core.CUDAPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = paddle.fluid.core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True)


if __name__ == '__main__':
    unittest.main()
