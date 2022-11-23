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
import paddle.fluid as fluid
from op_test import OpTest
import paddle


class TestDygraphBilinearTensorProductAPIError(unittest.TestCase):

    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            layer = fluid.dygraph.nn.BilinearTensorProduct(input1_dim=5,
                                                           input2_dim=4,
                                                           output_dim=1000)
            # the input must be Variable.
            x0 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                         [[1, 1, 1, 1]], fluid.CPUPlace())
            self.assertRaises(TypeError, layer, x0)
            # the input dtype must be float32 or float64
            x1 = fluid.data(name='x1', shape=[-1, 5], dtype="float16")
            x2 = fluid.data(name='x2', shape=[-1, 4], dtype="float32")
            self.assertRaises(TypeError, layer, x1, x2)


class TestBilinearTensorProductOp(OpTest):

    def setUp(self):
        self.op_type = "bilinear_tensor_product"
        self.python_api = paddle.nn.functional.bilinear
        batch_size = 6
        size0 = 5
        size1 = 4
        size2 = 5
        dtype = "float32" if fluid.core.is_compiled_with_rocm() else "float64"
        a = np.random.random((batch_size, size0)).astype(dtype)
        b = np.random.random((batch_size, size1)).astype(dtype)
        w = np.random.random((size2, size0, size1)).astype(dtype)
        bias = np.random.random((1, size2)).astype(dtype)
        output = np.zeros((batch_size, size2)).astype(dtype)
        for i in range(size2):
            w_i = w[i, :, :]
            output[:, i] = np.sum(np.matmul(a, w_i) * b, axis=1)
        self.inputs = {
            'X': a,
            'Y': b,
            'Weight': w,
            'Bias': bias,
        }
        self.outputs = {'Out': output + bias}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y', 'Weight', 'Bias'], 'Out', check_eager=True)


if __name__ == "__main__":
    unittest.main()
