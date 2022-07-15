#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

def np_sgn(x:np.ndarray):
    if x.dtype == 'complex128' or x.dtype == 'complex64':
        x_abs = np.abs(x)
        eps = np.finfo(x.dtype).eps
        x_abs = np.maximum(x_abs, eps)
        out = x / x_abs
    else:
        out = np.sign(x)
    return out

class TestSgn(OpTest):

    def setUp(self):
        self.op_type = "sgn"
        self.inputs = {
            'X': np.random.uniform(-10, 10, (10, 10)).astype("complex128")}
        self.outputs = {'Out': np_sgn(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSgnError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.sign, input1)
            # The input dtype of sgn must be float16, float32, float64,complex64,complex128.
            input2 = fluid.layers.data(name='input2',
                                       shape=[12, 10],
                                       dtype="int32")
            input3 = fluid.layers.data(name='input3',
                                       shape=[12, 10],
                                       dtype="int64")
            self.assertRaises(TypeError, fluid.layers.sign, input2)
            self.assertRaises(TypeError, fluid.layers.sign, input3)
            input4 = fluid.layers.data(name='input4',
                                       shape=[4],
                                       dtype="complex64")
            fluid.layers.sign(input4)


class TestSignAPI(unittest.TestCase):

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([3+4j, 7-24j, 0, 1+2j], dtype='complex64')
            x = paddle.to_tensor(np_x)
            z = paddle.sgn(x)
            np_z = z.numpy()
            z_expected = np_sgn(np_x)
            self.assertEqual((np_z == z_expected).all(), True)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
