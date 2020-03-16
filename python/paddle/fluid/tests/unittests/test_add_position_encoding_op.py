#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import numpy as np
import math
import paddle.fluid.core as core
from op_test import OpTest


class TestAddPositionEncodingTensorOp(OpTest):
    """
    This class is to test the AddPositionEncodingOp
    """

    def setUp(self):
        """
        the prepared section for add position encoding op
        """
        self.op_type = "add_position_encoding"
        self.dtype = np.float64
        self.init_input_output()

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x), }
        self.outputs = {'Out': self.out}
        self.attrs = {'alpha': self.alpha, 'beta': self.beta}

    def test_check_output(self):
        """
        check the correctness of output
        """
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        """
        check the correctness of grad
        """
        self.check_grad(['X'], 'Out', check_dygraph=False)

    def init_input_output(self):
        """
        init the input and output for test cases
        """
        self.alpha = 0.6
        self.beta = 0.5
        self.x = np.random.uniform(0.1, 1, [2, 15, 4]).astype(self.dtype)
        self.out = np.copy(self.x)

        batch_size = self.x.shape[0]
        max_length = self.x.shape[1]
        enc_size = self.x.shape[2]

        half_shape = int(enc_size / 2)
        for i in range(batch_size):
            for j in range(max_length):
                for k in range(half_shape):
                    val = j / pow(10000.0, k * 1.0 / (
                        half_shape - 1)) if half_shape > 1 else j / 10000.0
                    self.out[i, j, k] = \
                        self.x[i, j, k] * self.alpha + math.sin(val) * self.beta
                    self.out[i, j, half_shape + k] = \
                        self.x[i, j, half_shape + k] * self.alpha + math.cos(val) * self.beta


class TestAddPositionEncodingLoDTensorOp(OpTest):
    """
    This class is to test the AddPositionEncodingLoDTensorOp
    """

    def setUp(self):
        """
        the prepared section for add position encoding LoDTensor op
        """
        self.op_type = "add_position_encoding"
        self.dtype = np.float64
        self.init_input_output()

        self.inputs = {'X': (self.x, self.lod), }
        self.outputs = {'Out': (self.out, self.lod)}
        self.attrs = {'alpha': self.alpha, 'beta': self.beta}

    def test_check_output(self):
        """
        check the correctness of output
        """
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        """
        check the correctness of grad
        """
        self.check_grad(['X'], 'Out', check_dygraph=False)

    def init_input_output(self):
        """
        init the input and output for test cases
        """
        self.alpha = 0.6
        self.beta = 0.5
        self.x = np.random.uniform(0.1, 1, [20, 6]).astype(self.dtype)
        self.lod = [[13, 7]]
        self.out = np.copy(self.x)

        batch_size = len(self.lod[0])
        enc_size = self.x.shape[1]

        start = 0
        half_shape = int(enc_size / 2)
        for i in range(batch_size):
            max_length = self.lod[0][i]
            for j in range(max_length):
                for k in range(half_shape):
                    val = j / pow(10000.0, k * 1.0 / (
                        half_shape - 1)) if half_shape > 1 else j / 10000.0
                    pos = start + j
                    self.out[pos, k] = \
                        self.x[pos, k] * self.alpha + math.sin(val) * self.beta
                    self.out[pos, half_shape + k] = \
                        self.x[pos, half_shape + k] * self.alpha + math.cos(val) * self.beta
            start += max_length


if __name__ == '__main__':
    unittest.main()
