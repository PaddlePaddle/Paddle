#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from math import log
from math import exp
from op_test import OpTest
import unittest


def cvm_compute(input, output, item_width, use_cvm):

    cvm_offset = 0 if use_cvm else 2
    batch_size = input.shape[0]

    for idx in range(batch_size):
        if use_cvm:
            output[idx] = input[idx]
            output[idx][0] = log(output[idx][0] + 1)
            output[idx][1] = log(output[idx][1] + 1) - output[idx][0]
        else:
            output[idx] = input[idx][2::]


class TestCVMOpWithLodTensor(OpTest):
    """
        Test cvm op with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cvm"
        self.use_cvm = True

        batch_size = 8
        dims = 11

        lod = [[1]]
        self.inputs = {
            'X': (np.random.uniform(0, 1, [1, dims]).astype("float32"), lod),
            'CVM': np.array([[0.6, 0.4]]).astype("float32"),
        }
        self.attrs = {'use_cvm': False}
        out = []
        for index, emb in enumerate(self.inputs["X"][0]):
            out.append(emb[2:])
        self.outputs = {'Y': (np.array(out), lod)}

    def test_check_output(self):
        self.check_output()


class TestCVMOpWithOutLodTensor(OpTest):
    """
        Test cvm op with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cvm"
        self.use_cvm = True

        batch_size = 8
        item_width = 11

        input = np.random.randn(batch_size, item_width).astype('float32')

        if self.use_cvm:
            output = np.ones([8, item_width], np.float32)
        else:
            output = np.ones([8, item_width - cvm_offset], np.float32)

        self.inputs = {
            'X': (np.random.uniform(0, 1, [1, dims]).astype("float32"), lod),
            'CVM': np.array([[0.6, 0.4]]).astype("float32"),
        }
        self.attrs = {'use_cvm': False}
        out = []
        for index, emb in enumerate(self.inputs["X"][0]):
            out.append(emb[2:])
        self.outputs = {'Y': (np.array(out), lod)}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
