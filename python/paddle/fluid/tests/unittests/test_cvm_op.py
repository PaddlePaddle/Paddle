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


def cvm_compute(X, item_width, use_cvm):
    cvm_offset = 0 if use_cvm else 2
    batch_size = X.shape[0]

    Y = np.ones([batch_size, item_width - cvm_offset], np.float32)

    for idx in range(batch_size):
        if use_cvm:
            Y[idx] = X[idx]
            Y[idx][0] = log(Y[idx][0] + 1)
            Y[idx][1] = log(Y[idx][1] + 1) - Y[idx][0]
        else:
            Y[idx] = X[idx][2:]

    return Y


def cvm_grad_compute(DY, CVM, item_width, use_cvm):
    batch_size = DY.shape[0]
    DX = np.ones([batch_size, item_width], np.float32)

    for idx in range(batch_size):
        DX[idx][0] = CVM[idx][0]
        DX[idx][1] = CVM[idx][1]

        if use_cvm:
            DX[idx][2:] = DY[idx][2:]
        else:
            DX[idx][2:] = DY[idx]
    return DX


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


class TestCVMOpWithOutLodTensor1(OpTest):
    """
    Test cvm op with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cvm"
        self.use_cvm = True

        batch_size = 2
        item_width = 11

        input = np.random.uniform(0, 1,
                                  (batch_size, item_width)).astype('float32')
        output = cvm_compute(input, item_width, self.use_cvm)
        cvm = np.array([[0.6, 0.4]]).astype("float32")

        self.inputs = {'X': input, 'CVM': cvm}
        self.attrs = {'use_cvm': self.use_cvm}
        self.outputs = {'Y': output}

    def test_check_output(self):
        self.check_output()


class TestCVMOpWithOutLodTensor2(OpTest):
    """
    Test cvm op with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cvm"
        self.use_cvm = False

        batch_size = 2
        item_width = 11

        input = np.random.uniform(0, 1,
                                  (batch_size, item_width)).astype('float32')
        output = cvm_compute(input, item_width, self.use_cvm)
        cvm = np.array([[0.6, 0.4]]).astype("float32")

        self.inputs = {'X': input, 'CVM': cvm}
        self.attrs = {'use_cvm': self.use_cvm}
        self.outputs = {'Y': output}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
