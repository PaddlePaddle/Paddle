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
from op_test import OpTest
import random
import six


def np_partial_concat(inputs, start, length):
    assert (len(inputs[0].shape) == 2)
    size = inputs[0].shape[1]
    assert (start >= -size and start < size)

    if start < 0:
        start += size
    if length < 0:
        length = size - start
    assert (size >= start + length)

    elems = []
    for elem in inputs:
        assert (elem.shape == inputs[0].shape)
        elems.append(elem[:, start:start + length])
    res = np.concatenate(elems, axis=1)
    return np.concatenate(elems, axis=1)


class TestPartialConcatOp(OpTest):

    def setUp(self):
        self.op_type = "partial_concat"
        self.init_kernel_type()
        self.init_para()
        self.var_names = [
            'x' + str(num) for num in six.moves.range(self.var_num)
        ]
        self.vars = [np.random.random((self.batch_size, self.column)).astype(self.dtype)\
                     for num in six.moves.range(self.var_num) ]
        self.inputs = {'X': list(zip(self.var_names, self.vars))}
        self.attrs = {'start_index': self.start_index, 'length': self.length}
        y = np_partial_concat(self.vars[:], self.start_index, self.length)
        self.outputs = {'Out': y}

    def init_kernel_type(self):
        self.dtype = np.float64

    def init_para(self):
        self.batch_size = random.randint(10, 20)
        self.column = random.randint(101, 200)
        self.start_index = random.randint(0, self.column - 1)
        self.length = -1
        self.var_num = random.randint(1, 3)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        for var_name in self.var_names:
            self.check_grad([var_name], 'Out')


class TestPartialConcatOp2(TestPartialConcatOp):

    def init_para(self):
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = -5
        self.length = -1
        self.var_num = 3


class TestPartialConcatOp3(TestPartialConcatOp):

    def init_para(self):
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = 10
        self.length = 20
        self.var_num = 2


class TestPartialConcatOp4(TestPartialConcatOp):

    def init_para(self):
        self.batch_size = random.randint(1, 10)
        self.column = random.randint(101, 200)
        self.start_index = -1
        self.length = -1
        self.var_num = 1


if __name__ == '__main__':
    unittest.main()
