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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestAccuracyOp(OpTest):
    def setUp(self):
        self.op_type = "accuracy"
        self.dtype = np.float32
        self.init_dtype()
        n = 8192
        infer = np.random.random((n, 1)).astype(self.dtype)
        indices = np.random.randint(0, 2, (n, 1)).astype('int64')
        label = np.random.randint(0, 2, (n, 1)).astype('int64')
        self.inputs = {'Out': infer, 'Indices': indices, "Label": label}
        num_correct = 0
        for rowid in range(n):
            for ele in indices[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {
            'Accuracy': np.array([num_correct / float(n)]).astype(self.dtype),
            'Correct': np.array([num_correct]).astype("int32"),
            'Total': np.array([n]).astype("int32")
        }

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestAccuracyOpFp16(TestAccuracyOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestAccuracyOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of accuracy_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            label = fluid.layers.data(
                name='label', shape=[-1, 1], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.accuracy, x1, label)
            # The input dtype of accuracy_op must be float32 or float64.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.accuracy, x2, label)
            x3 = fluid.layers.data(name='input', shape=[-1, 2], dtype="float16")
            fluid.layers.accuracy(input=x3, label=label)


if __name__ == '__main__':
    unittest.main()
