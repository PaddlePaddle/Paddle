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


class TestSumOp1(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=2)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp2(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': -1, 'reverse': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {
            'Out': np.flip(
                np.flip(
                    self.inputs['X'], axis=2).cumsum(axis=2), axis=2)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp3(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 1}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp4(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 0}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp5(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.inputs = {'X': np.random.random((5, 20)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp7(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.inputs = {'X': np.random.random((100)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp8(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((5, 6, 4)).astype("float64")
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (5, 6, 1), dtype=np.float64), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
