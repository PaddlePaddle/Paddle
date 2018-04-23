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
from op_test import OpTest


class TestExpandOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random(12).astype("float32")}
        self.attrs = {'expand_times': [2]}
        output = np.tile(self.inputs['X'], 2)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2_Corner(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((12, 14)).astype("float32")}
        self.attrs = {'expand_times': [1, 1]}
        output = np.tile(self.inputs['X'], (1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank2(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((12, 14)).astype("float32")}
        self.attrs = {'expand_times': [2, 3]}
        output = np.tile(self.inputs['X'], (2, 3))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank3_Corner(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5)).astype("float32")}
        self.attrs = {'expand_times': [1, 1, 1]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank3(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5)).astype("float32")}
        self.attrs = {'expand_times': [2, 1, 4]}
        output = np.tile(self.inputs['X'], (2, 1, 4))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandOpRank4(OpTest):
    def setUp(self):
        self.op_type = "expand"
        self.inputs = {'X': np.random.random((2, 4, 5, 7)).astype("float32")}
        self.attrs = {'expand_times': [3, 2, 1, 2]}
        output = np.tile(self.inputs['X'], (3, 2, 1, 2))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
