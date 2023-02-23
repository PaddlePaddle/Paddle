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

import unittest

import numpy as np
from op_test import OpTest


def bcast(x, target_tensor):
    x_dims = x.shape
    y_dims = target_tensor.shape
    bcast_dims = []
    for i in range(len(x_dims)):
        bcast_dims.append(int(y_dims[i] / x_dims[i]))
    bcast_dims = np.array(bcast_dims).astype("int64")
    return bcast_dims


class TestExpandAsOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(100).astype("float64")
        target_tensor = np.random.rand(200).astype("float64")
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank2(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(10, 12).astype("float64")
        target_tensor = np.random.rand(20, 24).astype("float64")
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank3(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(2, 3, 20).astype("float64")
        target_tensor = np.random.rand(4, 6, 40).astype("float64")
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpandAsOpRank4(OpTest):
    def setUp(self):
        self.op_type = "expand_as"
        x = np.random.rand(1, 1, 7, 16).astype("float64")
        target_tensor = np.random.rand(4, 6, 14, 32).astype("float64")
        self.inputs = {'X': x, 'target_tensor': target_tensor}
        self.attrs = {}
        bcast_dims = bcast(x, target_tensor)
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
