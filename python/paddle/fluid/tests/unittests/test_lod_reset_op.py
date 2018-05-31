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


class TestLodResetOpByAttr(OpTest):
    def setUp(self):
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[0, 3, 5, 10]]
        target_lod_0 = [0, 7, 10]
        self.inputs = {'X': (x, lod)}
        self.attrs = {'target_lod': target_lod_0}
        self.outputs = {'Out': (x, [target_lod_0])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestLodResetOpByInput(OpTest):
    def setUp(self):
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[0, 3, 5, 10]]
        target_lod_0 = [0, 4, 7, 10]
        self.inputs = {
            'X': (x, lod),
            'Y': np.array([target_lod_0]).astype('int32')
        }
        self.outputs = {'Out': (x, [target_lod_0])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("Y"))


class TestLodResetOpBoth(OpTest):
    def setUp(self):
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[0, 3, 5, 10]]
        target_lod_0_attr = [0, 7, 10]
        target_lod_0_in = [0, 4, 7, 10]
        self.inputs = {
            'X': (x, lod),
            'Y': np.array(target_lod_0_in).astype('int32')
        }
        self.attrs = {'target_lod': target_lod_0_attr}
        self.outputs = {'Out': (x, [target_lod_0_in])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("Y"))


class TestLodResetOpYIsLoDTensor(OpTest):
    def setUp(self):
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[0, 3, 5, 10]]
        y = np.random.random((10, 10)).astype("float32")
        target_lod_0 = [[0, 4, 7, 10]]
        self.inputs = {'X': (x, lod), 'Y': (y, target_lod_0)}
        self.outputs = {'Out': (x, target_lod_0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("Y"))


if __name__ == '__main__':
    unittest.main()
