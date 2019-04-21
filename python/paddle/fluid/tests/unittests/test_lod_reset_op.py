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


class TestLodResetOpByAttr(OpTest):
    def setUp(self):
        self.test_gc = True
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[3, 2, 5]]
        # target_offset_lod and target_lod are the same lod info represented
        # in offset-based format and length-based format, respectively.
        target_offset_lod = [0, 7, 10]
        target_lod = [7, 3]
        self.inputs = {'X': (x, lod)}
        # The `target_lod` attribute is still based on offset
        self.attrs = {'target_lod': target_offset_lod}
        self.outputs = {'Out': (x, [target_lod])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestLodResetOpByInput(OpTest):
    def setUp(self):
        self.test_gc = True
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[3, 2, 5]]
        # target_offset_lod and target_lod are the same lod info represented
        # in offset-based format and length-based format, respectively.
        target_offset_lod = [0, 4, 7, 10]
        target_lod = [4, 3, 3]
        self.inputs = {
            'X': (x, lod),
            'Y': np.array([target_offset_lod]).astype('int32')
        }
        self.outputs = {'Out': (x, [target_lod])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("Y"))


class TestLodResetOpBoth(OpTest):
    def setUp(self):
        self.test_gc = True
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[3, 2, 5]]
        target_offset_lod_attr = [0, 7, 10]
        target_offset_lod_in = [0, 4, 7, 10]
        target_lod_in = [4, 3, 3]
        self.inputs = {
            'X': (x, lod),
            'Y': np.array(target_offset_lod_in).astype('int32')
        }
        self.attrs = {'target_lod': target_offset_lod_attr}
        self.outputs = {'Out': (x, [target_lod_in])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("Y"))


class TestLodResetOpYIsLoDTensor(OpTest):
    def setUp(self):
        self.test_gc = True
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[3, 2, 5]]
        y = np.random.random((10, 10)).astype("float32")
        target_lod = [[4, 3, 3]]
        self.inputs = {'X': (x, lod), 'Y': (y, target_lod)}
        self.outputs = {'Out': (x, target_lod)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("Y"))


if __name__ == '__main__':
    unittest.main()
