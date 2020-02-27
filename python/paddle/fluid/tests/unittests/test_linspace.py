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


class TestLinspaceOpCommonCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([0]).astype(dtype),
            'Stop': np.array([10]).astype(dtype),
            'Num': np.array([11]).astype('int32')
        }

        self.outputs = {'Out': np.arange(0, 11).astype(dtype)}

    def test_check_output(self):
        self.check_output()


class TestLinspaceOpReverseCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
            'Num': np.array([11]).astype('int32')
        }

        self.outputs = {'Out': np.arange(10, -1, -1).astype(dtype)}

    def test_check_output(self):
        self.check_output()


class TestLinspaceOpNumOneCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
            'Num': np.array([1]).astype('int32')
        }

        self.outputs = {'Out': np.array(10, dtype=dtype)}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
