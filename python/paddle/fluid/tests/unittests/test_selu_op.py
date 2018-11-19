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
import paddle.fluid.core as core
from op_test import OpTest

import time

class TestSeluOpFloat64(OpTest):
    def setUp(self):
        self.op_type = "selu"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'X': np.concatenate([(np.random.random(size=(50)) + 0.005).astype(self.dtype),
                                (-np.random.random(size=(50)) - 0.005).astype(self.dtype)])}
        scale = np.asarray([1.0507009873554804934193349852946], dtype=np.float64)
        alpha = np.asarray([1.6732632423543772848170429916717], dtype=np.float64)
        self.outputs = {'Out': (self.inputs['X'] >= 0) * scale * self.inputs['X'] + \
                        (self.inputs['X'] < 0) * scale * alpha * (np.exp(self.inputs['X']) - 1.0)}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        # atol means the absolute tolerance parameter
        # double (float64) inputs should only tolerate very small diff (1e-10)
        self.check_output(atol=1e-10)

    def test_check_grad_normal(self):
        # delta should not be too big, for the precision of numeric gradient
        # double (float64) inputs should only tolerate very small relative error (1e-10)
        self.check_grad(['X'], 'Out',
                        max_relative_error=1e-10,
                        numeric_grad_delta=5e-6)


class TestSeluOpFloat32(OpTest):
    def setUp(self):
        self.op_type = "selu"
        self.dtype = np.float32
        self.init_dtype_type()
        self.inputs = {
            'X': np.concatenate([(np.random.random(size=(50)) + 0.005).astype(self.dtype),
                                (-np.random.random(size=(50)) - 0.005).astype(self.dtype)])}
        scale = np.asarray([1.0507009873554804934193349852946], dtype=np.float64)
        alpha = np.asarray([1.6732632423543772848170429916717], dtype=np.float64)
        self.outputs = {'Out': (self.inputs['X'] >= 0) * scale * self.inputs['X'] + \
                        (self.inputs['X'] < 0) * scale * alpha * (np.exp(self.inputs['X']) - 1.0)}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(atol=1e-5)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out',
                        max_relative_error=1e-3,
                        numeric_grad_delta=5e-3)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSeluOpFloat64(OpTest):
    def setUp(self):
        self.op_type = "selu"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'X': np.concatenate([(np.random.random(size=(50)) + 0.005).astype(self.dtype),
                                (-np.random.random(size=(50)) - 0.005).astype(self.dtype)])}
        scale = np.asarray([1.0507009873554804934193349852946], dtype=np.float64)
        alpha = np.asarray([1.6732632423543772848170429916717], dtype=np.float64)
        self.outputs = {'Out': (self.inputs['X'] >= 0) * scale * self.inputs['X'] + \
                        (self.inputs['X'] < 0) * scale * alpha * (np.exp(self.inputs['X']) - 1.0)}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        # atol means the absolute tolerance parameter
        # double (float64) inputs should only tolerate very small diff (1e-10)
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-10)

    def test_check_grad_normal(self):
        # delta should not be too big, for the precision of numeric gradient
        # double (float64) inputs should only tolerate very small relative error (1e-10)
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
                place, ['X'],
                'Out',
                max_relative_error=1e-9,
                numeric_grad_delta=5e-6)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSeluOpCUDAFloat32(OpTest):
    def setUp(self):
        self.op_type = "selu"
        self.dtype = np.float32
        self.init_dtype_type()
        self.inputs = {
            'X': np.concatenate([(np.random.random(size=(50)) + 0.005).astype(self.dtype),
                                (-np.random.random(size=(50)) - 0.005).astype(self.dtype)])}
        scale = np.asarray([1.0507009873554804934193349852946], dtype=np.float64)
        alpha = np.asarray([1.6732632423543772848170429916717], dtype=np.float64)
        self.outputs = {'Out': (self.inputs['X'] >= 0) * scale * self.inputs['X'] + \
                        (self.inputs['X'] < 0) * scale * alpha * (np.exp(self.inputs['X']) - 1.0)}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-5)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
                place, ['X'],
                'Out',
                max_relative_error=1e-3,
                numeric_grad_delta=5e-3)

if __name__ == "__main__":
    unittest.main()
