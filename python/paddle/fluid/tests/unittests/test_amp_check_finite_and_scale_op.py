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
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid


@skip_check_grad_ci(
    reason="Operator amp_check_finite_and_scale is used for dygraph auto-mixed_precision training only. It has no grad op."
)
class TestAmpCheckFiniteAndScaleOp(OpTest):
    def setUp(self):
        self.op_type = "amp_check_finite_and_scale"
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([0]),
            'Out': [('out0', x * scale)],
        }
        self.place = fluid.CUDAPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(place=self.place)


@skip_check_grad_ci(
    reason="Operator amp_check_finite_and_scale is used for dygraph auto-mixed_precision training only. It has no grad op."
)
class TestAmpCheckFiniteAndScaleOpWithNan(OpTest):
    def setUp(self):
        self.op_type = "amp_check_finite_and_scale"
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.nan
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }
        self.place = fluid.CUDAPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(place=self.place, no_check_set=['Out'])


@skip_check_grad_ci(
    reason="Operator amp_check_finite_and_scale is used for dygraph auto-mixed_precision training only. It has no grad op."
)
class TestAmpCheckFiniteAndScaleOpWithInf(OpTest):
    def setUp(self):
        self.op_type = "amp_check_finite_and_scale"
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.inf
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }
        self.place = fluid.CUDAPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(place=self.place, no_check_set=['Out'])


if __name__ == '__main__':
    unittest.main()
