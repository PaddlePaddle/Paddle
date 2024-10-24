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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


class TestLabelSmoothOp(OpTest):
    def config(self):
        self.op_type = "label_smooth"
        self.python_api = paddle.nn.functional.label_smooth
        self.init_dtype()
        self.epsilon = 0.1
        batch_size, self.label_dim = 10, 12
        self.label = np.zeros((batch_size, self.label_dim)).astype(self.dtype)
        nonzero_index = np.random.randint(self.label_dim, size=(batch_size))
        self.label[np.arange(batch_size), nonzero_index] = 1

    def setUp(self):
        self.config()
        smoothed_label = (
            1 - self.epsilon
        ) * self.label + self.epsilon / self.label_dim
        self.inputs = {'X': self.label}
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {'Out': smoothed_label}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.supports_bfloat16(),
    "core is not compiled with CUDA or place do not support bfloat16",
)
class TestLabelSmoothOpBF16(OpTest):
    def config(self):
        self.op_type = "label_smooth"
        self.python_api = paddle.nn.functional.label_smooth
        self.epsilon = 0.1
        self.dtype = np.uint16
        batch_size, self.label_dim = 10, 12
        self.label = np.zeros((batch_size, self.label_dim)).astype(np.float32)
        nonzero_index = np.random.randint(self.label_dim, size=(batch_size))
        self.label[np.arange(batch_size), nonzero_index] = 1

    def setUp(self):
        self.config()
        smoothed_label = (
            1 - self.epsilon
        ) * self.label + self.epsilon / self.label_dim
        self.inputs = {'X': convert_float_to_uint16(self.label)}
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {'Out': convert_float_to_uint16(smoothed_label)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ["X"], "Out", check_pir=True)


class TestLabelSmoothFP16OP(TestLabelSmoothOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOpWithPriorDist(TestLabelSmoothOp):
    def setUp(self):
        self.config()
        dist = np.random.random((1, self.label_dim)).astype(self.dtype)
        smoothed_label = (1 - self.epsilon) * self.label + self.epsilon * dist
        self.inputs = {'X': self.label, 'PriorDist': dist}
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {'Out': smoothed_label}


class TestLabelSmoothFP16OPWithPriorDist(TestLabelSmoothOpWithPriorDist):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothBF16OPWithPriorDist(TestLabelSmoothOpBF16):
    def setUp(self):
        self.config()
        dist = np.random.random((1, self.label_dim)).astype(np.float32)
        smoothed_label = (1 - self.epsilon) * self.label + self.epsilon * dist
        self.inputs = {
            'X': convert_float_to_uint16(self.label),
            'PriorDist': convert_float_to_uint16(dist),
        }
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {'Out': convert_float_to_uint16(smoothed_label)}


class TestLabelSmoothOp3D(TestLabelSmoothOp):
    def setUp(self):
        super().setUp()
        self.inputs['X'] = self.inputs['X'].reshape(
            [2, -1, self.inputs['X'].shape[-1]]
        )
        self.outputs['Out'] = self.outputs['Out'].reshape(
            self.inputs['X'].shape
        )


class TestLabelSmoothOp3DBF16(TestLabelSmoothOpBF16):
    def setUp(self):
        super().setUp()
        self.inputs['X'] = self.inputs['X'].reshape(
            [2, -1, self.inputs['X'].shape[-1]]
        )
        self.outputs['Out'] = self.outputs['Out'].reshape(
            self.inputs['X'].shape
        )


class TestLabelSmoothFP16OP3D(TestLabelSmoothOp3D):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOpWithPriorDist3D(TestLabelSmoothOpWithPriorDist):
    def setUp(self):
        super().setUp()
        self.inputs['X'] = self.inputs['X'].reshape(
            [2, -1, self.inputs['X'].shape[-1]]
        )
        self.outputs['Out'] = self.outputs['Out'].reshape(
            self.inputs['X'].shape
        )


class TestLabelSmoothFP16OPWithPriorDist3D(TestLabelSmoothOpWithPriorDist3D):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothBF16OpWithPriorDist3D(TestLabelSmoothBF16OPWithPriorDist):
    def setUp(self):
        super().setUp()
        self.inputs['X'] = self.inputs['X'].reshape(
            [2, -1, self.inputs['X'].shape[-1]]
        )
        self.outputs['Out'] = self.outputs['Out'].reshape(
            self.inputs['X'].shape
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
