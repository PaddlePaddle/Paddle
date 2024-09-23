#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestScaleOp(OpTest):
    def setUp(self):
        self.init_shape()
        self.op_type = "scale"
        self.inputs = {'X': np.random.random(self.shape).astype(np.float32)}
        self.attrs = {'scale': -2.3, 'use_mkldnn': True, 'bias': 0.2}
        self.use_mkldnn = True
        self.outputs = {
            'Out': (self.inputs['X'] * self.attrs['scale']) + self.attrs['bias']
        }

    def init_shape(self):
        self.shape = [10, 10]

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=False
        )


class TestScaleOp_ZeroDim(TestScaleOp):
    def init_shape(self):
        self.shape = []


class TestScaleOpBiasNotAfterScale(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.inputs = {'X': np.random.random((10, 10)).astype(np.float32)}
        self.attrs = {
            'scale': 1.5,
            'use_mkldnn': True,
            'bias': 2.3,
            'bias_after_scale': False,
        }
        self.use_mkldnn = True
        self.outputs = {
            'Out': (self.inputs['X'] + self.attrs['bias']) * self.attrs['scale']
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=False
        )


class TestScaleOpScaleTensor(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.scale = -2.3
        self.inputs = {
            'X': np.random.random((10, 10)).astype(np.float32),
            'ScaleTensor': np.array([self.scale]).astype(np.float32),
        }
        self.attrs = {}
        self.outputs = {'Out': self.inputs['X'] * self.scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=False
        )


class TestScaleOpScaleTensorNotBiasAfterScale(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.scale = -1.2
        self.inputs = {
            'X': np.random.random((10, 10)).astype(np.float32),
            'ScaleTensor': np.array([self.scale]).astype(np.float32),
        }
        self.attrs = {'bias': -6.8, 'bias_after_scale': False}
        self.outputs = {
            'Out': (self.inputs['X'] + self.attrs['bias'])
            * self.inputs['ScaleTensor']
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=False
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
