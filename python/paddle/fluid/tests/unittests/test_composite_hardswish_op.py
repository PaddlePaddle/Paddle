# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from test_activation_op import TestHardSwish as BaseTestHardSwish
from test_activation_op import ref_hardswish

import paddle

paddle.enable_static()


class TestHardSwish(BaseTestHardSwish):
    def setUp(self):
        self.op_type = 'hard_swish'
        self.prim_op_type = "comp"
        self.init_dtype()
        self.init_shape()
        self.init_error()
        self.init_comp_type()
        self.python_api = paddle.nn.functional.hardswish

        np.random.seed(1024)
        x = np.random.uniform(-6, 6, self.shape).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        # the same with TestAbs
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02
        out = ref_hardswish(x, threshold, scale, offset)

        self.inputs = {'X': x}
        self.attrs = {'threshold': threshold, 'scale': scale, 'offset': offset}
        self.outputs = {'Out': out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_eager=True,
            check_prim=True,
            max_relative_error=self.max_relative_error,
        )

    def test_check_output(self):
        self.check_output(check_eager=True, check_prim=True)

    def init_error(self):
        self.max_relative_error = 0.005

    def init_comp_type(self):
        pass


class TestHardSwish_ZeroDim(TestHardSwish):
    def init_shape(self):
        self.shape = []


class TesthardSwishFP32(TestHardSwish):
    def init_dtype(self):
        self.dtype = np.float32

    def init_error(self):
        super().init_error()
        self.max_relative_error = 0.01


class TestHardSwish_ZeroDim_FP32(TesthardSwishFP32):
    def init_shape(self):
        self.shape = []


class TesthardSwishFP16(TesthardSwishFP32):
    def init_dtype(self):
        self.dtype = np.float16


class TestHardSwish_ZeroDim_FP16(TesthardSwishFP16):
    def init_shape(self):
        self.shape = []


if __name__ == "__main__":
    unittest.main()
