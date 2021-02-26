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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("..")
from op_test_xpu import OpTest, XPUOpTest
from op_test import skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_


class TestXPUReduceMaxOp(XPUOpTest):
    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.use_xpu = True
        self.use_mkldnn = False
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keep_dim,
            'reduce_all': self.reduce_all
        }
        self.inputs = {'X': np.random.random(self.shape).astype("float32")}
        if self.attrs['reduce_all']:
            self.outputs = {'Out': self.inputs['X'].max()}
        else:
            self.outputs = {
                'Out': self.inputs['X'].max(axis=self.axis,
                                            keepdims=self.attrs['keep_dim'])
            }

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')

    def init_op_type(self):
        self.op_type = "reduce_max"
        self.use_mkldnn = False
        self.keep_dim = False
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (5, 6, 10)
        self.axis = (-1, )


if __name__ == '__main__':
    unittest.main()
