#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2022


class TestCheckFiniteAndUnscaleOp(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "check_finite_and_unscale"
        self.init_dtype()
        self.init_test_case()

    def init_test_case(self):
        x = np.random.random((129, 129)).astype(self.dtype)
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([0]),
            'Out': [('out0', x / scale)],
        }

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestCheckFiniteAndUnscaleOpWithNan(TestCheckFiniteAndUnscaleOp):

    def init_test_case(self):
        x = np.random.random((129, 129)).astype(self.dtype)
        x[128][128] = np.nan
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }

    def test_check_output(self):
        # When input contains nan, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(self.place, no_check_set=['Out'])


class TestCheckFiniteAndUnscaleOpWithInf(TestCheckFiniteAndUnscaleOp):

    def init_test_case(self):
        x = np.random.random((129, 129)).astype(self.dtype)
        x[128][128] = np.inf
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }

    def test_check_output(self):
        # When input contains inf, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(self.place, no_check_set=['Out'])


class TestCheckFiniteAndUnscaleOpMultiInput(TestCheckFiniteAndUnscaleOp):

    def init_test_case(self):
        x0 = np.random.random((129, 129)).astype(self.dtype)
        x1 = np.random.random((129, 129)).astype(self.dtype)
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x0), ('x1', x1)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([0]),
            'Out': [('out0', x0 / scale), ('out1', x1 / scale)],
        }


class TestCheckFiniteAndUnscaleOpMultiInputWithNan(TestCheckFiniteAndUnscaleOp):

    def init_test_case(self):
        x0 = np.random.random((129, 129)).astype(self.dtype)
        x0[128][128] = np.nan
        x1 = np.random.random((129, 129)).astype(self.dtype)
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x0), ('x1', x1)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x0 / scale), ('out1', x1 / scale)],
        }

    def test_check_output(self):
        # When input contains inf, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(self.place, no_check_set=['Out'])


class TestCheckFiniteAndUnscaleOpMultiInputWithInf(TestCheckFiniteAndUnscaleOp):

    def init_test_case(self):
        x0 = np.random.random((129, 129)).astype(self.dtype)
        x0[128][128] = np.nan
        x1 = np.random.random((129, 129)).astype(self.dtype)
        x1[128][128] = np.inf
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x0), ('x1', x1)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x0 / scale), ('out1', x1 / scale)],
        }

    def test_check_output(self):
        # When input contains inf, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(self.place, no_check_set=['Out'])


if __name__ == '__main__':
    unittest.main()
