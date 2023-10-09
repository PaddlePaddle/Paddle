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
from op_test import OpTest

from paddle.static.amp import amp_nn


def check_finite_and_unscale_wrapper(x, scale):
    _, found_inf = amp_nn.check_finite_and_unscale([x], scale)
    return x, found_inf


class TestCheckFiniteAndUnscaleOp(OpTest):
    def setUp(self):
        self.op_type = "check_finite_and_unscale"
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ["out0", "FoundInfinite"]
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        scale = np.random.random(1).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([0]),
            'Out': [('out0', x / scale)],
        }

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestCheckFiniteAndUnscaleOpWithNan(OpTest):
    def setUp(self):
        self.op_type = "check_finite_and_unscale"
        self.init_dtype()
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ["out0", "FoundInfinite"]
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.nan
        scale = np.random.random(1).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        # When input contains nan, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output(no_check_set=['Out'])


class TestCheckFiniteAndUnscaleOpWithInf(OpTest):
    def setUp(self):
        self.op_type = "check_finite_and_unscale"
        self.init_dtype()
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ["out0", "FoundInfinite"]
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.inf
        scale = np.random.random(1).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        # When input contains inf, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output(no_check_set=['Out'])


if __name__ == '__main__':
    unittest.main()
