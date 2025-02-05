# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestCheckFiniteAndUnscaleOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'check_finite_and_unscale'
        self.use_dynamic_create_class = False

    class TestCheckFiniteAndUnscaleOpNormal(XPUOpTest):
        def setUp(self):
            self.op_type = "check_finite_and_unscale"
            self.init_dtype()
            x = np.random.random((8, 8)).astype(self.dtype)
            scale = np.random.random(1).astype(np.float32)
            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {
                'FoundInfinite': np.array([0]),
                'Out': [('out0', x / scale)],
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestCheckFiniteAndUnscaleOpWithNan(XPUOpTest):
        def setUp(self):
            self.op_type = "check_finite_and_unscale"
            self.init_dtype()
            x = np.random.random((256, 256))
            idx1 = np.random.randint(255)
            idx2 = np.random.randint(255)
            x[idx1][idx2] = np.nan
            x[idx2][idx1] = np.nan
            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
            else:
                x = x.astype(self.dtype)
            scale = np.random.random(1).astype(np.float32)

            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {
                'FoundInfinite': np.array([1]),
                'Out': [('out0', x)],
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            # When input contains nan, do not check the output,
            # since the output may be nondeterministic and will be discarded.
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['Out'])

    class TestCheckFiniteAndUnscaleOpWithInf(XPUOpTest):
        def setUp(self):
            self.op_type = "check_finite_and_unscale"
            self.init_dtype()
            x = np.random.random((256, 256))
            idx1 = np.random.randint(255)
            idx2 = np.random.randint(255)
            x[idx1][idx2] = np.nan
            x[idx2][idx1] = np.nan
            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
            else:
                x = x.astype(self.dtype)
            scale = np.random.random(1).astype(np.float32)
            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {
                'FoundInfinite': np.array([1]),
                'Out': [('out0', x)],
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            # When input contains inf, do not check the output,
            # since the output may be nondeterministic and will be discarded.
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['Out'])

    class TestCheckFiniteAndUnscaleOpWithInfAndNan(XPUOpTest):
        def setUp(self):
            self.op_type = "check_finite_and_unscale"
            self.init_dtype()
            x = np.random.random((256, 256))
            idx1 = np.random.randint(255)
            idx2 = np.random.randint(255)
            x[idx1][idx2] = np.inf
            x[idx2][idx1] = np.nan
            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
            else:
                x = x.astype(self.dtype)
            scale = np.random.random(1).astype(np.float32)
            self.inputs = {'X': [('x0', x)], 'Scale': scale}
            self.outputs = {
                'FoundInfinite': np.array([1]),
                'Out': [('out0', x)],
            }

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            # When input contains inf, do not check the output,
            # since the output may be nondeterministic and will be discarded.
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['Out'])


support_types = get_xpu_op_support_types('check_finite_and_unscale')
for stype in support_types:
    create_test_class(globals(), XPUTestCheckFiniteAndUnscaleOp, stype)

if __name__ == '__main__':
    unittest.main()
