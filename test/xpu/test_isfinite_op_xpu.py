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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()
np.random.seed(10)


class XPUTestIsNANOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'isnan_v2'
        self.use_dynamic_create_class = False

    class TestIsNAN(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "isnan_v2"
            self.place = paddle.XPUPlace(0)
            self.set_inputs_output()

        def init_dtype(self):
            self.dtype = self.in_type

        def set_inputs_output(self):
            x = np.random.uniform(0.1, 1, [11, 17])
            x[0] = np.nan
            x[-1] = np.nan
            x[1] = np.inf
            x[-2] = -np.inf

            out = np.isnan(x).astype(bool)

            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
            else:
                x = x.astype(self.dtype)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)


support_types = get_xpu_op_support_types('isnan_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestIsNANOp, stype)


class XPUTestIsFiniteOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'isfinite_v2'
        self.use_dynamic_create_class = False

    class TestIsFinite(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "isfinite_v2"
            self.place = paddle.XPUPlace(0)
            self.set_inputs_output()

        def init_dtype(self):
            self.dtype = self.in_type

        def set_inputs_output(self):
            x = np.random.uniform(0.1, 1, [11, 17])
            x[0] = np.nan
            x[-1] = np.nan
            x[1] = np.inf
            x[-2] = -np.inf

            out = np.isfinite(x).astype(bool)

            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
            else:
                x = x.astype(self.dtype)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)


support_types = get_xpu_op_support_types('isfinite_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestIsFiniteOp, stype)


class XPUTestIsInfOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'isinf_v2'
        self.use_dynamic_create_class = False

    class TestIsInf(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "isinf_v2"
            self.place = paddle.XPUPlace(0)
            self.set_inputs_output()

        def init_dtype(self):
            self.dtype = self.in_type

        def set_inputs_output(self):
            x = np.random.uniform(0.1, 1, [11, 17])
            x[0] = np.nan
            x[-1] = np.nan
            x[1] = np.inf
            x[-2] = -np.inf

            out = np.isinf(x).astype(bool)

            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
            else:
                x = x.astype(self.dtype)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)


support_types = get_xpu_op_support_types('isinf_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestIsInfOp, stype)


if __name__ == '__main__':
    unittest.main()
