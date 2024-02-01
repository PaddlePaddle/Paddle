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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base
from paddle.base import framework

paddle.enable_static()


class XPUTestAssignValueOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'assign_value'
        self.use_dynamic_create_class = False

    class TestAssignValueOp(XPUOpTest):
        def init(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = 'assign_value'

        def setUp(self):
            self.init()
            self.inputs = {}
            self.attrs = {}
            self.init_data()
            self.attrs["shape"] = self.value.shape
            self.attrs["dtype"] = framework.convert_np_dtype_to_dtype_(
                self.value.dtype
            )
            self.outputs = {"Out": self.value}

        def init_data(self):
            self.value = np.random.random(size=(2, 5)).astype(np.float32)
            self.attrs["values"] = [float(v) for v in self.value.flat]

        def test_forward(self):
            self.check_output_with_place(self.place)

    class TestAssignValueOp2(TestAssignValueOp):
        def init_data(self):
            self.value = np.random.random(size=(2, 5)).astype(np.int32)
            self.attrs["values"] = [int(v) for v in self.value.flat]

    class TestAssignValueOp3(TestAssignValueOp):
        def init_data(self):
            self.value = np.random.random(size=(2, 5)).astype(np.int64)
            self.attrs["values"] = [int(v) for v in self.value.flat]

    class TestAssignValueOp4(TestAssignValueOp):
        def init_data(self):
            self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(
                np.bool_
            )
            self.attrs["values"] = [int(v) for v in self.value.flat]

    class TestAssignValueOp5(TestAssignValueOp):
        def init_data(self):
            self.value = np.random.random(size=(2, 5)).astype(np.float64)
            self.attrs["values"] = [float(v) for v in self.value.flat]

    class TestAssignValueOp6(TestAssignValueOp):
        def init_data(self):
            self.value = (
                np.random.random(size=(2, 5))
                + 1j * np.random.random(size=(2, 5))
            ).astype(np.complex64)
            self.attrs["values"] = list(self.value.flat)

    class TestAssignValueOp7(TestAssignValueOp):
        def init_data(self):
            self.value = (
                np.random.random(size=(2, 5))
                + 1j * np.random.random(size=(2, 5))
            ).astype(np.complex128)
            self.attrs["values"] = list(self.value.flat)


class TestAssignApi(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(
            self.dtype
        )
        self.place = base.XPUPlace(0)

    def init_dtype(self):
        self.dtype = "float32"

    def test_assign(self):
        main_program = base.Program()
        with base.program_guard(main_program):
            x = paddle.assign(self.value)

        exe = base.Executor(self.place)
        [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
        np.testing.assert_allclose(fetched_x, self.value)
        self.assertEqual(fetched_x.dtype, self.value.dtype)


class TestAssignApi2(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int32"


class TestAssignApi3(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int64"


class TestAssignApi4(TestAssignApi):
    def setUp(self):
        self.init_dtype()
        self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(
            np.bool_
        )
        self.place = base.XPUPlace(0)

    def init_dtype(self):
        self.dtype = "bool"


class TestAssignApi5(TestAssignApi):
    def init_dtype(self):
        self.dtype = "float64"


class TestAssignApi6(TestAssignApi):
    def setUp(self):
        self.init_dtype()
        self.value = (
            np.random.random(size=(2, 5)) + 1j * (np.random.random(size=(2, 5)))
        ).astype(np.complex64)
        self.place = base.XPUPlace(0)

    def init_dtype(self):
        self.dtype = "complex64"


class TestAssignApi7(TestAssignApi):
    def setUp(self):
        self.init_dtype()
        self.value = (
            np.random.random(size=(2, 5)) + 1j * (np.random.random(size=(2, 5)))
        ).astype(np.complex128)
        self.place = base.XPUPlace(0)

    def init_dtype(self):
        self.dtype = "complex128"


support_types = get_xpu_op_support_types('assign_value')
for stype in support_types:
    create_test_class(globals(), XPUTestAssignValueOp, stype)

if __name__ == '__main__':
    unittest.main()
