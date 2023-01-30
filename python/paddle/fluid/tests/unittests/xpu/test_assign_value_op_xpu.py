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

<<<<<<< HEAD
import sys
import unittest

import numpy as np

sys.path.append("..")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers
=======
from __future__ import print_function

import unittest
import numpy as np
import sys

sys.path.append("..")
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
import paddle
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class XPUTestAssignValueOp(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'assign_value'
        self.use_dynamic_create_class = False

    class TestAssignValueOp(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                self.value.dtype
            )
=======
                self.value.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.outputs = {"Out": self.value}

        def init_data(self):
            self.value = np.random.random(size=(2, 5)).astype(self.dtype)
            self.attrs["fp32_values"] = [float(v) for v in self.value.flat]

        def test_forward(self):
            self.check_output_with_place(self.place)

    class TestAssignValueOp2(TestAssignValueOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_data(self):
            self.value = np.random.random(size=(2, 5)).astype(np.int32)
            self.attrs["int32_values"] = [int(v) for v in self.value.flat]

    class TestAssignValueOp3(TestAssignValueOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_data(self):
            self.value = np.random.random(size=(2, 5)).astype(np.int64)
            self.attrs["int64_values"] = [int(v) for v in self.value.flat]

    class TestAssignValueOp4(TestAssignValueOp):
<<<<<<< HEAD
        def init_data(self):
            self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(
                np.bool_
            )
=======

        def init_data(self):
            self.value = np.random.choice(a=[False, True],
                                          size=(2, 5)).astype(np.bool)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.attrs["bool_values"] = [int(v) for v in self.value.flat]


class TestAssignApi(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.init_dtype()
        self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(
            self.dtype
        )
=======

    def setUp(self):
        self.init_dtype()
        self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(
            self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.place = fluid.XPUPlace(0)

    def init_dtype(self):
        self.dtype = "float32"

    def test_assign(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
<<<<<<< HEAD
            x = paddle.tensor.create_tensor(dtype=self.dtype)
=======
            x = layers.create_tensor(dtype=self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            layers.assign(input=self.value, output=x)

        exe = fluid.Executor(self.place)
        [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
        np.testing.assert_allclose(fetched_x, self.value)
        self.assertEqual(fetched_x.dtype, self.value.dtype)


class TestAssignApi2(TestAssignApi):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = "int32"


class TestAssignApi3(TestAssignApi):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = "int64"


class TestAssignApi4(TestAssignApi):
<<<<<<< HEAD
    def setUp(self):
        self.init_dtype()
        self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(
            np.bool_
        )
=======

    def setUp(self):
        self.init_dtype()
        self.value = np.random.choice(a=[False, True],
                                      size=(2, 5)).astype(np.bool)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.place = fluid.XPUPlace(0)

    def init_dtype(self):
        self.dtype = "bool"


support_types = get_xpu_op_support_types('assign_value')
for stype in support_types:
    create_test_class(globals(), XPUTestAssignValueOp, stype)

if __name__ == '__main__':
    unittest.main()
