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

import sys

sys.path.append("..")
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.op_test import (OpTest,
                                                  convert_float_to_uint16,
                                                  convert_uint16_to_float)
from paddle import _C_ops, _legacy_C_ops
import op_test
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestSumOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'sum'
        self.use_dynamic_create_class = False

    class TestSumOp(XPUOpTest):

        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "sum"
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            x0 = np.random.random(self.shape).astype(self.dtype)
            x1 = np.random.random(self.shape).astype(self.dtype)
            x2 = np.random.random(self.shape).astype(self.dtype)
            self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
            y = x0 + x1 + x2
            self.outputs = {'Out': y}

        def init_dtype(self):
            self.dtype = self.in_type

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.dtype

        def set_shape(self):
            self.shape = (3, 10)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['x0'], 'Out')

    class TestSumOp1(TestSumOp):

        def set_shape(self):
            self.shape = (5)

    class TestSumOp2(TestSumOp):

        def set_shape(self):
            self.shape = (1, 1, 1, 1, 1)

    class TestSumOp3(TestSumOp):

        def set_shape(self):
            self.shape = (10, 5, 7)

    class TestSumOp4(TestSumOp):

        def set_shape(self):
            self.shape = (2, 2, 3, 3)


def create_test_sum_fp16_class(parent):

    class TestSumFp16Case(parent):

        def init_kernel_type(self):
            self.dtype = np.float16

        def test_w_is_selected_rows(self):
            place = core.XPUPlace(0)
            # if core.is_float16_supported(place):
            for inplace in [True, False]:
                self.check_with_place(place, inplace)

    cls_name = "{0}_{1}".format(parent.__name__, "SumFp16Test")
    TestSumFp16Case.__name__ = cls_name
    globals()[cls_name] = TestSumFp16Case


class API_Test_Add_n(unittest.TestCase):

    def test_api(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input0 = fluid.layers.fill_constant(shape=[2, 3],
                                                dtype='int64',
                                                value=5)
            input1 = fluid.layers.fill_constant(shape=[2, 3],
                                                dtype='int64',
                                                value=3)
            expected_result = np.empty((2, 3))
            expected_result.fill(8)
            sum_value = paddle.add_n([input0, input1])
            exe = fluid.Executor(fluid.XPUPlace(0))
            result = exe.run(fetch_list=[sum_value])

            self.assertEqual((result == expected_result).all(), True)

        with fluid.dygraph.guard():
            input0 = paddle.ones(shape=[2, 3], dtype='float32')
            expected_result = np.empty((2, 3))
            expected_result.fill(2)
            sum_value = paddle.add_n([input0, input0])

            self.assertEqual((sum_value.numpy() == expected_result).all(), True)


class TestRaiseSumError(unittest.TestCase):

    def test_errors(self):

        def test_type():
            fluid.layers.sum([11, 22])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            data2 = fluid.data(name="input2", shape=[10], dtype="int8")
            fluid.layers.sum([data1, data2])

        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            fluid.layers.sum(data1)

        self.assertRaises(TypeError, test_dtype1)


class TestRaiseSumsError(unittest.TestCase):

    def test_errors(self):

        def test_type():
            fluid.layers.sums([11, 22])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            data2 = fluid.data(name="input2", shape=[10], dtype="int8")
            fluid.layers.sums([data1, data2])

        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            fluid.layers.sums(data1)

        self.assertRaises(TypeError, test_dtype1)

        def test_out_type():
            data1 = fluid.data(name="input1", shape=[10], dtype="flaot32")
            data2 = fluid.data(name="input2", shape=[10], dtype="float32")
            fluid.layers.sums([data1, data2], out=[10])

        self.assertRaises(TypeError, test_out_type)

        def test_out_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="flaot32")
            data2 = fluid.data(name="input2", shape=[10], dtype="float32")
            out = fluid.data(name="out", shape=[10], dtype="int8")
            fluid.layers.sums([data1, data2], out=out)

        self.assertRaises(TypeError, test_out_dtype)


class TestSumOpError(unittest.TestCase):

    def test_errors(self):

        def test_empty_list_input():
            with fluid.dygraph.guard():
                fluid._legacy_C_ops.sum([])

        def test_list_of_none_input():
            with fluid.dygraph.guard():
                fluid._legacy_C_ops.sum([None])

        self.assertRaises(Exception, test_empty_list_input)
        self.assertRaises(Exception, test_list_of_none_input)


support_types = get_xpu_op_support_types('sum')
for stype in support_types:
    create_test_class(globals(), XPUTestSumOp, stype)

if __name__ == "__main__":
    unittest.main()
