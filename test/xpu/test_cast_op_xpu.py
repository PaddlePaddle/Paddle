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
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16, convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle
from paddle import base
from paddle.base import Program, core, program_guard

typeid_dict = {
    'int16': int(core.VarDesc.VarType.INT16),
    'int32': int(core.VarDesc.VarType.INT32),
    'int64': int(core.VarDesc.VarType.INT64),
    'float32': int(core.VarDesc.VarType.FP32),
    'float16': int(core.VarDesc.VarType.FP16),
    'bfloat16': int(core.VarDesc.VarType.BF16),
    'bool': int(core.VarDesc.VarType.BOOL),
    'int8': int(core.VarDesc.VarType.INT8),
    'uint8': int(core.VarDesc.VarType.UINT8),
    'float64': int(core.VarDesc.VarType.FP64),
}


class XPUTestCastOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'cast'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestCastOp
        classes = []
        for out_type in {
            'float16',
            'bfloat16',
            'float32',
            'int32',
            'int64',
            'int8',
            'uint8',
            'bool',
            'float64',
        }:
            class_name = 'XPUTestCastOp_outtype_' + out_type
            attr_dict = {'out_typename': out_type}
            classes.append([class_name, attr_dict])
        return base_class, classes

    class TestCastOp(XPUOpTest):
        def setUp(self):
            self.init_shape()
            ipt = np.random.random(size=self.shape)
            in_typename = self.in_type_str
            out_typename = (
                'float32'
                if not hasattr(self, 'out_typename')
                else self.out_typename
            )

            if in_typename == "bfloat16":
                ipt_x = convert_float_to_uint16(ipt)
            else:
                ipt_x = ipt.astype(in_typename)

            if out_typename == "bfloat16":
                opt = convert_uint16_to_float(convert_float_to_uint16(ipt_x))
            else:
                opt = ipt_x.astype(out_typename)

            self.inputs = {'X': ipt_x}
            self.outputs = {'Out': opt}
            self.attrs = {
                'in_dtype': typeid_dict[in_typename],
                'out_dtype': typeid_dict[out_typename],
            }
            self.op_type = 'cast'
            self.__class__.no_need_check_grad = True

        def init_shape(self):
            self.shape = [10, 10]

        def test_check_output(self):
            self.check_output()

    @check_run_big_shape_test()
    class TestCastOpLargeShape1(TestCastOp):
        def init_shape(self):
            self.shape = [1, 8192, 5120]

    @check_run_big_shape_test()
    class TestCastOpLargeShape2(TestCastOp):
        def init_shape(self):
            self.shape = [1, 8192]

    @check_run_big_shape_test()
    class TestCastOpLargeShape3(TestCastOp):
        def init_shape(self):
            self.shape = [1, 8192, 1, 128]

    @check_run_big_shape_test()
    class TestCastOpLargeShape4(TestCastOp):
        def init_shape(self):
            self.shape = [8192]

    @check_run_big_shape_test()
    class TestCastOpLargeShape5(TestCastOp):
        def init_shape(self):
            self.shape = [31776]

    @check_run_big_shape_test()
    class TestCastOpLargeShape6(TestCastOp):
        def init_shape(self):
            self.shape = [5120]

    @check_run_big_shape_test()
    class TestCastOpLargeShape7(TestCastOp):
        def init_shape(self):
            self.shape = [3456]

    @check_run_big_shape_test()
    class TestCastOpLargeShape8(TestCastOp):
        def init_shape(self):
            self.shape = [1920]

    @check_run_big_shape_test()
    class TestCastOpLargeShape9(TestCastOp):
        def init_shape(self):
            self.shape = [31776, 5120]

    @check_run_big_shape_test()
    class TestCastOpLargeShape10(TestCastOp):
        def init_shape(self):
            self.shape = [5120, 1920]

    @check_run_big_shape_test()
    class TestCastOpLargeShape11(TestCastOp):
        def init_shape(self):
            self.shape = [640, 5120]

    @check_run_big_shape_test()
    class TestCastOpLargeShape12(TestCastOp):
        def init_shape(self):
            self.shape = [5120, 3456]

    @check_run_big_shape_test()
    class TestCastOpLargeShape13(TestCastOp):
        def init_shape(self):
            self.shape = [1728, 5120]

    @check_run_big_shape_test()
    class TestCastOpLargeShape14(TestCastOp):
        def init_shape(self):
            self.shape = [5120, 31776]

    @check_run_big_shape_test()
    class TestCastOpLargeShape15(TestCastOp):
        def init_shape(self):
            self.shape = [27724]

    @check_run_big_shape_test()
    class TestCastOpLargeShape16(TestCastOp):
        def init_shape(self):
            self.shape = [32, 5120]

    @check_run_big_shape_test()
    class TestCastOpLargeShape17(TestCastOp):
        def init_shape(self):
            self.shape = [1728, 32]

    @check_run_big_shape_test()
    class TestCastOpLargeShape18(TestCastOp):
        def init_shape(self):
            self.shape = [32, 3456]

    @check_run_big_shape_test()
    class TestCastOpLargeShape19(TestCastOp):
        def init_shape(self):
            self.shape = [32, 3456]

    @check_run_big_shape_test()
    class TestCastOpLargeShape20(TestCastOp):
        def init_shape(self):
            self.shape = [5120, 32]

    @check_run_big_shape_test()
    class TestCastOpLargeShape21(TestCastOp):
        def init_shape(self):
            self.shape = [640, 32]

    @check_run_big_shape_test()
    class TestCastOpLargeShape22(TestCastOp):
        def init_shape(self):
            self.shape = [32, 1920]

    @check_run_big_shape_test()
    class TestCastOpLargeShape23(TestCastOp):
        def init_shape(self):
            self.shape = [19984]


support_types = get_xpu_op_support_types('cast')
for stype in support_types:
    create_test_class(globals(), XPUTestCastOp, stype)


class TestCastOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of cast_op must be Variable.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.XPUPlace(0)
            )
            self.assertRaises(TypeError, paddle.cast, x1, 'int32')


class TestCastOpEmpty(unittest.TestCase):
    def test_cast_op_empty(self):
        if paddle.is_compiled_with_xpu():
            paddle.set_device('xpu')
            paddle.disable_static()
            data = paddle.ones([0, 10], dtype='float32')
            out = paddle.cast(data, 'int32')
            self.assertEqual(out.shape, data.shape)
            self.assertEqual(out.dtype, paddle.int32)
            paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
