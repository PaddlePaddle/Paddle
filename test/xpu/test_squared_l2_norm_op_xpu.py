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
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle


class XPUTestSquaredL2NormOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'squared_l2_norm'
        self.use_dynamic_create_class = False

    class TestSquaredL2NormOp(XPUOpTest):
        def init(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = 'squared_l2_norm'

        def setUp(self):
            self.init()
            self.use_mkldnn = False
            self.max_relative_error = 0.05
            self.set_inputs()

            if self.dtype == np.uint16:
                # bfloat16 actually
                new_x = convert_float_to_uint16(self.x)
            else:
                new_x = self.x.astype(self.dtype)

            out = np.square(np.linalg.norm(self.x))

            if self.dtype == np.uint16:
                # bfloat16 actually
                new_out = convert_float_to_uint16(out)
            else:
                new_out = out.astype(self.dtype)

            new_out = np.array([new_out])

            self.inputs = {'X': new_x}
            self.outputs = {'Out': new_out}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def set_inputs(self):
            self.x = np.random.uniform(-1, 1, (13, 19))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.1

    class TestSquaredL2NormOp_1(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.2, 0.2, (8, 128, 24))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.02

    class TestSquaredL2NormOp_2(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (2, 128, 256))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape1(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (5120, 32))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape2(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (32, 1920))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape3(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (640, 32))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape4(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (32, 5120))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape5(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (32, 3456))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape6(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (1728, 32))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape7(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (31776, 5120))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape8(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (5120, 1920))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape9(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (1920,))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape10(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (640, 5120))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape11(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (5120, 3456))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape12(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (3456,))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape13(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (1728, 5120))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape14(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (5120, 31776))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01

    @check_run_big_shape_test()
    class TestSquaredL2NormOpLargeShape15(TestSquaredL2NormOp):
        def set_inputs(self):
            self.x = np.random.uniform(-0.1, 0.1, (31776,))
            self.x[np.abs(self.x) < self.max_relative_error] = 0.01


support_types = get_xpu_op_support_types('squared_l2_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestSquaredL2NormOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    paddle.seed(10)
    unittest.main()
