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


import copy
import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()


class XPUTestElementwiseAddOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_add'
        self.use_dynamic_create_class = False

    class TestElementwiseAddOp(XPUOpTest):
        def setUp(self):
            self.op_type = "elementwise_add"
            self.init_dtype()
            self.init_input_output()
            self.init_axis()
            self.init_max_relative_error()
            self.inputs = {
                'X': OpTest.np_dtype_to_base_dtype(self.x),
                'Y': OpTest.np_dtype_to_base_dtype(self.y),
            }
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
            self.outputs = {'Out': self.out}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad_normal(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X', 'Y'],
                    'Out',
                    max_relative_error=self.max_relative_error,
                )

        def test_check_grad_ignore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    max_relative_error=self.max_relative_error,
                )

        def test_check_grad_ignore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    no_grad_set=set("Y"),
                    max_relative_error=self.max_relative_error,
                )

        def init_input_output(self):
            self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.out = np.add(self.x, self.y)

        def init_dtype(self):
            self.dtype = self.in_type

        def init_axis(self):
            self.axis = -1

        def init_max_relative_error(self):
            self.max_relative_error = 0.006

    class TestElementwiseAddOp_ZeroDim1(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.y = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_ZeroDim2(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.y = np.random.uniform(-1, 1, [13, 17]).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_ZeroDim3(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.uniform(-1, 1, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(-1, 1, []).astype(self.dtype)
            self.out = self.x + self.y

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast."
    )
    class TestElementwiseAddOp_scalar(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)
            self.out = self.x + self.y

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1,1) to test broadcast."
    )
    class TestElementwiseAddOp_scalar2(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1, 1).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_Vector(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.random((100,)).astype(self.dtype)
            self.y = np.random.random((100,)).astype(self.dtype)
            self.out = np.add(self.x, self.y)

    class TestElementwiseAddOp_broadcast_0(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(100, 2, 3).astype(self.dtype)
            self.y = np.random.rand(100).astype(self.dtype)
            self.out = self.x + self.y.reshape(100, 1, 1)

        def init_axis(self):
            self.axis = 0

    class TestElementwiseAddOp_broadcast_1(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 100, 3).astype(self.dtype)
            self.y = np.random.rand(100).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 100, 1)

        def init_axis(self):
            self.axis = 1

    class TestElementwiseAddOp_broadcast_2(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 3, 100).astype(self.dtype)
            self.y = np.random.rand(100).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 1, 100)

    class TestElementwiseAddOp_broadcast_3(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
            self.y = np.random.rand(10, 12).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 10, 12, 1)

        def init_axis(self):
            self.axis = 1

    class TestElementwiseAddOp_broadcast_4(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(100, 2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(100, 1).astype(self.dtype)
            self.out = self.x + self.y.reshape(100, 1, 1, 1)

        def init_axis(self):
            self.axis = 0

    class TestElementwiseAddOp_broadcast_5(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(10, 3, 12).astype(self.dtype)
            self.y = np.random.rand(10, 1, 12).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_broadcast_6(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 12, 3, 5).astype(self.dtype)
            self.y = np.random.rand(2, 12, 1, 5).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_broadcast_7(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(1, 1, 20, 5).astype(self.dtype)
            self.y = np.random.rand(20, 5, 1, 1).astype(self.dtype)
            self.out = self.x + self.y

    class TestElementwiseAddOp_rowwise_add_0(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 10, 12).astype(self.dtype)
            self.y = np.random.rand(10, 12).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 10, 12)

        def init_axis(self):
            self.axis = 1

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast."
    )
    class TestElementwiseAddOp_rowwise_add_1(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(100, 1).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)
            self.out = self.x + self.y.reshape(1, 1)

        def init_axis(self):
            self.axis = 1

    class TestElementwiseAddOp_channelwise_add(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(100, 2, 3).astype(self.dtype)
            self.y = np.random.rand(100, 1, 1).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            self.axis = -1

    class TestElementwiseAddOp_commonuse_add1(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(2, 3, 100).astype(self.dtype)
            self.y = np.random.rand(1, 1, 100).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            self.axis = -1

    class TestElementwiseAddOp_commonuse_add2(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(10, 3, 1, 4).astype(self.dtype)
            self.y = np.random.rand(10, 1, 12, 1).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            self.axis = -1

    class TestElementwiseAddOp_xsize_lessthan_ysize_add(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(10, 12).astype(self.dtype)
            self.y = np.random.rand(2, 3, 10, 12).astype(self.dtype)
            self.out = self.x + self.y

        def init_axis(self):
            self.axis = 2

    @check_run_big_shape_test()
    class TestElementwiseAddOpLargeShape1(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(8192, 1920).astype(self.dtype)
            self.y = np.random.rand(1920).astype(self.dtype)
            self.out = self.x + self.y

    @check_run_big_shape_test()
    class TestElementwiseAddOpLargeShape2(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(1, 8192, 5, 128).astype(self.dtype)
            self.y = np.random.rand(1, 8192, 5, 128).astype(self.dtype)
            self.out = self.x + self.y

    @check_run_big_shape_test()
    class TestElementwiseAddOpLargeShape3(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(1024, 5120).astype(self.dtype)
            self.y = np.random.rand(5120).astype(self.dtype)
            self.out = self.x + self.y

    @check_run_big_shape_test()
    class TestElementwiseAddOpLargeShape4(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(8192, 3456).astype(self.dtype)
            self.y = np.random.rand(3456).astype(self.dtype)
            self.out = self.x + self.y

    @check_run_big_shape_test()
    class TestElementwiseAddOpLargeShape5(TestElementwiseAddOp):
        def init_input_output(self):
            self.x = np.random.rand(1, 8192, 31776).astype(self.dtype)
            self.y = np.random.rand(31776).astype(self.dtype)
            self.out = self.x + self.y

    class TestAddOp(unittest.TestCase):
        def test_name(self):
            with base.program_guard(base.Program()):
                x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
                y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')

                y_1 = paddle.add(x, y, name='add_res')
                if not paddle.framework.use_pir_api():
                    self.assertEqual(('add_res' in y_1.name), True)

        def test_declarative(self):
            with base.program_guard(base.Program()):

                def gen_data():
                    return {
                        "x": np.array([2, 3, 4]).astype('float32'),
                        "y": np.array([1, 5, 2]).astype('float32'),
                    }

                x = paddle.static.data(name="x", shape=[3], dtype='float32')
                y = paddle.static.data(name="y", shape=[3], dtype='float32')
                z = paddle.add(x, y)

                place = base.XPUPlace(0)
                exe = base.Executor(place)
                z_value = exe.run(feed=gen_data(), fetch_list=[z])
                z_expected = np.array([3.0, 8.0, 6.0])
                self.assertEqual((z_value == z_expected).all(), True)

        def test_dygraph(self):
            with base.dygraph.guard():
                np_x = np.array([2, 3, 4]).astype('float32')
                np_y = np.array([1, 5, 2]).astype('float32')
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.add(x, y)
                np_z = z.numpy()
                z_expected = np.array([3.0, 8.0, 6.0])
                self.assertEqual((np_z == z_expected).all(), True)


support_types = get_xpu_op_support_types('elementwise_add')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseAddOp, stype)


@unittest.skipIf(
    core.get_xpu_device_version(0) != core.XPUVersion.XPU3,
    "only supported on XPU3",
)
class TestTensorFloat32Bfloat16OrFloat16Add(unittest.TestCase):
    def _float32_bfloat16_or_float16_add(self, y_dtype):
        paddle.disable_static()
        test_num = 5
        val_range = 10000
        shapes = []
        for i in range(test_num):
            shape = [
                np.random.randint(1, val_range),
                np.random.randint(1, val_range),
            ]
            shapes.append(shape)

        for i, shape in enumerate(shapes):
            x = paddle.randn(list(shape), dtype=paddle.float32)
            x_copy = copy.deepcopy(x)
            y = paddle.randn(list(shape), dtype=y_dtype)
            x.add_(y)
            x_copy.add_(paddle.cast(y, paddle.float32))
            np.testing.assert_equal(x.numpy(), x_copy.numpy())
            del x, x_copy

    def test_float32_bfloat16_add(self):
        self._float32_bfloat16_or_float16_add(y_dtype=paddle.bfloat16)

    def test_float32_float16_add(self):
        self._float32_bfloat16_or_float16_add(y_dtype=paddle.float16)


if __name__ == '__main__':
    unittest.main()
