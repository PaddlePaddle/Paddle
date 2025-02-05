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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestTakeAlongAxis(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'take_along_axis'

    class TestXPUTakeAlongAxisOp(XPUOpTest):
        def setUp(self):
            self.op_type = "take_along_axis"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type

            self.init_config()
            xnp = np.random.random(self.x_shape).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )
            self.target = np.take_along_axis(xnp, self.index, self.axis)
            broadcast_shape_list = list(self.x_shape)
            broadcast_shape_list[self.axis] = self.index.shape[self.axis]
            self.broadcast_shape = tuple(broadcast_shape_list)
            self.index_broadcast = np.broadcast_to(
                self.index, self.broadcast_shape
            )
            self.inputs = {
                'Input': (
                    xnp
                    if self.dtype != np.uint16
                    else convert_float_to_uint16(xnp)
                ),
                'Index': self.index_broadcast,
            }
            self.attrs = {'Axis': self.axis}
            self.outputs = {'Result': self.target}

        def init_config(self):
            self.in_type = np.float32
            self.x_shape = (1, 4, 10)
            self.index_type = np.int32
            self.index = np.array([[[0, 1, 3, 5, 6]]]).astype(self.index_type)
            self.axis = 2

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                self.check_output_with_place(self.place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                self.check_grad_with_place(self.place, ['Input'], 'Result')

    class TestCase1(TestXPUTakeAlongAxisOp):
        def init_config(self):
            self.in_type = np.float32
            self.x_shape = (1, 10, 100)
            self.index_type = np.int32
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase2(TestXPUTakeAlongAxisOp):
        def init_config(self):
            self.in_type = np.float32
            self.x_shape = (1, 10, 100)
            self.index_type = np.int64
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase3(TestXPUTakeAlongAxisOp):
        def init_config(self):
            self.in_type = np.float16
            self.x_shape = (1, 10, 100)
            self.index_type = np.int32
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase4(TestXPUTakeAlongAxisOp):
        def init_config(self):
            self.in_type = np.float16
            self.x_shape = (1, 10, 100)
            self.index_type = np.int64
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2

    class TestCase5(TestXPUTakeAlongAxisOp):
        def init_config(self):
            self.in_type = np.float32
            self.x_shape = (1, 10, 100)
            self.index_type = np.int32
            self.index = np.array([[[0], [1], [3], [5], [8]]]).astype(
                self.index_type
            )
            self.axis = 1

    class TestCase6(TestXPUTakeAlongAxisOp):
        def init_config(self):
            self.in_type = np.uint16
            self.x_shape = (1, 10, 100)
            self.index_type = np.int64
            self.index = np.array([[[0, 1, 3, 5, 13]]]).astype(self.index_type)
            self.axis = 2


class XPUTestTakeAlongAxisAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3]
        self.index_shape = [1, 3]
        self.index_np = np.array([[0, 1, 2]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.XPUPlace(0)]
        self.axis = 0

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            index = paddle.static.data('Index', self.index_shape, "int64")
            out = paddle.take_along_axis(x, index, self.axis)
            exe = paddle.static.Executor(self.place[0])
            res = exe.run(
                feed={'X': self.x_np, 'Index': self.index_np}, fetch_list=[out]
            )
        out_ref = np.array(
            np.take_along_axis(self.x_np, self.index_np, self.axis)
        )
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place[0])
        x_tensor = paddle.to_tensor(self.x_np)
        self.index = paddle.to_tensor(self.index_np)
        out = paddle.take_along_axis(x_tensor, self.index, self.axis)
        out_ref = np.array(
            np.take_along_axis(self.x_np, self.index_np, self.axis)
        )
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()


class TestTakeAlongAxisAPICase1(XPUTestTakeAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [4, 2]
        self.index_np = np.array([[0, 0], [1, 0], [0, 0], [1, 0]]).astype(
            'int64'
        )
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.XPUPlace(0)]
        self.axis = 0


support_types = get_xpu_op_support_types('take_along_axis')
for stype in support_types:
    create_test_class(globals(), XPUTestTakeAlongAxis, stype)

if __name__ == "__main__":
    unittest.main()
