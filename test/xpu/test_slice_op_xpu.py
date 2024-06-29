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
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


# Situation 1: starts(list, no tensor), ends(list, no tensor)
# 1.1 without attr(decrease)
class XPUTestSliceOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'slice'
        self.use_dynamic_create_class = False

    class TestSliceOp(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = "slice"
            self.config()
            self.inputs = {'Input': self.input}
            self.outputs = {'Out': self.out}
            self.attrs = {
                'axes': self.axes,
                'starts': self.starts,
                'ends': self.ends,
                'infer_flags': self.infer_flags,
                "use_xpu": True,
            }

        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [1, 0, 2]
            self.ends = [3, 3, 4]
            self.axes = [0, 1, 2]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[1:3, 0:3, 2:4, :]

        def test_check_grad_normal(self):
            if self.dtype == np.float16:
                self.check_grad_with_place(self.place, ['Input'], 'Out')
            else:
                user_defined_grad_outputs = np.random.random(
                    self.out.shape
                ).astype(self.dtype)
                self.check_grad_with_place(
                    self.place,
                    ['Input'],
                    'Out',
                    user_defined_grad_outputs=user_defined_grad_outputs,
                )

    class TestCase1(TestSliceOp):
        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [-3, 0, 2]
            self.ends = [3, 100, -1]
            self.axes = [0, 1, 2]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[-3:3, 0:100, 2:-1, :]

    class TestCase2(TestSliceOp):
        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [-3, 0, 2]
            self.ends = [3, 100, -1]
            self.axes = [0, 1, 3]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[-3:3, 0:100, :, 2:-1]

    @check_run_big_shape_test()
    class TestCaseLargeShape1(TestSliceOp):
        def config(self):
            self.input = np.random.random([8192, 5120])
            self.starts = [0, 5119]
            self.ends = [8192, 5120]
            self.axes = [0, 1]
            self.infer_flags = [1, 1]
            self.out = self.input[:, -1:]

    @check_run_big_shape_test()
    class TestCaseLargeShape2(TestSliceOp):
        def config(self):
            self.input = np.random.random([2, 1, 8192, 1, 128])
            self.starts = [0, 0, 0, 0, 0]
            self.ends = [2, 1, 1, 1, 128]
            self.axes = [0, 1, 2, 3, 4]
            self.infer_flags = [1, 1, 1, 1, 1]
            self.out = self.input[:, :, -1:, :, :]


# 1.2 with attr(decrease)
class XPUTestSliceOp_decs_dim(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'slice'
        self.use_dynamic_create_class = False

    class TestSliceOp_decs_dim(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = "slice"
            self.config()
            self.inputs = {'Input': self.input}
            self.outputs = {'Out': self.out}
            self.attrs = {
                'axes': self.axes,
                'starts': self.starts,
                'ends': self.ends,
                'infer_flags': self.infer_flags,
                'decrease_axis': self.decrease_axis,
                "use_xpu": True,
            }

        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [1, 0, 2]
            self.ends = [2, 3, 4]
            self.axes = [0, 1, 2]
            self.decrease_axis = [0]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[1, 0:3, 2:4, :]

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad_normal(self):
            if self.dtype == np.float16:
                self.check_grad_with_place(self.place, ['Input'], 'Out')
            else:
                user_defined_grad_outputs = np.random.random(
                    self.out.shape
                ).astype(self.dtype)
                self.check_grad_with_place(
                    self.place,
                    ['Input'],
                    'Out',
                    user_defined_grad_outputs=user_defined_grad_outputs,
                )

    class TestSliceOp_decs_dim_2(TestSliceOp_decs_dim):
        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [1, 0, 2]
            self.ends = [2, 1, 4]
            self.axes = [0, 1, 2]
            self.decrease_axis = [0, 1]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[1, 0, 2:4, :]

    class TestSliceOp_decs_dim_3(TestSliceOp_decs_dim):
        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [-1, 0, 2]
            self.ends = [1000000, 1, 4]
            self.axes = [0, 1, 2]
            self.decrease_axis = [0, 1]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[-1, 0, 2:4, :]

    class TestSliceOp_decs_dim_4(TestSliceOp_decs_dim):
        def config(self):
            self.input = np.random.random([3, 4, 5, 7]).astype(self.dtype)
            self.starts = [0, 1, 2, 3]
            self.ends = [1, 2, 3, 4]
            self.axes = [0, 1, 2, 3]
            self.decrease_axis = [0, 1, 2]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[0, 1, 2, 3:4]

    class TestSliceOp_decs_dim_5(TestSliceOp_decs_dim):
        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [-1]
            self.ends = [1000000]
            self.axes = [3]
            self.decrease_axis = [3]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[:, :, :, -1]

    class TestSliceOp_decs_dim_6(TestSliceOp_decs_dim):
        def config(self):
            self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
            self.starts = [0, 1, 2, 3]
            self.ends = [1, 2, 3, 4]
            self.axes = [0, 1, 2, 3]
            self.decrease_axis = [0, 1, 2, 3]
            self.infer_flags = [1, 1, 1]
            self.out = self.input[0, 1, 2, 3]


support_types = get_xpu_op_support_types('slice')
for stype in support_types:
    create_test_class(globals(), XPUTestSliceOp, stype)
    create_test_class(globals(), XPUTestSliceOp_decs_dim, stype)

if __name__ == '__main__':
    unittest.main()
