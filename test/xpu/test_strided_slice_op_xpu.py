# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def strided_slice_native_forward(input, axes, starts, ends, strides):
    dim = input.ndim
    start = []
    end = []
    stride = []
    for i in range(dim):
        start.append(0)
        end.append(input.shape[i])
        stride.append(1)

    for i in range(len(axes)):
        start[axes[i]] = starts[i]
        end[axes[i]] = ends[i]
        stride[axes[i]] = strides[i]

    result = {
        1: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0]
        ],
        2: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0], start[1] : end[1] : stride[1]
        ],
        3: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
        ],
        4: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
        ],
        5: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
            start[4] : end[4] : stride[4],
        ],
        6: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
            start[4] : end[4] : stride[4],
            start[5] : end[5] : stride[5],
        ],
    }[dim](input, start, end, stride)

    return result


class XPUTestStrideSliceOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'strided_slice'
        self.use_dynamic_create_class = False

    class XPUTestStrideSliceOp(XPUOpTest):
        def setUp(self):
            self.op_type = 'strided_slice'
            self.dtype = self.in_type
            self.initTestCase()
            self.input = np.random.random(self.inshape).astype(self.dtype)
            self.python_api = paddle.strided_slice
            self.output = strided_slice_native_forward(
                self.input, self.axes, self.starts, self.ends, self.strides
            )

            self.inputs = {'Input': self.input}
            self.outputs = {'Out': self.output}
            self.attrs = {
                'axes': self.axes,
                'starts': self.starts,
                'ends': self.ends,
                'strides': self.strides,
                'infer_flags': self.infer_flags,
            }

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            self.check_grad_with_place(paddle.XPUPlace(0), ['Input'], 'Out')

        def initTestCase(self):
            self.inshape = 100
            self.axes = [0]
            self.starts = [-4]
            self.ends = [-1]
            self.strides = [1]
            self.infer_flags = [1]

    class XPUTestStrideSliceOp1(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = 100
            self.axes = [0]
            self.starts = [3]
            self.ends = [8]
            self.strides = [1]
            self.infer_flags = [1]

    class XPUTestStrideSliceOp2(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (4, 8, 12)
            self.axes = [0, 1, 2]
            self.starts = [3, 4, 5]
            self.ends = [4, 5, 6]
            self.strides = [1, 1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp3(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (4, 8, 12, 4, 40)
            self.axes = [0, 1, 2, 3, 4]
            self.starts = [3, 4, 5, 1, 10]
            self.ends = [4, 5, 6, 2, 30]
            self.strides = [1, 1, 1, 2, 2]
            self.infer_flags = [1, 1, 1, 1, 1]

    class XPUTestStrideSliceOp4(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (3, 4, 10)
            self.axes = [0, 1, 2]
            self.starts = [0, -1, 0]
            self.ends = [2, -3, 5]
            self.strides = [1, -1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp5(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (5, 5, 5)
            self.axes = [0, 1, 2]
            self.starts = [1, 0, 0]
            self.ends = [2, 1, 3]
            self.strides = [1, 1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp6(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (5, 5, 5)
            self.axes = [0, 1, 2]
            self.starts = [1, -1, 0]
            self.ends = [2, -3, 3]
            self.strides = [1, -1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp7(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (5, 5, 5)
            self.axes = [0, 1, 2]
            self.starts = [1, 0, 0]
            self.ends = [2, 2, 3]
            self.strides = [1, 1, 1]
            self.infer_flags = [1, 1, 1]

    class XPUTestStrideSliceOp8(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (3, 3, 3, 6, 7, 8)
            self.axes = [0, 1, 2, 3, 4, 5]
            self.starts = [1, 0, 0, 0, 1, 2]
            self.ends = [2, 2, 3, 1, 2, 8]
            self.strides = [1, 1, 1, 1, 1, 2]
            self.infer_flags = [1, 1, 1, 1, 1]

    class XPUTestStrideSliceOp_eb_1(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (1, 4, 4096, 128)
            self.axes = [0, 1, 2, 3]
            self.starts = [0, 0, 0, 0]
            self.ends = [1, 4, 4096, 128]
            self.strides = [1, 1, 1, 2]
            self.infer_flags = [1, 1, 1, 1]

    class XPUTestStrideSliceOp_eb_2(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (1, 4, 4096, 128)
            self.axes = [0, 1, 2, 3]
            self.starts = [0, 0, 0, 1]
            self.ends = [1, 4, 4096, 128]
            self.strides = [1, 1, 1, 2]
            self.infer_flags = [1, 1, 1, 1]

    @check_run_big_shape_test()
    class XPUTestStrideSliceOpLargeShape1(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (1, 8192, 5, 128)
            self.axes = [0, 1, 2, 3]
            self.starts = [0, 0, 0, 0]
            self.ends = [1, 8192, 5, 128]
            self.strides = [1, 1, 1, 2]
            self.infer_flags = [1, 1, 1, 1]

    @check_run_big_shape_test()
    class XPUTestStrideSliceOpLargeShape2(XPUTestStrideSliceOp):
        def initTestCase(self):
            self.inshape = (8192, 3456)
            self.axes = [0, 1]
            self.starts = [0, 0]
            self.ends = [8192, 3456]
            self.strides = [1, 2]
            self.infer_flags = [1, 1]


support_types = get_xpu_op_support_types('strided_slice')
for stype in support_types:
    create_test_class(globals(), XPUTestStrideSliceOp, stype)

if __name__ == "__main__":
    unittest.main()
