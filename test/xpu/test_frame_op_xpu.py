# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from numpy.lib.stride_tricks import as_strided
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def frame_from_numpy(x, frame_length, hop_length, axis=-1):
    if axis == -1 and not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    elif axis == 0 and not x.flags["F_CONTIGUOUS"]:
        x = np.asfortranarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    if axis == -1:
        shape = [*list(x.shape)[:-1], frame_length, n_frames]
        strides = [*strides, hop_length * x.itemsize]
    else:
        shape = [n_frames, frame_length, *list(x.shape)[1:]]
        strides = [hop_length * x.itemsize, *strides]

    return as_strided(x, shape=shape, strides=strides)


class XPUTestFrameOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "frame"

    class TestFrameOpBase(XPUOpTest):
        def setUp(self):
            self.op_type = "frame"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_case()
            np.random.seed(2026)

            if self.dtype == np.uint16:
                x_fp32 = np.random.uniform(-1.0, 1.0, size=self.shape).astype(
                    np.float32
                )
                self.inputs = {"X": convert_float_to_uint16(x_fp32)}
                out = frame_from_numpy(
                    x_fp32, self.frame_length, self.hop_length, self.axis
                ).copy()
                self.outputs = {"Out": convert_float_to_uint16(out)}
            elif np.issubdtype(self.dtype, np.integer):
                x = np.random.randint(-3, 4, size=self.shape).astype(self.dtype)
                self.inputs = {"X": x}
                self.outputs = {
                    "Out": frame_from_numpy(
                        x, self.frame_length, self.hop_length, self.axis
                    ).copy()
                }
            else:
                x = np.random.uniform(-1.0, 1.0, size=self.shape).astype(
                    self.dtype
                )
                self.inputs = {"X": x}
                self.outputs = {
                    "Out": frame_from_numpy(
                        x, self.frame_length, self.hop_length, self.axis
                    ).copy()
                }

            self.attrs = {
                "frame_length": self.frame_length,
                "hop_length": self.hop_length,
                "axis": self.axis,
            }

        def init_case(self):
            self.shape = (150,)
            self.frame_length = 50
            self.hop_length = 15
            self.axis = -1

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            # int/int64 不支持梯度；其余类型沿用 frame op 梯度检查。
            if self.dtype in [np.int32, np.int64]:
                return
            if self.dtype in [np.float16, np.uint16]:
                self.check_grad_with_place(
                    self.place, ["X"], "Out", max_relative_error=1e-2
                )
            else:
                self.check_grad_with_place(self.place, ["X"], "Out")

    class TestFrameAxis0Case1(TestFrameOpBase):
        def init_case(self):
            self.shape = (150,)
            self.frame_length = 50
            self.hop_length = 15
            self.axis = 0

    class TestFrameAxisNeg1Case2(TestFrameOpBase):
        def init_case(self):
            self.shape = (8, 150)
            self.frame_length = 50
            self.hop_length = 15
            self.axis = -1

    class TestFrameAxis0Case3(TestFrameOpBase):
        def init_case(self):
            self.shape = (150, 8)
            self.frame_length = 50
            self.hop_length = 15
            self.axis = 0

    class TestFrameAxisNeg1Case4(TestFrameOpBase):
        def init_case(self):
            self.shape = (4, 2, 150)
            self.frame_length = 50
            self.hop_length = 15
            self.axis = -1

    class TestFrameAxis0Case5(TestFrameOpBase):
        def init_case(self):
            self.shape = (150, 4, 2)
            self.frame_length = 50
            self.hop_length = 15
            self.axis = 0

    class TestFrameSmallCase(TestFrameOpBase):
        def init_case(self):
            self.shape = (17,)
            self.frame_length = 2
            self.hop_length = 3
            self.axis = -1

    class TestFrameLenEqSeqAxisNeg1(TestFrameOpBase):
        def init_case(self):
            self.shape = (31,)
            self.frame_length = 31
            self.hop_length = 7
            self.axis = -1

    class TestFrameLenEqSeqAxis0(TestFrameOpBase):
        def init_case(self):
            self.shape = (31, 2)
            self.frame_length = 31
            self.hop_length = 7
            self.axis = 0

    class TestFrameLen1AxisNeg1(TestFrameOpBase):
        def init_case(self):
            self.shape = (3, 19)
            self.frame_length = 1
            self.hop_length = 1
            self.axis = -1

    class TestFrameLen1Axis0(TestFrameOpBase):
        def init_case(self):
            self.shape = (19, 3)
            self.frame_length = 1
            self.hop_length = 1
            self.axis = 0

    class TestFrameNoOverlapAxisNeg1(TestFrameOpBase):
        def init_case(self):
            self.shape = (2, 19)
            self.frame_length = 4
            self.hop_length = 5
            self.axis = -1

    class TestFrameNoOverlapAxis0(TestFrameOpBase):
        def init_case(self):
            self.shape = (19, 2)
            self.frame_length = 4
            self.hop_length = 5
            self.axis = 0

    class TestFrameHop1OverlapAxisNeg1(TestFrameOpBase):
        def init_case(self):
            self.shape = (2, 129)
            self.frame_length = 2
            self.hop_length = 1
            self.axis = -1

    class TestFrameHop1OverlapAxis0(TestFrameOpBase):
        def init_case(self):
            self.shape = (129, 2)
            self.frame_length = 2
            self.hop_length = 1
            self.axis = 0

    class TestFrame3DAxisNeg1(TestFrameOpBase):
        def init_case(self):
            self.shape = (2, 5, 19)
            self.frame_length = 4
            self.hop_length = 3
            self.axis = -1

    class TestFrame3DAxis0(TestFrameOpBase):
        def init_case(self):
            self.shape = (19, 2, 5)
            self.frame_length = 4
            self.hop_length = 3
            self.axis = 0

    class TestFrame4DAxisNeg1(TestFrameOpBase):
        def init_case(self):
            self.shape = (2, 3, 4, 33)
            self.frame_length = 2
            self.hop_length = 3
            self.axis = -1

    class TestFrame4DAxis0(TestFrameOpBase):
        def init_case(self):
            self.shape = (33, 2, 3, 4)
            self.frame_length = 2
            self.hop_length = 3
            self.axis = 0


support_types = get_xpu_op_support_types("frame")
for stype in support_types:
    create_test_class(globals(), XPUTestFrameOp, stype)


if __name__ == "__main__":
    unittest.main()
