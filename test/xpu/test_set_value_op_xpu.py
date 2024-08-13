#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# Test set_value op in static graph mode

import sys
import unittest
from functools import reduce

import numpy as np

sys.path.append("../")
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle
from paddle.base.layer_helper import LayerHelper


class XPUTestSetValueOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'set_value'
        self.use_dynamic_create_class = False

    class XPUTestSetValueBase(XPUOpTest):
        def setUp(self):
            paddle.enable_static()
            self.__class__.op_type = "set_value"
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)
            self.set_dtype()
            self.set_value()
            self.set_shape()
            dtype = self.dtype
            if self.dtype == "bfloat16":
                dtype = "float32"
            self.data = np.ones(self.shape).astype(dtype)
            self.program = paddle.static.Program()

        def set_shape(self):
            self.shape = [2, 3, 4]

        def set_value(self):
            self.value = 6

        def set_dtype(self):
            self.dtype = self.in_type
            if self.in_type == np.bool_:
                self.dtype = "bool"
            elif self.in_type == np.uint16:
                self.dtype = "bfloat16"

        def _call_setitem(self, x):
            x[0, 0] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (0, 0), self.value)
            return x

        def _get_answer(self):
            self.data[0, 0] = self.value

    class XPUTestSetValueApi(XPUTestSetValueBase):
        def _run_static(self):
            paddle.enable_static()
            with paddle.static.program_guard(self.program):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                x = self._call_setitem_static_api(x)

            exe = paddle.static.Executor(self.place)
            out = exe.run(self.program, fetch_list=[x])
            paddle.disable_static()
            return out

        def _run_dynamic(self):
            paddle.disable_static()
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            self._call_setitem(x)
            out = x.numpy()
            paddle.enable_static()
            return out

        def test_api(self):
            self._get_answer()
            static_out = self._run_static()
            dynamic_out = self._run_dynamic()
            if self.dtype == "bfloat16":
                self.data = convert_float_to_uint16(self.data)
            error_msg = (
                "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
            )
            self.assertTrue(
                (self.data == static_out).all(),
                msg=error_msg.format("static", self.data, static_out),
            )
            self.assertTrue(
                (self.data == dynamic_out).all(),
                msg=error_msg.format("dynamic", self.data, dynamic_out),
            )

    # 1. Test different type of item: int, Python slice, Paddle Tensor
    # 1.1 item is int
    class XPUTestSetValueItemInt(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, 0, self.value)
            return x

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueItemInt2(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [6, 6, 6, 6, 6]

        def _call_setitem(self, x):
            x[0, 3, 4] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (0, 3, 4), self.value)
            return x

        def _get_answer(self):
            self.data[0, 3, 4] = self.value

    class XPUTestSetValueItemInt3(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [6, 6, 6, 6, 6]

        def _call_setitem(self, x):
            x[1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (1), self.value)
            return x

        def _get_answer(self):
            self.data[1] = self.value

    # 1.2 item is slice
    # 1.2.1 step is 1
    class XPUTestSetValueItemSlice(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:2] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(0, 2), self.value)
            return x

        def _get_answer(self):
            self.data[0:2] = self.value

    class XPUTestSetValueItemSlice2(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:-1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(0, -1), self.value)
            return x

        def _get_answer(self):
            self.data[0:-1] = self.value

    class XPUTestSetValueItemSlice3(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:-1, 0:2] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x, (slice(0, -1), slice(0, 2)), self.value
            )
            return x

        def _get_answer(self):
            self.data[0:-1, 0:2] = self.value

    class XPUTestSetValueItemSlice4(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, 1:2, :] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x,
                (slice(0, None), slice(1, 2), slice(None, None, None)),
                self.value,
            )
            return x

        def _get_answer(self):
            self.data[0:, 1:2, :] = self.value

    class XPUTestSetValueItemSlice5(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, 1:1, :] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x, (slice(0), slice(1, 1), slice(None, None, None)), self.value
            )
            return x

        def _get_answer(self):
            self.data[0:, 1:1, :] = self.value

    class XPUTestSetValueItemSliceInWhile(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            elif self.in_type == np.bool_:
                self.dtype = "bool"
            elif self.in_type == np.uint16:
                self.dtype = "bfloat16"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            def cond(i, x):
                return i < 1

            def body(i, x):
                x[i] = self.value
                i = i + 1
                return i, x

            i = paddle.zeros(shape=[], dtype='int32')
            i, x = paddle.static.nn.while_loop(cond, body, [i, x])

        def _call_setitem_static_api(self, x):
            def cond(i, x):
                return i < 1

            def body(i, x):
                x = paddle.static.setitem(x, i, self.value)
                i = i + 1
                return i, x

            i = paddle.zeros(shape=[], dtype='int32')
            i, x = paddle.static.nn.while_loop(cond, body, [i, x])
            return x

        def _get_answer(self):
            self.data[0] = self.value

    # 1.2.2 step > 1
    class XPUTestSetValueItemSliceStep(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [5, 5, 5]

        def _call_setitem(self, x):
            x[0:2:2] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(0, 2, 2), self.value)
            return x

        def _get_answer(self):
            self.data[0:2:2] = self.value

    class XPUTestSetValueItemSliceStep2(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [7, 5, 5]

        def _call_setitem(self, x):
            x[0:-1:3] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(0, -1, 3), self.value)
            return x

        def _get_answer(self):
            self.data[0:-1:3] = self.value

    class XPUTestSetValueItemSliceStep3(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:-1, 0:2, ::2] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x, (slice(0, -1), slice(0, 2), slice(None, None, 2)), self.value
            )
            return x

        def _get_answer(self):
            self.data[0:-1, 0:2, ::2] = self.value

    class XPUTestSetValueItemSliceStep4(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, 1:2:2, :] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x,
                (slice(0, None), slice(1, 2, 2), slice(None, None, None)),
                self.value,
            )
            return x

        def _get_answer(self):
            self.data[0:, 1:2:2, :] = self.value

    # 1.2.3 step < 0
    class XPUTestSetValueItemSliceNegativeStep(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type in [np.float16, np.uint16]:
                self.dtype = "float32"
            elif self.in_type == np.bool_:
                self.dtype = "bool"
            else:
                self.dtype = self.in_type

        def set_shape(self):
            self.shape = [5, 2]

        def set_value(self):
            self.value = np.array([3, 4])

        def _call_setitem(self, x):
            x[5:2:-1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(5, 2, -1), self.value)
            return x

        def _get_answer(self):
            self.data[5:2:-1] = self.value

    class XPUTestSetValueItemSliceNegativeStep2(
        XPUTestSetValueItemSliceNegativeStep
    ):
        def set_shape(self):
            self.shape = [5]

        def set_value(self):
            self.value = np.array([3, 4])
            # print("self.value: \n", self.value)

        def _call_setitem(self, x):
            x[1::-1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(1, None, -1), self.value)
            return x

        def _get_answer(self):
            self.data[1::-1] = self.value

    class XPUTestSetValueItemSliceNegativeStep3(
        XPUTestSetValueItemSliceNegativeStep
    ):
        def set_shape(self):
            self.shape = [3]

        def set_value(self):
            self.value = np.array([3, 4, 5])

        def _call_setitem(self, x):
            x[::-1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(None, None, -1), self.value)
            return x

        def _get_answer(self):
            self.data[::-1] = self.value

    class XPUTestSetValueItemSliceNegativeStep4(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            elif self.in_type == np.bool_:
                self.dtype = "bool"
            elif self.in_type == np.uint16:
                self.dtype = "bfloat16"
            else:
                self.dtype = self.in_type

        def set_shape(self):
            self.shape = [3, 4, 5]

        def _call_setitem(self, x):
            x[2:0:-1, 0:2, ::-1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x,
                (slice(2, 0, -1), slice(0, 2), slice(None, None, -1)),
                self.value,
            )
            return x

        def _get_answer(self):
            self.data[2:0:-1, 0:2, ::-1] = self.value

        # 1.2.3 step < 0 and stride < -1

    class XPUTestSetValueItemSliceNegativeStep5(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            elif self.in_type == np.bool_:
                self.dtype = "bool"
            else:
                self.dtype = self.in_type

        def set_shape(self):
            self.shape = [5, 5, 5]

        def _call_setitem(self, x):
            x[2:-1:-2] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(2, -1, -2), self.value)
            return x

        def _get_answer(self):
            paddle.enable_static()
            with paddle.static.program_guard(self.program):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                x = self._call_setitem_static_api(x)

            exe = paddle.static.Executor(paddle.CPUPlace())
            self.data = exe.run(self.program, fetch_list=[x])
            paddle.disable_static()

        def test_api(self):
            self._get_answer()
            static_out = self._run_static()
            dynamic_out = self._run_dynamic()

            error_msg = (
                "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
            )
            self.assertTrue(
                (self.data[0] == static_out[0]).all(),
                msg=error_msg.format("static", self.data, static_out),
            )
            self.assertTrue(
                (self.data == dynamic_out).all(),
                msg=error_msg.format("dynamic", self.data, dynamic_out),
            )

    # 1.3 item is Ellipsis
    class XPUTestSetValueItemEllipsis1(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, ..., 1:] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x, (slice(0, None), ..., slice(1, None)), self.value
            )
            return x

        def _get_answer(self):
            self.data[0:, ..., 1:] = self.value

    class XPUTestSetValueItemEllipsis2(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, ...] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (slice(0, None), ...), self.value)
            return x

        def _get_answer(self):
            self.data[0:, ...] = self.value

    class XPUTestSetValueItemEllipsis3(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[..., 1:] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (..., slice(1, None)), self.value)
            return x

        def _get_answer(self):
            self.data[..., 1:] = self.value

    class XPUTestSetValueItemEllipsis4(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[...] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (...), self.value)
            return x

        def _get_answer(self):
            self.data[...] = self.value

    # 1.4 item is Paddle Tensor
    class XPUTestSetValueItemTensor(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            elif self.in_type == np.bool_:
                self.dtype = "bool"
            elif self.in_type == np.uint16:
                self.dtype = "bfloat16"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            zero = paddle.full([], 0, dtype="int32")
            x[zero] = self.value

        def _call_setitem_static_api(self, x):
            zero = paddle.full([], 0, dtype="int32")
            x = paddle.static.setitem(x, zero, self.value)
            return x

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueItemTensor2(XPUTestSetValueItemTensor):
        def _call_setitem(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x[zero:two] = self.value

        def _call_setitem_static_api(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x = paddle.static.setitem(x, slice(zero, two), self.value)
            return x

        def _get_answer(self):
            self.data[0:2] = self.value

    class XPUTestSetValueItemTensor3(XPUTestSetValueItemTensor):
        def _call_setitem(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x[zero:-1, 0:two] = self.value

        def _call_setitem_static_api(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x = paddle.static.setitem(
                x, (slice(zero, -1), slice(0, two)), self.value
            )
            return x

        def _get_answer(self):
            self.data[0:-1, 0:2] = self.value

    class XPUTestSetValueItemTensor4(XPUTestSetValueItemTensor):
        def _call_setitem(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x[0:-1, zero:2, 0:6:two] = self.value

        def _call_setitem_static_api(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x = paddle.static.setitem(
                x, (slice(0, -1), slice(zero, 2), slice(0, 6, two)), self.value
            )
            return x

        def _get_answer(self):
            self.data[0:-1, 0:2, ::2] = self.value

    class XPUTestSetValueItemTensor5(XPUTestSetValueItemTensor):
        def _call_setitem(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x[zero:, 1:2:two, :] = self.value

        def _call_setitem_static_api(self, x):
            zero = paddle.full([], 0, dtype="int32")
            two = paddle.full([], 2, dtype="int64")
            x = paddle.static.setitem(
                x,
                (slice(zero, None), slice(1, 2, two), slice(None, None, None)),
                self.value,
            )
            return x

        def _get_answer(self):
            self.data[0:, 1:2:2, :] = self.value

    class XPUTestSetValueItemTensor6(XPUTestSetValueItemTensor):
        def set_shape(self):
            self.shape = [3, 4, 5]

        def _call_setitem(self, x):
            minus1 = paddle.full([], -1, dtype="int32")
            zero = paddle.full([], 0, dtype="int32")
            x[2:zero:minus1, 0:2, 10:-6:minus1] = self.value

        def _call_setitem_static_api(self, x):
            minus1 = paddle.full([], -1, dtype="int32")
            zero = paddle.full([], 0, dtype="int32")
            x = paddle.static.setitem(
                x,
                (slice(2, zero, minus1), slice(0, 2), slice(10, -6, minus1)),
                self.value,
            )
            return x

        def _get_answer(self):
            self.data[2:0:-1, 0:2, ::-1] = self.value

    # 1.5 item is None
    class XPUTestSetValueItemNone1(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type in [np.float16, np.uint16]:
                self.dtype = "float32"
            elif self.in_type == np.bool_:
                self.dtype = "bool"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[None] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, None, self.value)
            return x

        def _get_answer(self):
            self.data[None] = self.value

    class XPUTestSetValueItemNone2(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[0, None, 1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (0, None, 1), self.value)
            return x

        def _get_answer(self):
            self.data[0, None, 1] = self.value

    class XPUTestSetValueItemNone3(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[:, None, None, 1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x, (slice(None, None, None), None, None, 1), self.value
            )
            return x

        def _get_answer(self):
            self.data[:, None, None, 1] = self.value

    class XPUTestSetValueItemNone4(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[0, 0, None, 1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (0, 0, None, 1), self.value)
            return x

        def _get_answer(self):
            self.data[0, 0, None, 1] = self.value

    class XPUTestSetValueItemNone5(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[0, None, 0, None, 1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (0, None, 0, None, 1), self.value)
            return x

        def _get_answer(self):
            self.data[0, None, 0, None, 1] = self.value

    class XPUTestSetValueItemNone6(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[None, 0, 0, None, 0] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (None, 0, 0, None, 0), self.value)
            return x

        def _get_answer(self):
            self.data[None, 0, 0, None, 0] = self.value

    class XPUTestSetValueItemNone7(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[:, None, 1] = np.zeros(self.shape)[:, None, 0]

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x,
                (slice(None, None, None), None, 1),
                np.zeros(self.shape)[:, None, 0],
            )
            return x

        def _get_answer(self):
            self.data[:, None, 1] = np.zeros(self.shape)[:, None, 0]

    class XPUTestSetValueItemNone8(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[:, 1, None] = np.zeros(self.shape)[:, 0, None]

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x,
                (slice(None, None, None), 1, None),
                np.zeros(self.shape)[:, 0, None],
            )
            return x

        def _get_answer(self):
            self.data[:, 1, None] = np.zeros(self.shape)[:, 0, None]

    class XPUTestSetValueItemNone9(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x,
                (None, slice(None, None, None), 1, ..., None),
                np.zeros(self.shape)[0, 0, :, None],
            )
            return x

        def _get_answer(self):
            self.data[None, :, 1, ..., None] = np.zeros(self.shape)[
                0, 0, :, None
            ]

    class XPUTestSetValueItemNone10(XPUTestSetValueItemNone1):
        def _call_setitem(self, x):
            x[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x,
                (..., None, slice(None, None, None), None),
                np.zeros(self.shape)[..., None, :, None],
            )
            return x

        def _get_answer(self):
            self.data[..., None, :, None] = np.zeros(self.shape)[
                ..., None, :, None
            ]

    # 1.6 item is list or Tensor of bol
    class XPUTestSetValueItemBool1(XPUTestSetValueApi):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            x[[True, False]] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, [True, False], self.value)
            return x

        def _get_answer(self):
            self.data[[True, False]] = self.value

    class XPUTestSetValueItemBool2(XPUTestSetValueApi):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            x[[False, False]] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, [False, False], self.value)
            return x

        def _get_answer(self):
            self.data[[False, False]] = self.value

    class XPUTestSetValueItemBool3(XPUTestSetValueApi):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            x[[False, True]] = np.zeros(self.shape[2])

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, [False, True], np.zeros(self.shape[2]))
            return x

        def _get_answer(self):
            self.data[[False, True]] = np.zeros(self.shape[2])

    class XPUTestSetValueItemBool4(XPUTestSetValueApi):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            idx = paddle.assign(np.array([False, True]))
            x[idx] = np.zeros(self.shape[2])

        def _call_setitem_static_api(self, x):
            idx = paddle.assign(np.array([False, True]))
            x = paddle.static.setitem(x, idx, np.zeros(self.shape[2]))
            return x

        def _get_answer(self):
            self.data[np.array([False, True])] = np.zeros(self.shape[2])

    class XPUTestSetValueItemBool5(XPUTestSetValueApi):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            idx = paddle.assign(
                np.array([[False, True, False], [True, True, False]])
            )
            x[idx] = self.value

        def _call_setitem_static_api(self, x):
            idx = paddle.assign(
                np.array([[False, True, False], [True, True, False]])
            )
            x = paddle.static.setitem(x, idx, self.value)
            return x

        def _get_answer(self):
            self.data[np.array([[False, True, False], [True, True, False]])] = (
                self.value
            )

    class XPUTestSetValueItemBool6(XPUTestSetValueApi):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            x[0, ...] = 0
            x[x > 0] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, (0, ...), 0)
            x = paddle.static.setitem(x, x > 0, self.value)
            return x

        def _get_answer(self):
            self.data[0, ...] = 0
            self.data[self.data > 0] = self.value

    # 2. Test different type of value: Tensor
    # 2.1 value is a Paddle Tensor (int32, int64, float32, bool)
    def create_test_value_tensor_int32(parent):
        class XPUTestValueInt(parent):
            def set_dtype(self):
                self.dtype = "int32"

            def _call_setitem(self, x):
                value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
                x[0, 1] = value

            def _call_setitem_static_api(self, x):
                value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
                x = paddle.static.setitem(x, (0, 1), value)
                return x

            def _get_answer(self):
                self.data[0, 1] = 3

        cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt32")
        XPUTestValueInt.__name__ = cls_name
        globals()[cls_name] = XPUTestValueInt

    create_test_value_tensor_int32(XPUTestSetValueItemInt)
    create_test_value_tensor_int32(XPUTestSetValueItemSlice)
    create_test_value_tensor_int32(XPUTestSetValueItemSlice2)
    create_test_value_tensor_int32(XPUTestSetValueItemSlice3)
    create_test_value_tensor_int32(XPUTestSetValueItemSlice4)

    def create_test_value_tensor_int64(parent):
        class XPUTestValueInt(parent):
            def set_dtype(self):
                self.dtype = "int64"

            def _call_setitem(self, x):
                value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
                x[0, 1] = value

            def _call_setitem_static_api(self, x):
                value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
                x = paddle.static.setitem(x, (0, 1), value)
                return x

            def _get_answer(self):
                self.data[0, 1] = 3

        cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt64")
        XPUTestValueInt.__name__ = cls_name
        globals()[cls_name] = XPUTestValueInt

    create_test_value_tensor_int64(XPUTestSetValueItemInt)
    create_test_value_tensor_int64(XPUTestSetValueItemSlice)
    create_test_value_tensor_int64(XPUTestSetValueItemSlice2)
    create_test_value_tensor_int64(XPUTestSetValueItemSlice3)
    create_test_value_tensor_int64(XPUTestSetValueItemSlice4)

    def create_test_value_tensor_fp32(parent):
        class XPUTestValueInt(parent):
            def set_dtype(self):
                self.dtype = "float32"

            def _call_setitem(self, x):
                value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
                x[0, 1] = value

            def _call_setitem_static_api(self, x):
                value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
                x = paddle.static.setitem(x, (0, 1), value)
                return x

            def _get_answer(self):
                self.data[0, 1] = 3

        cls_name = "{}_{}".format(parent.__name__, "ValueTensorFp32")
        XPUTestValueInt.__name__ = cls_name
        globals()[cls_name] = XPUTestValueInt

    create_test_value_tensor_fp32(XPUTestSetValueItemInt)
    create_test_value_tensor_fp32(XPUTestSetValueItemSlice)
    create_test_value_tensor_fp32(XPUTestSetValueItemSlice2)
    create_test_value_tensor_fp32(XPUTestSetValueItemSlice3)
    create_test_value_tensor_fp32(XPUTestSetValueItemSlice4)

    def create_test_value_tensor_bool(parent):
        class XPUTestValueInt(parent):
            def set_dtype(self):
                self.dtype = "bool"

            def _call_setitem(self, x):
                value = paddle.full(
                    shape=[1], fill_value=False, dtype=self.dtype
                )
                x[0, 1] = value

            def _call_setitem_static_api(self, x):
                value = paddle.full(
                    shape=[1], fill_value=False, dtype=self.dtype
                )
                x = paddle.static.setitem(x, (0, 1), value)
                return x

            def _get_answer(self):
                self.data[0, 1] = False

        cls_name = "{}_{}".format(parent.__name__, "ValueTensorBool")
        XPUTestValueInt.__name__ = cls_name
        globals()[cls_name] = XPUTestValueInt

    create_test_value_tensor_bool(XPUTestSetValueItemInt)
    create_test_value_tensor_bool(XPUTestSetValueItemSlice)
    create_test_value_tensor_bool(XPUTestSetValueItemSlice2)
    create_test_value_tensor_bool(XPUTestSetValueItemSlice3)
    create_test_value_tensor_bool(XPUTestSetValueItemSlice4)

    # 3. Test different shape of value
    class XPUTestSetValueValueShape1(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type in [np.float16, np.uint16]:
                self.dtype = "float32"
            elif self.in_type == np.bool_:
                self.dtype = "bool"
            else:
                self.dtype = self.in_type

        def set_value(self):
            self.value = np.array([3, 4, 5, 6])  # shape is (4,)

        def _call_setitem(self, x):
            x[0] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, 0, self.value)
            return x

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueValueShape2(XPUTestSetValueValueShape1):
        def set_value(self):
            self.value = np.array([[3, 4, 5, 6]])  # shape is (1,4)

        def _call_setitem(self, x):
            x[0:1] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, slice(0, 1), self.value)
            return x

        def _get_answer(self):
            self.data[0:1] = self.value

    class XPUTestSetValueValueShape3(XPUTestSetValueValueShape1):
        def set_value(self):
            self.value = np.array(
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
            )  # shape is (3,4)

        def _call_setitem(self, x):
            x[0] = self.value

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, 0, self.value)
            return x

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueValueShape4(XPUTestSetValueValueShape1):
        def set_value(self):
            self.value = np.array(
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
            ).astype(
                self.dtype
            )  # shape is (3,4)

        def _call_setitem(self, x):
            x[0] = paddle.assign(self.value)  # x is Paddle.Tensor

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(x, 0, paddle.assign(self.value))
            return x

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueValueShape5(XPUTestSetValueValueShape1):
        def set_value(self):
            self.value = np.array([3, 3, 3]).astype(self.dtype)

        def set_shape(self):
            self.shape = [3, 4]

        def _call_setitem(self, x):
            x[:, 0] = paddle.assign(self.value)  # x is Paddle.Tensor

        def _call_setitem_static_api(self, x):
            x = paddle.static.setitem(
                x, (slice(None, None, None), 0), paddle.assign(self.value)
            )
            return x

        def _get_answer(self):
            self.data[:, 0] = self.value

    # 4. Test error
    class XPUTestError(XPUTestSetValueBase):
        def _value_type_error(self):
            with self.assertRaisesRegex(
                TypeError,
                "Only support to assign an integer, float, numpy.ndarray or paddle.Tensor",
            ):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                value = [1]
                if paddle.in_dynamic_mode():
                    x[0] = value
                else:
                    x = paddle.static.setitem(x, 0, value)

        def _dtype_error(self):
            with self.assertRaisesRegex(
                TypeError,
                "When assign a numpy.ndarray, integer or float to a paddle.Tensor, ",
            ):
                y = paddle.ones(shape=self.shape, dtype="float16")
                y[0] = 1

        def _step_error(self):
            with self.assertRaisesRegex(ValueError, "step can not be 0"):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                if paddle.in_dynamic_mode():
                    x[0:1:0] = self.value
                else:
                    x = paddle.static.setitem(x, slice(0, 1, 0), self.value)

        def _ellipsis_error(self):
            with self.assertRaisesRegex(
                IndexError, "An index can only have a single ellipsis"
            ):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                x[..., ...] = self.value
            with self.assertRaisesRegex(ValueError, "the start or end is None"):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                one = paddle.ones([1])
                x[::one] = self.value

        def _bool_list_error(self):
            with self.assertRaises(IndexError):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                if paddle.in_dynamic_mode():
                    x[[True, False], [True, False]] = 0
                else:
                    x = paddle.static.setitem(
                        x, ([True, False], [True, False]), 0
                    )

        def _bool_tensor_error(self):
            with self.assertRaises(IndexError):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                idx = paddle.assign([True, False, True])
                if paddle.in_dynamic_mode():
                    x[idx] = 0
                else:
                    x = paddle.static.setitem(x, idx, 0)

        def _broadcast_mismatch(self):
            program = paddle.static.Program()
            with paddle.static.program_guard(program):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                value = np.array([3, 4, 5, 6, 7])
                x = paddle.static.setitem(x, 0, value)
            exe = paddle.static.Executor(paddle.XPUPlace(0))
            with self.assertRaises(ValueError):
                exe.run(program)

        def test_error(self):
            paddle.enable_static()
            with paddle.static.program_guard(self.program):
                self._value_type_error()
                self._bool_list_error()
                self._bool_tensor_error()
            self._broadcast_mismatch()

    # 5. Test backward
    class XPUTestBackward(XPUOpTest):
        def setUp(self):
            self.__class__.op_type = "set_value"
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)

        def test_static(self):
            paddle.enable_static()
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()

            x_np = np.random.random(size=(4, 4)).astype('float32')
            y_np = np.random.random(size=(4, 4)).astype('float32')
            label_np = np.random.randint(2, size=(4, 1)).astype('int64')

            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[4, 4], dtype='float32')
                y = paddle.static.data(name="y", shape=[4, 4], dtype='float32')
                x.stop_gradient = False
                y.stop_gradient = False

                label = paddle.static.data(
                    name="label", shape=[4, 1], dtype='int64'
                )

                z = paddle.add(x, y)
                var = y[0, :]
                z = paddle.static.setitem(z, (0, slice(None)), var)

                prediction = paddle.static.nn.fc(
                    x=z, size=2, activation='softmax'
                )

                cost = paddle.nn.functional.cross_entropy(
                    input=prediction, label=label
                )
                loss = paddle.mean(cost)
                sgd = paddle.optimizer.SGD(learning_rate=0.01)
                sgd.minimize(loss)

            exe = paddle.static.Executor(self.place)
            exe.run(startup_program)

            if paddle.framework.use_pir_api():
                exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np, "label": label_np},
                    fetch_list=[],
                )
            else:
                var_grad, z_grad = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np, "label": label_np},
                    fetch_list=[var.name + "@GRAD", z.name + "@GRAD"],
                )
                self.assertTrue((var_grad == z_grad[0, :]).all())
            paddle.disable_static()

    class XPUTestGradientTruncated(XPUOpTest):
        def setUp(self):
            self.__class__.op_type = "set_value"
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)

        def test_consistent_with_competitor(self):
            paddle.disable_static()

            def set_value(t, value):
                a = t * t
                a[0, 1] = value
                y = a * a
                return y.sum()

            # case 1
            array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape(
                [1, 2, 1, 3, 1, 4]
            )
            value = np.arange(100, 104, dtype="float32").reshape(1, 4)

            inps = paddle.to_tensor(array, stop_gradient=False)
            value = paddle.to_tensor(value, stop_gradient=False)

            loss = set_value(inps, value)
            loss.backward()

            value_grad = np.array([[600.0, 606.0, 612.0, 618.0]])
            input_grad = np.array(
                [
                    [
                        [
                            [
                                [[4.0, 32.0, 108.0, 256.0]],
                                [[500.0, 864.0, 1372.0, 2048.0]],
                                [[2916.0, 4000.0, 5324.0, 6912.0]],
                            ]
                        ],
                        [
                            [
                                [[0.0, 0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0, 0.0]],
                            ]
                        ],
                    ]
                ]
            )
            np.testing.assert_array_equal(
                inps.grad.numpy(),
                input_grad,
                err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
            )
            np.testing.assert_array_equal(
                value.grad.numpy(),
                value_grad,
                err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
            )

            # case 2
            array = np.arange(1, 2 * 3 * 4 + 1, dtype="float32").reshape(
                [4, 2, 3]
            )
            value = np.arange(100, 100 + 1, dtype="float32")

            inps2 = paddle.to_tensor(array, stop_gradient=False)
            value2 = paddle.to_tensor(value, stop_gradient=False)

            loss = set_value(inps2, value2)
            loss.backward()

            value_grad2 = np.array([600.0])
            input_grad2 = np.array(
                [
                    [[4.0, 32.0, 108.0], [0.0, 0.0, 0.0]],
                    [[1372.0, 2048.0, 2916.0], [4000.0, 5324.0, 6912.0]],
                    [[8788.0, 10976.0, 13500.0], [16384.0, 19652.0, 23328.0]],
                    [[27436.0, 32000.0, 37044.0], [42592.0, 48668.0, 55296.0]],
                ]
            )
            np.testing.assert_array_equal(
                inps2.grad.numpy(),
                input_grad2,
                err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps2.grad.numpy()}',
            )
            np.testing.assert_array_equal(
                value2.grad.numpy(),
                value_grad2,
                err_msg=f'The gradient of input should be \n{value_grad},\n but received {value2.grad.numpy()}',
            )

            # case 3
            def set_value3(t, value):
                a = t * t
                a[0, :, 0, :] = value
                y = a * a
                return y.sum()

            array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape(
                [4, 3, 1, 1, 2, 1]
            )
            value = np.arange(100, 100 + 2, dtype="float32").reshape(1, 2, 1)

            inps = paddle.to_tensor(array, stop_gradient=False)
            value = paddle.to_tensor(value, stop_gradient=False)

            loss = set_value3(inps, value)
            loss.backward()

            value_grad = np.array([[[600.0], [606.0]]])
            input_grad = np.array(
                [
                    [
                        [[[[0.0], [0.0]]]],
                        [[[[0.0], [0.0]]]],
                        [[[[0.0], [0.0]]]],
                    ],
                    [
                        [[[[1372.0], [2048.0]]]],
                        [[[[2916.0], [4000.0]]]],
                        [[[[5324.0], [6912.0]]]],
                    ],
                    [
                        [[[[8788.0], [10976.0]]]],
                        [[[[13500.0], [16384.0]]]],
                        [[[[19652.0], [23328.0]]]],
                    ],
                    [
                        [[[[27436.0], [32000.0]]]],
                        [[[[37044.0], [42592.0]]]],
                        [[[[48668.0], [55296.0]]]],
                    ],
                ]
            )
            np.testing.assert_array_equal(
                inps.grad.numpy(),
                input_grad,
                err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
            )
            np.testing.assert_array_equal(
                value.grad.numpy(),
                value_grad,
                err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
            )

            # case 4: step >0
            def set_value4(t, value):
                a = t * t
                a[0, :, 0, ::3] = value
                y = a * a
                return y.sum()

            array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape(
                [2, 3, 1, 4, 1]
            )
            value = np.arange(100, 100 + 2, dtype="float32").reshape(1, 2, 1)

            inps = paddle.to_tensor(array, stop_gradient=False)
            value = paddle.to_tensor(value, stop_gradient=False)

            loss = set_value4(inps, value)
            loss.backward()

            value_grad = np.array([[[600.0], [606.0]]])
            input_grad = np.array(
                [
                    [
                        [[[0.0], [32.0], [108.0], [0.0]]],
                        [[[0.0], [864.0], [1372.0], [0.0]]],
                        [[[0.0], [4000.0], [5324.0], [0.0]]],
                    ],
                    [
                        [[[8788.0], [10976.0], [13500.0], [16384.0]]],
                        [[[19652.0], [23328.0], [27436.0], [32000.0]]],
                        [[[37044.0], [42592.0], [48668.0], [55296.0]]],
                    ],
                ]
            )
            np.testing.assert_array_equal(
                inps.grad.numpy(),
                input_grad,
                err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
            )
            np.testing.assert_array_equal(
                value.grad.numpy(),
                value_grad,
                err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
            )

            # case 5:a[0].shape==value.shape
            def set_value5(t, value):
                a = t * t
                a[0] = value
                y = a * a
                return y.sum()

            array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape(
                [2, 3, 4]
            )
            value = np.arange(100, 100 + 12, dtype="float32").reshape(3, 4)

            inps = paddle.to_tensor(array, stop_gradient=False)
            value = paddle.to_tensor(value, stop_gradient=False)

            loss = set_value5(inps, value)
            loss.backward()

            value_grad = np.array(
                [
                    [200.0, 202.0, 204.0, 206.0],
                    [208.0, 210.0, 212.0, 214.0],
                    [216.0, 218.0, 220.0, 222.0],
                ]
            )
            input_grad = np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [8788.0, 10976.0, 13500.0, 16384.0],
                        [19652.0, 23328.0, 27436.0, 32000.0],
                        [37044.0, 42592.0, 48668.0, 55296.0],
                    ],
                ]
            )
            np.testing.assert_array_equal(
                inps.grad.numpy(),
                input_grad,
                err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
            )
            np.testing.assert_array_equal(
                value.grad.numpy(),
                value_grad,
                err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
            )

            # case 6: pass stop_gradient from value to x
            x = paddle.zeros([8, 8], dtype='float32')
            value = paddle.to_tensor([10], dtype='float32', stop_gradient=False)

            self.assertTrue(x.stop_gradient)
            self.assertTrue(x.is_leaf)

            x[0, :] = value

            self.assertTrue(not x.stop_gradient)
            self.assertTrue(not x.is_leaf)

        def test_static_graph(self):
            paddle.enable_static()
            with paddle.pir_utils.OldIrGuard():
                to_string = lambda x, i: x + '_' + str(i)
                numel = lambda input_shape: reduce(
                    lambda x, y: x * y, input_shape, 1
                )

                def op1(x):
                    value = paddle.tensor.fill_constant([1], "float32", 1)
                    # test stop_gradient
                    value.stop_gradient = True
                    x.stop_gradient = False
                    start = paddle.tensor.fill_constant(
                        [1], "int32", 5, force_cpu=True
                    )
                    end = paddle.tensor.fill_constant(
                        [1], "int32", 0, force_cpu=True
                    )
                    step = paddle.tensor.fill_constant(
                        [1], "int32", -2, force_cpu=True
                    )

                    inputs = {
                        'Input': x,
                        'ValueTensor': value,
                        'StartsTensorList': [
                            start,
                        ],
                        'EndsTensorList': [
                            end,
                        ],
                        'StepsTensorList': [
                            step,
                        ],
                    }

                    helper = LayerHelper("set_value")
                    y = helper.create_variable_for_type_inference(dtype=x.dtype)

                    helper.append_op(
                        type="set_value",
                        inputs=inputs,
                        outputs={'Out': y},
                        attrs={'axes': [0]},
                    )

                    return y, value

                def op2(x):
                    value = paddle.tensor.fill_constant([1, 3, 2], "float32", 1)
                    # test stop_gradient
                    value.stop_gradient = False
                    x.stop_gradient = False
                    attrs = {
                        'axes': [0],
                        'starts': [6],
                        'ends': [0],
                        'steps': [-4],
                        'decrease_axes': [],
                        'none_axes': [],
                        'dtype': paddle.float32,
                    }
                    inputs = {'Input': x, 'ValueTensor': value}

                    helper = LayerHelper("set_value")
                    y = helper.create_variable_for_type_inference(dtype=x.dtype)

                    helper.append_op(
                        type="set_value",
                        inputs=inputs,
                        outputs={'Out': y},
                        attrs=attrs,
                    )

                    return y, value

                def op3(x):
                    value = paddle.tensor.fill_constant([1], "float32", 1)
                    x.stop_gradient = True
                    value.stop_gradient = False
                    start = paddle.tensor.fill_constant(
                        [1], "int32", 0, force_cpu=True
                    )
                    end = paddle.tensor.fill_constant(
                        [1], "int32", 5, force_cpu=True
                    )
                    step = paddle.tensor.fill_constant(
                        [1], "int32", 3, force_cpu=True
                    )

                    inputs = {
                        'Input': x,
                        'ValueTensor': value,
                        'StartsTensorList': [
                            start,
                        ],
                        'EndsTensorList': [
                            end,
                        ],
                        'StepsTensorList': [
                            step,
                        ],
                    }

                    helper = LayerHelper("set_value")
                    y = helper.create_variable_for_type_inference(dtype=x.dtype)

                    helper.append_op(
                        type="set_value",
                        inputs=inputs,
                        outputs={'Out': y},
                        attrs={'axes': [0]},
                    )

                    return y, value

                def set_value(array, i, op):
                    name_x = to_string('x', i)
                    x = paddle.static.data(
                        name=name_x, shape=array.shape, dtype='float32'
                    )

                    # set_value_op in __get/setitem__ is an inplace operation.
                    # When `input.stop_gradient = True` and `value.stop_gradient = False`,
                    # set_value_grad_op will not be run during backward.
                    y, value = op(x)
                    y2 = y + 1
                    loss = paddle.sum(y2)
                    sgd = paddle.optimizer.Adam()
                    sgd.minimize(loss)
                    place = self.place

                    prog = paddle.static.default_main_program()
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    fetch_list = []
                    if not x.stop_gradient:
                        fetch_list.append(x.grad_name)
                    if not value.stop_gradient:
                        fetch_list.append(value.grad_name)
                    out = exe.run(
                        prog, feed={x.name: array}, fetch_list=fetch_list
                    )
                    return out

                input_shape = [7, 6, 5, 4, 3, 2]

                array = np.arange(
                    0, numel(input_shape), dtype="float32"
                ).reshape(input_shape)

                for i in range(len(input_shape)):
                    program = paddle.static.Program()
                    with paddle.static.program_guard(program):
                        out1 = set_value(array, i, op1)
                        self.assertTrue((out1[0][5:0:-2] == 0).all())

                    if len(array.shape) > 2:
                        program2 = paddle.static.Program()
                        with paddle.static.program_guard(program2):
                            out2 = set_value(array, i, op2)
                            self.assertTrue((out2[0][6:0:-4] == 0).all())

                    program3 = paddle.static.Program()
                    with paddle.static.program_guard(program3):
                        out3 = set_value(array, i, op3)
                        self.assertTrue(
                            (numel(out1[0][0:5:3].shape) == out3[0]).all()
                        )

                    array = array[0]
            paddle.disable_static()

    class XPUTestSetValueInplace(XPUOpTest):
        def setUp(self):
            self.__class__.op_type = "set_value"
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)

        def test_inplace(self):
            paddle.disable_static()
            with paddle.base.dygraph.guard():
                paddle.seed(100)
                a = paddle.rand(shape=[1, 4])
                a.stop_gradient = False
                b = a[:] * 1
                c = b
                b[paddle.zeros([], dtype='int32')] = 1.0

                self.assertTrue(id(b) == id(c))
                np.testing.assert_array_equal(b.numpy(), c.numpy())
                self.assertEqual(b.inplace_version, 1)

            paddle.enable_static()

    class XPUTestSetValueInplaceLeafVar(XPUOpTest):
        def setUp(self):
            self.__class__.op_type = "set_value"
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)

        def test_inplace_var_become_leaf_var(self):
            paddle.disable_static()

            a_grad_1, b_grad_1, a_grad_2, b_grad_2 = 0, 1, 2, 3
            with paddle.base.dygraph.guard():
                paddle.seed(100)
                a = paddle.rand(shape=[1, 4])
                b = paddle.rand(shape=[1, 4])
                a.stop_gradient = False
                b.stop_gradient = False
                c = a / b
                c.sum().backward()
                a_grad_1 = a.grad.numpy()
                b_grad_1 = b.grad.numpy()

            with paddle.base.dygraph.guard():
                paddle.seed(100)
                a = paddle.rand(shape=[1, 4])
                b = paddle.rand(shape=[1, 4])
                a.stop_gradient = False
                b.stop_gradient = False
                c = a / b
                d = paddle.zeros((4, 4))
                self.assertTrue(d.stop_gradient)
                d[0, :] = c
                self.assertFalse(d.stop_gradient)
                d[0, :].sum().backward()
                a_grad_2 = a.grad.numpy()
                b_grad_2 = b.grad.numpy()

            np.testing.assert_array_equal(a_grad_1, a_grad_2)
            np.testing.assert_array_equal(b_grad_1, b_grad_2)
            paddle.enable_static()


support_types = get_xpu_op_support_types('set_value')
for stype in support_types:
    create_test_class(globals(), XPUTestSetValueOp, stype)

if __name__ == '__main__':
    unittest.main()
