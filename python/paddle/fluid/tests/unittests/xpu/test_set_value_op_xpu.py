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

import numpy as np

# from functools import reduce


sys.path.append("../")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle

# from paddle.fluid.layer_helper import LayerHelper


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
            self.data = np.ones(self.shape).astype(self.dtype)
            self.program = paddle.static.Program()

        def set_shape(self):
            self.shape = [2, 3, 4]

        def set_value(self):
            self.value = 6

        def set_dtype(self):
            self.dtype = self.in_type

        def _call_setitem(self, x):
            x[0, 0] = self.value

        def _get_answer(self):
            self.data[0, 0] = self.value

    class XPUTestSetValueApi(XPUTestSetValueBase):
        def _run_static(self):
            paddle.enable_static()
            with paddle.static.program_guard(self.program):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                self._call_setitem(x)

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

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueItemInt2(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [6, 6, 6, 6, 6]

        def _call_setitem(self, x):
            x[0, 3, 4] = self.value

        def _get_answer(self):
            self.data[0, 3, 4] = self.value

    class XPUTestSetValueItemInt3(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [6, 6, 6, 6, 6]

        def _call_setitem(self, x):
            x[1] = self.value

        def _get_answer(self):
            self.data[1] = self.value

    # 1.2 item is slice
    # 1.2.1 step is 1
    class XPUTestSetValueItemSlice(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:2] = self.value

        def _get_answer(self):
            self.data[0:2] = self.value

    class XPUTestSetValueItemSlice2(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:-1] = self.value

        def _get_answer(self):
            self.data[0:-1] = self.value

    class XPUTestSetValueItemSlice3(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:-1, 0:2] = self.value

        def _get_answer(self):
            self.data[0:-1, 0:2] = self.value

    class XPUTestSetValueItemSlice4(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, 1:2, :] = self.value

        def _get_answer(self):
            self.data[0:, 1:2, :] = self.value

    class XPUTestSetValueItemSlice5(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, 1:1, :] = self.value

        def _get_answer(self):
            self.data[0:, 1:1, :] = self.value

    class XPUTestSetValueItemSliceInWhile(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            def cond(i, x):
                return i < 1

            def body(i, x):
                x[i] = self.value
                i = i + 1
                return i, x

            i = paddle.zeros(shape=(1,), dtype='int32')
            i, x = paddle.static.nn.while_loop(cond, body, [i, x])

        def _get_answer(self):
            self.data[0] = self.value

    # 1.2.2 step > 1
    class XPUTestSetValueItemSliceStep(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [5, 5, 5]

        def _call_setitem(self, x):
            x[0:2:2] = self.value

        def _get_answer(self):
            self.data[0:2:2] = self.value

    class XPUTestSetValueItemSliceStep2(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [7, 5, 5]

        def _call_setitem(self, x):
            x[0:-1:3] = self.value

        def _get_answer(self):
            self.data[0:-1:3] = self.value

    class XPUTestSetValueItemSliceStep3(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:-1, 0:2, ::2] = self.value

        def _get_answer(self):
            self.data[0:-1, 0:2, ::2] = self.value

    class XPUTestSetValueItemSliceStep4(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, 1:2:2, :] = self.value

        def _get_answer(self):
            self.data[0:, 1:2:2, :] = self.value

    # 1.2.3 step < 0
    class XPUTestSetValueItemSliceNegetiveStep(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_shape(self):
            self.shape = [5, 2]

        def set_value(self):
            self.value = np.array([3, 4])

        def _call_setitem(self, x):
            x[5:2:-1] = self.value

        def _get_answer(self):
            self.data[5:2:-1] = self.value

    class XPUTestSetValueItemSliceNegetiveStep2(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_shape(self):
            self.shape = [5]

        def set_value(self):
            self.value = np.array([3, 4])
            # print("self.value: \n", self.value)

        def _call_setitem(self, x):
            x[1::-1] = self.value

        def _get_answer(self):
            self.data[1::-1] = self.value

    class XPUTestSetValueItemSliceNegetiveStep3(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_shape(self):
            self.shape = [3]

        def set_value(self):
            self.value = np.array([3, 4, 5])

        def _call_setitem(self, x):
            x[::-1] = self.value

        def _get_answer(self):
            self.data[::-1] = self.value

    class XPUTestSetValueItemSliceNegetiveStep4(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [3, 4, 5]

        def _call_setitem(self, x):
            x[2:0:-1, 0:2, ::-1] = self.value

        def _get_answer(self):
            self.data[2:0:-1, 0:2, ::-1] = self.value

        # 1.2.3 step < 0 and stride < -1

    class XPUTestSetValueItemSliceNegetiveStep5(XPUTestSetValueApi):
        def set_shape(self):
            self.shape = [5, 5, 5]

        def _call_setitem(self, x):
            x[2:-1:-2] = self.value

        def _get_answer(self):
            paddle.enable_static()
            with paddle.static.program_guard(self.program):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                self._call_setitem(x)

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

        def _get_answer(self):
            self.data[0:, ..., 1:] = self.value

    class XPUTestSetValueItemEllipsis2(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[0:, ...] = self.value

        def _get_answer(self):
            self.data[0:, ...] = self.value

    class XPUTestSetValueItemEllipsis3(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[..., 1:] = self.value

        def _get_answer(self):
            self.data[..., 1:] = self.value

    class XPUTestSetValueItemEllipsis4(XPUTestSetValueApi):
        def _call_setitem(self, x):
            x[...] = self.value

        def _get_answer(self):
            self.data[...] = self.value

    # 1.4 item is Paddle Tensor
    class XPUTestSetValueItemTensor(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            zero = paddle.full([1], 0, dtype="int32")
            x[zero] = self.value

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueItemTensor2(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            zero = paddle.full([1], 0, dtype="int32")
            two = paddle.full([1], 2, dtype="int64")
            x[zero:two] = self.value

        def _get_answer(self):
            self.data[0:2] = self.value

    class XPUTestSetValueItemTensor3(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            zero = paddle.full([1], 0, dtype="int32")
            two = paddle.full([1], 2, dtype="int64")
            x[zero:-1, 0:two] = self.value

        def _get_answer(self):
            self.data[0:-1, 0:2] = self.value

    class XPUTestSetValueItemTensor4(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            zero = paddle.full([1], 0, dtype="int32")
            two = paddle.full([1], 2, dtype="int64")
            x[0:-1, zero:2, 0:6:two] = self.value

        def _get_answer(self):
            self.data[0:-1, 0:2, ::2] = self.value

    class XPUTestSetValueItemTensor5(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            zero = paddle.full([1], 0, dtype="int32")
            two = paddle.full([1], 2, dtype="int64")
            x[zero:, 1:2:two, :] = self.value

        def _get_answer(self):
            self.data[0:, 1:2:2, :] = self.value

    class XPUTestSetValueItemTensor6(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_shape(self):
            self.shape = [3, 4, 5]

        def _call_setitem(self, x):
            minus1 = paddle.full([1], -1, dtype="int32")
            zero = paddle.full([1], 0, dtype="int32")
            x[2:zero:minus1, 0:2, 10:-6:minus1] = self.value

        def _get_answer(self):
            self.data[2:0:-1, 0:2, ::-1] = self.value

    # 1.5 item is None
    class XPUTestSetValueItemNone1(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[None] = self.value

        def _get_answer(self):
            self.data[None] = self.value

    class XPUTestSetValueItemNone2(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[0, None, 1] = self.value

        def _get_answer(self):
            self.data[0, None, 1] = self.value

    class XPUTestSetValueItemNone3(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[:, None, None, 1] = self.value

        def _get_answer(self):
            self.data[:, None, None, 1] = self.value

    class XPUTestSetValueItemNone4(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[0, 0, None, 1] = self.value

        def _get_answer(self):
            self.data[0, 0, None, 1] = self.value

    class XPUTestSetValueItemNone5(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[0, None, 0, None, 1] = self.value

        def _get_answer(self):
            self.data[0, None, 0, None, 1] = self.value

    class XPUTestSetValueItemNone6(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[None, 0, 0, None, 0] = self.value

        def _get_answer(self):
            self.data[None, 0, 0, None, 0] = self.value

    class XPUTestSetValueItemNone7(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[:, None, 1] = np.zeros(self.shape)[:, None, 0]

        def _get_answer(self):
            self.data[:, None, 1] = np.zeros(self.shape)[:, None, 0]

    class XPUTestSetValueItemNone8(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[:, 1, None] = np.zeros(self.shape)[:, 0, None]

        def _get_answer(self):
            self.data[:, 1, None] = np.zeros(self.shape)[:, 0, None]

    class XPUTestSetValueItemNone9(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]

        def _get_answer(self):
            self.data[None, :, 1, ..., None] = np.zeros(self.shape)[
                0, 0, :, None
            ]

    class XPUTestSetValueItemNone10(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]

        def _get_answer(self):
            self.data[..., None, :, None] = np.zeros(self.shape)[
                ..., None, :, None
            ]

    # 1.6 item is list or Tensor of bol
    class XPUTestSetValueItemBool1(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[[True, False]] = self.value

        def _get_answer(self):
            self.data[[True, False]] = self.value

    class XPUTestSetValueItemBool2(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[[False, False]] = self.value

        def _get_answer(self):
            self.data[[False, False]] = self.value

    class XPUTestSetValueItemBool3(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[[False, True]] = np.zeros(self.shape[2])

        def _get_answer(self):
            self.data[[False, True]] = np.zeros(self.shape[2])

    class XPUTestSetValueItemBool4(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            idx = paddle.assign(np.array([False, True]))
            x[idx] = np.zeros(self.shape[2])

        def _get_answer(self):
            self.data[np.array([False, True])] = np.zeros(self.shape[2])

    class XPUTestSetValueItemBool5(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            idx = paddle.assign(
                np.array([[False, True, False], [True, True, False]])
            )
            x[idx] = self.value

        def _get_answer(self):
            self.data[
                np.array([[False, True, False], [True, True, False]])
            ] = self.value

    class XPUTestSetValueItemBool6(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def _call_setitem(self, x):
            x[0, ...] = 0
            x[x > 0] = self.value

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

            def _get_answer(self):
                self.data[0, 1] = 3

        cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorInt32")
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

            def _get_answer(self):
                self.data[0, 1] = 3

        cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorInt64")
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

            def _get_answer(self):
                self.data[0, 1] = 3

        cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorFp32")
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

            def _get_answer(self):
                self.data[0, 1] = False

        cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorBool")
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
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_value(self):
            self.value = np.array([3, 4, 5, 6])  # shape is (4,)

        def _call_setitem(self, x):
            x[0] = self.value

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueValueShape2(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_value(self):
            self.value = np.array([[3, 4, 5, 6]])  # shape is (1,4)

        def _call_setitem(self, x):
            x[0:1] = self.value

        def _get_answer(self):
            self.data[0:1] = self.value

    class XPUTestSetValueValueShape3(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_value(self):
            self.value = np.array(
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
            )  # shape is (3,4)

        def _call_setitem(self, x):
            x[0] = self.value

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueValueShape4(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_value(self):
            self.value = np.array(
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
            ).astype(
                self.dtype
            )  # shape is (3,4)

        def _call_setitem(self, x):
            x[0] = paddle.assign(self.value)  # x is Paddle.Tensor

        def _get_answer(self):
            self.data[0] = self.value

    class XPUTestSetValueValueShape5(XPUTestSetValueApi):
        def set_dtype(self):
            if self.in_type == np.float16:
                self.dtype = "float32"
            else:
                self.dtype = self.in_type

        def set_value(self):
            self.value = np.array([3, 3, 3]).astype(self.dtype)

        def set_shape(self):
            self.shape = [3, 4]

        def _call_setitem(self, x):
            x[:, 0] = paddle.assign(self.value)  # x is Paddle.Tensor

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
                x[0] = value

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
                x[0:1:0] = self.value

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
            with self.assertRaises(TypeError):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                x[[True, False, 0]] = 0

            with self.assertRaises(IndexError):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                x[[True, False], [True, False]] = 0

        def _bool_tensor_error(self):
            with self.assertRaises(IndexError):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                idx = paddle.assign([True, False, True])
                x[idx] = 0

        def _broadcast_mismatch(self):
            program = paddle.static.Program()
            with paddle.static.program_guard(program):
                x = paddle.ones(shape=self.shape, dtype=self.dtype)
                value = np.array([3, 4, 5, 6, 7])
                x[0] = value
            exe = paddle.static.Executor(paddle.XPUPlace(0))
            with self.assertRaises(ValueError):
                exe.run(program)

        def test_error(self):
            paddle.enable_static()
            with paddle.static.program_guard(self.program):
                self._value_type_error()
                self._step_error()
                self._bool_list_error()
                self._bool_tensor_error()
            self._broadcast_mismatch()


support_types = get_xpu_op_support_types('set_value')
for stype in support_types:
    create_test_class(globals(), XPUTestSetValueOp, stype)

if __name__ == '__main__':
    unittest.main()
