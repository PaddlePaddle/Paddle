# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def np_masked_fill(x, mask, value):
    x_b, mask_b = np.broadcast_arrays(x, mask)
    v_b = np.broadcast_to(value, x_b.shape)
    out = np.where(mask_b, v_b, x_b)
    return out


def _np_reduce_to_shape(x, target_shape, out_dtype):
    if x.shape == tuple(target_shape):
        return x.astype(out_dtype, copy=False)

    target_shape = tuple(target_shape)
    target_shape_padded = (1,) * (x.ndim - len(target_shape)) + target_shape
    reduce_axes = [
        axis
        for axis, (t_dim, x_dim) in enumerate(zip(target_shape_padded, x.shape))
        if t_dim == 1 and x_dim != 1
    ]
    if reduce_axes:
        x = x.sum(axis=tuple(reduce_axes), keepdims=True)
    x = x.reshape(target_shape)
    return x.astype(out_dtype, copy=False)


def np_masked_fill_grad(x, mask, value, out_grad):
    out_shape = np.broadcast(np.empty(x.shape), np.empty(mask.shape)).shape
    x_b, mask_b = np.broadcast_arrays(x, mask)
    out_grad_b = np.broadcast_to(out_grad, out_shape)
    dx_full = np.where(mask_b, np.zeros_like(out_grad_b), out_grad_b)
    dv_full = np.where(mask_b, out_grad_b, np.zeros_like(out_grad_b))
    dx = _np_reduce_to_shape(dx_full, x.shape, x.dtype)
    dv = _np_reduce_to_shape(dv_full, value.shape, value.dtype)
    return dx, dv


class XPUTestMaskedFillGradOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'masked_fill'

    class TestMaskedFillGradBase(XPUOpTest):
        def setUp(self):
            self.init()
            self.init_config()
            self.init_data()

            self.inputs = {
                'x': self.x,
                'mask': self.mask,
                'value': self.value,
            }
            self.attrs = {}
            self.outputs = {'out': self.out}

        def init_config(self):
            self.op_type = "masked_fill"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.__class__.no_need_check_grad = False
            self.is_scalar_value = getattr(self, 'is_scalar_value', False)
            self.mask_shape = getattr(self, 'mask_shape', self.x_shape)
            self.value_shape = getattr(self, 'value_shape', self.x_shape)

        def init_data(self):
            self.mask = np.random.randint(0, 2, size=self.mask_shape).astype(
                'bool'
            )

            if self.dtype == np.uint16:
                x_fp32 = np.random.randn(*self.x_shape).astype('float32')
                if x_fp32.size == 0:
                    self.x = x_fp32.astype(np.uint16)
                else:
                    self.x = convert_float_to_uint16(x_fp32)

                if self.is_scalar_value:
                    scalar_fp32 = float(np.random.randn())
                    scalar_arr = np.array([scalar_fp32], dtype='float32')
                    self.value = convert_float_to_uint16(scalar_arr)
                    v_np = scalar_fp32
                else:
                    v_fp32 = np.random.randn(*self.value_shape).astype(
                        'float32'
                    )
                    if v_fp32.size == 0:
                        self.value = v_fp32.astype(np.uint16)
                    else:
                        self.value = convert_float_to_uint16(v_fp32)
                    v_np = v_fp32

                self.out = np_masked_fill(
                    x_fp32,
                    self.mask,
                    v_np,
                ).astype(x_fp32.dtype)
            else:
                self.x = np.random.randn(*self.x_shape).astype(self.dtype)

                if self.is_scalar_value:
                    scalar = float(np.random.randn())
                    self.value = np.array([scalar]).astype(self.dtype)
                    v_np = scalar
                else:
                    v_np = np.random.randn(*self.value_shape).astype(self.dtype)
                    self.value = v_np

                self.out = np_masked_fill(self.x, self.mask, v_np).astype(
                    self.dtype
                )

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            # Only floating types support numeric grad check in OpTest.
            if self.dtype not in [np.float32, np.float16, np.uint16]:
                self.__class__.no_need_check_grad = True
                return
            self.check_grad_with_place(self.place, ['x'], 'out')

        def init(self):
            self.x_shape = (16, 32)
            self.mask_shape = self.x_shape
            self.value_shape = self.x_shape

    # ------------------ Tensor Value ------------------
    class TestMaskedFillGradTensorValue1D(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (64,)
            self.mask_shape = self.x_shape
            self.value_shape = self.x_shape

    class TestMaskedFillGradTensorValue4D(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (2, 3, 4, 5)
            self.mask_shape = self.x_shape
            self.value_shape = self.x_shape

    class TestMaskedFillGradTensorValueBroadcastMask(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (10, 4, 5)
            self.mask_shape = (1, 4, 1)
            self.value_shape = (10, 4, 5)

    class TestMaskedFillGradTensorValueBroadcastValue(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (2, 4, 5)
            self.mask_shape = (2, 4, 5)
            self.value_shape = (1, 5)

    # ------------------ Scalar Value ------------------
    class TestMaskedFillGradScalarValue4D(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (2, 3, 4, 5)
            self.mask_shape = self.x_shape

        def init_config(self):
            super().init_config()
            self.is_scalar_value = True

    class TestMaskedFillGradScalarValueBroadcastMask(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (10, 4, 5)
            self.mask_shape = (1, 4, 1)

        def init_config(self):
            super().init_config()
            self.is_scalar_value = True

    class TestMaskedFillGradScalarValueBroadcastMask2(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (10, 1)
            self.mask_shape = (1, 5)

        def init_config(self):
            super().init_config()
            self.is_scalar_value = True

    # ------------------ Empty Tensor ------------------
    class TestMaskedFillGradEmptyX(TestMaskedFillGradBase):
        """Test empty x tensor (numel == 0)"""

        def init(self):
            self.x_shape = (0, 5)
            self.mask_shape = (0, 5)
            self.value_shape = (0, 5)

        def test_check_grad(self):
            # Skip grad check for empty tensor
            pass

    class TestMaskedFillGradEmptyMask(TestMaskedFillGradBase):
        """Test empty mask tensor (numel == 0)"""

        def init(self):
            self.x_shape = (5, 0)
            self.mask_shape = (5, 0)
            self.value_shape = (5, 0)

        def test_check_grad(self):
            # Skip grad check for empty tensor
            pass

    # ------------------ Both Mask and Value Broadcast ------------------
    class TestMaskedFillGradBothBroadcast(TestMaskedFillGradBase):
        """Test both mask and value need broadcast"""

        def init(self):
            self.x_shape = (4, 5, 6)
            self.mask_shape = (1, 5, 1)
            self.value_shape = (4, 1, 6)

    # ------------------ Large Tensor ------------------
    class TestMaskedFillGradLargeTensor(TestMaskedFillGradBase):
        """Test with larger tensor to ensure kernel handles large data"""

        def init(self):
            self.x_shape = (128, 256)
            self.mask_shape = self.x_shape
            self.value_shape = self.x_shape

    # ------------------ Broadcast X ------------------
    class TestMaskedFillGradBroadcastX(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (1, 5)
            self.mask_shape = (10, 5)
            self.value_shape = self.x_shape

    # ------------------ Broadcast X and Value ------------------
    class TestMaskedFillGradBroadcastXAndValue(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (1, 5)
            self.mask_shape = (10, 5)
            self.value_shape = (1, 1)

    # ------------------ Complex Broadcast ------------------
    class TestMaskedFillGradComplexBroadcast(TestMaskedFillGradBase):
        def init(self):
            self.x_shape = (10, 1, 5)
            self.mask_shape = (1, 4, 5)
            self.value_shape = (10, 1, 1)


support_types = get_xpu_op_support_types('masked_fill')
for stype in support_types:
    create_test_class(globals(), XPUTestMaskedFillGradOp, stype)


if __name__ == '__main__':
    unittest.main()
