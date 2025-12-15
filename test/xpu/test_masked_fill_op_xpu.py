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

import struct
import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()


def convert_float_to_uint16(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack('<I', struct.pack('<f', x))[0] >> 16,
        otypes=[np.uint16],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


def np_masked_fill(x, mask, value):
    v = value
    x_b, mask_b = np.broadcast_arrays(x, mask)
    v_b = np.broadcast_to(v, x_b.shape)

    out = np.where(mask_b, v_b, x_b)
    return out


class XPUTestMaskedFillOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'masked_fill'

    class TestMaskedFillBase(XPUOpTest):
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
            self.check_grad_with_place(self.place, ['x'], 'out')

        def init(self):
            self.x_shape = (16, 32)
            self.mask_shape = self.x_shape
            self.value_shape = self.x_shape

    class TestMaskedFillTensorValue1D(TestMaskedFillBase):
        def init(self):
            self.x_shape = (64,)
            self.mask_shape = self.x_shape
            self.value_shape = self.x_shape

    class TestMaskedFillTensorValue4D(TestMaskedFillBase):
        def init(self):
            self.x_shape = (2, 3, 4, 5)
            self.mask_shape = self.x_shape
            self.value_shape = self.x_shape

    class TestMaskedFillTensorValueBroadcastMask(TestMaskedFillBase):
        def init(self):
            self.x_shape = (10, 4, 5)
            self.mask_shape = (
                1,
                4,
                1,
            )
            self.value_shape = (10, 4, 5)

    class TestMaskedFillTensorValueBroadcastValue(TestMaskedFillBase):
        def init(self):
            self.x_shape = (2, 4, 5)
            self.mask_shape = (2, 4, 5)
            self.value_shape = (1, 5)

    # ------------------ Scalar Value  ------------------
    class TestMaskedFillScalarValue4D(TestMaskedFillBase):
        def init(self):
            self.x_shape = (2, 3, 4, 5)
            self.mask_shape = self.x_shape

        def init_config(self):
            super().init_config()
            self.is_scalar_value = True

    class TestMaskedFillScalarValueBroadcastMask(TestMaskedFillBase):
        def init(self):
            self.x_shape = (10, 4, 5)
            self.mask_shape = (1, 4, 1)

        def init_config(self):
            super().init_config()
            self.is_scalar_value = True

    class TestMaskedFillScalarValueBroadcastMask2(TestMaskedFillBase):
        def init(self):
            self.x_shape = (10, 1)
            self.mask_shape = (1, 5)

        def init_config(self):
            super().init_config()
            self.is_scalar_value = True


support_types = get_xpu_op_support_types('masked_fill')
for stype in support_types:
    create_test_class(globals(), XPUTestMaskedFillOp, stype)


class TestMaskedFillAPIXPU(unittest.TestCase):
    def setUp(self):
        if not core.is_compiled_with_xpu():
            self.skipTest("XPU is not available")

    def run_api_test(self, mode, x_shape, mask_shape, value_type):
        x_np = np.random.randn(*x_shape).astype('float32')
        mask_np = np.random.randint(0, 2, size=mask_shape).astype('bool')

        if mode == 'dygraph':
            paddle.disable_static()
            paddle.set_device("xpu")
            x = paddle.to_tensor(x_np)
            mask = paddle.to_tensor(mask_np, dtype='bool')

            if value_type == 'scalar':
                scalar = float(np.random.randn())
                value = scalar
                expect = np_masked_fill(x_np, mask_np, scalar)
            else:  # tensor value
                value_np = np.random.randn(*value_type).astype('float32')
                value = paddle.to_tensor(value_np)
                expect = np_masked_fill(x_np, mask_np, value_np)

            out = paddle.masked_fill(x, mask, value)
            res = out.numpy()
            paddle.enable_static()

        else:  # static graph
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            main_prog = base.Program()
            startup_prog = base.Program()

            with base.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=x_shape, dtype='float32')
                mask = paddle.static.data(
                    name='mask', shape=mask_shape, dtype='bool'
                )

                if value_type == 'scalar':
                    scalar = float(np.random.randn())
                    value = paddle.full([], scalar, dtype='float32')
                    expect = np_masked_fill(x_np, mask_np, scalar)
                else:  # tensor value
                    value_np = np.random.randn(*value_type).astype('float32')
                    value = paddle.static.data(
                        name='value', shape=value_type, dtype='float32'
                    )
                    expect = np_masked_fill(x_np, mask_np, value_np)

                out = paddle.masked_fill(x, mask, value)

                exe = base.Executor(place)
                exe.run(startup_prog)
                feed_dict = {'x': x_np, 'mask': mask_np}
                if value_type != 'scalar':
                    feed_dict['value'] = value_np

                (res,) = exe.run(
                    main_prog,
                    feed=feed_dict,
                    fetch_list=[out],
                )

        np.testing.assert_allclose(res, expect, rtol=1e-5, atol=1e-5)

    def test_dygraph_scalar_value_broadcast_mask(self):
        self.run_api_test(
            'dygraph', x_shape=(10, 5), mask_shape=(10, 1), value_type='scalar'
        )

    def test_dygraph_scalar_value_1d(self):
        self.run_api_test(
            'dygraph', x_shape=(20,), mask_shape=(20,), value_type='scalar'
        )

    def test_dygraph_tensor_value_no_broadcast(self):
        self.run_api_test(
            'dygraph', x_shape=(5, 5), mask_shape=(5, 5), value_type=(5, 5)
        )

    def test_static_graph_tensor_value_broadcast(self):
        self.run_api_test(
            'static', x_shape=(4, 5), mask_shape=(4, 1), value_type=(4, 5)
        )

    def test_static_graph_scalar_value_large_dim(self):
        self.run_api_test(
            'static',
            x_shape=(2, 3, 4, 5),
            mask_shape=(2, 3, 4, 5),
            value_type='scalar',
        )


if __name__ == '__main__':
    unittest.main()
