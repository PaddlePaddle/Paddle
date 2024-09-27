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

import itertools
import os
import unittest

import numpy as np

import paddle
from paddle.framework import core

paddle.enable_static()


def numpy_ref(_x, value, axes, starts, ends, strides):
    x = np.copy(_x)

    try:
        value = np.broadcast_to(value, x.shape)
    except:
        pass

    indices_x = []
    indices_v = []
    for ndim_idx in range(x.ndim):
        if ndim_idx not in axes:
            ind = list(range(x.shape[ndim_idx]))
            indices_x.append(ind)
            indices_v.append(ind)
        else:
            _idx = list(axes).index(ndim_idx)
            ind_x = list(range(starts[_idx], ends[_idx], strides[_idx]))
            ind_v = list(range(len(ind_x)))
            indices_x.append(ind_x)
            indices_v.append(ind_v)

    for index_x, index_v in zip(
        itertools.product(*indices_x), itertools.product(*indices_v)
    ):
        x[index_x] = value[index_v]

    return x


class TestSliceScatterApi(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)

        self.init_shape()

        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def init_np(self):
        self.x_np = np.random.random(self.x_shape).astype(
            'uint16' if self.dtype == 'bfloat16' else self.dtype
        )
        self.value_np = np.random.random(self.value_shape).astype(
            'uint16' if self.dtype == 'bfloat16' else self.dtype
        )

    def init_dtype(self):
        self.dtype = 'float64'

    def init_shape(self):
        self.x_shape = [8, 6]
        self.value_shape = [8, 2]
        self.axes = [1]
        self.starts = [2]
        self.ends = [6]
        self.strides = [2]

    def test_api_static(self):
        paddle.enable_static()
        self.init_dtype()
        self.init_np()

        for place in self.place:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('x', self.x_shape, self.dtype)
                value = paddle.static.data(
                    'value', self.value_shape, self.dtype
                )

                out = paddle.slice_scatter(
                    x,
                    value,
                    axes=self.axes,
                    starts=self.starts,
                    ends=self.ends,
                    strides=self.strides,
                )
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={
                        'x': self.x_np,
                        'value': self.value_np,
                    },
                    fetch_list=[out],
                )[0]

            out_ref = numpy_ref(
                self.x_np,
                self.value_np,
                axes=self.axes,
                starts=self.starts,
                ends=self.ends,
                strides=self.strides,
            )

            np.testing.assert_allclose(res, out_ref)

    def test_api_dygraph(self):
        self.init_dtype()
        self.init_np()
        for place in self.place:
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.slice_scatter(
                x_tensor,
                value_tensor,
                axes=self.axes,
                starts=self.starts,
                ends=self.ends,
                strides=self.strides,
            )
            out_ref = numpy_ref(
                self.x_np,
                self.value_np,
                axes=self.axes,
                starts=self.starts,
                ends=self.ends,
                strides=self.strides,
            )

            np.testing.assert_allclose(out.numpy(), out_ref)

            paddle.enable_static()


class TestSliceScatterApiIntComplex128(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'complex128'


class TestSliceScatterApiIntComplex64(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'complex64'


class TestSliceScatterApiInt64(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'int64'


class TestSliceScatterApiInt32(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'int32'


class TestSliceScatterApiInt16(TestSliceScatterApi):
    def init_dtype(self):
        # old ir `set_value` not support this dtype
        if paddle.framework.in_dynamic_or_pir_mode():
            self.dtype = 'int16'
        else:
            self.dtype = 'float64'


class TestSliceScatterApiInt8(TestSliceScatterApi):
    def init_dtype(self):
        # old ir `set_value` not support this dtype
        if paddle.framework.in_dynamic_or_pir_mode():
            self.dtype = 'int8'
        else:
            self.dtype = 'float64'


class TestSliceScatterApiUint8(TestSliceScatterApi):
    def init_dtype(self):
        # old ir `set_value` not support this dtype
        if paddle.framework.in_dynamic_or_pir_mode():
            self.dtype = 'uint8'
        else:
            self.dtype = 'float64'


class TestSliceScatterApiBool(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'bool'


class TestSliceScatterApiBfloat16(TestSliceScatterApi):
    def init_dtype(self):
        # old ir `set_value` not support this dtype
        if paddle.framework.in_dynamic_or_pir_mode():
            self.dtype = 'bfloat16'
        else:
            self.dtype = 'float64'


class TestSliceScatterApiFloat16(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'float16'


class TestSliceScatterApiFloat32(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApi3D(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 6, 3]
        self.value_shape = [8, 2, 3]
        self.axes = [1]
        self.starts = [2]
        self.ends = [6]
        self.strides = [2]


class TestSliceScatterApi3DFloat32(TestSliceScatterApi3D):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApi4D(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 6, 3, 5]
        self.value_shape = [8, 2, 3, 5]
        self.axes = [1]
        self.starts = [2]
        self.ends = [6]
        self.strides = [2]


class TestSliceScatterApi4DFloat32(TestSliceScatterApi4D):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApi4DAxis3(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 6, 3, 9]
        self.value_shape = [8, 6, 3, 2]
        self.axes = [3]
        self.starts = [2]
        self.ends = [6]
        self.strides = [2]


class TestSliceScatterApi4DAxis3Float32(TestSliceScatterApi4DAxis3):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApiBroadcast2D(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 9]
        self.value_shape = [8, 1]
        self.axes = [1]
        self.starts = [2]
        self.ends = [6]
        self.strides = [2]


class TestSliceScatterApiBroadcast2DFloat32(TestSliceScatterApiBroadcast2D):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApiBroadcast3D(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 9, 6]
        self.value_shape = [1, 9, 1]
        self.axes = [0, 2]
        self.starts = [2, 3]
        self.ends = [7, 5]
        self.strides = [3, 2]


class TestSliceScatterApiBroadcast3DFloat32(TestSliceScatterApiBroadcast3D):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterTensorApi(unittest.TestCase):
    def test_tensor(self):
        paddle.disable_static()
        _x = np.random.rand(8, 6)
        _value = np.random.rand(8, 3)

        x = paddle.to_tensor(_x)
        value = paddle.to_tensor(_value)

        axes = [1]
        starts = [0]
        ends = [6]
        strides = [2]

        out = x.slice_scatter(value, axes, starts, ends, strides)
        out_ref = numpy_ref(_x, _value, axes, starts, ends, strides)

        np.testing.assert_allclose(out.numpy(), out_ref)

        paddle.enable_static()


class TestSliceScatterApiError(unittest.TestCase):
    def test_error_ndim(self):
        paddle.disable_static()
        with self.assertRaises(ValueError):
            x = paddle.to_tensor(np.random.rand(8, 6, 3))
            value = paddle.to_tensor(np.random.rand(8, 3))
            _ = paddle.slice_scatter(
                x, value, axes=[0], starts=[0], ends=[8], strides=[1]
            )

    def test_error_index(self):
        paddle.disable_static()
        with self.assertRaises(ValueError):
            x = paddle.to_tensor(np.random.rand(8, 6))
            value = paddle.to_tensor(np.random.rand(8, 3))
            _ = paddle.slice_scatter(
                x, value, axes=[1], starts=[0], ends=[6], strides=[1]
            )

        with self.assertRaises(ValueError):
            x = paddle.to_tensor(np.random.rand(8, 6))
            value = paddle.to_tensor(np.random.rand(2, 6))
            _ = paddle.slice_scatter(
                x, value, axes=[0], starts=[0], ends=[8], strides=[1]
            )


if __name__ == '__main__':
    unittest.main()
