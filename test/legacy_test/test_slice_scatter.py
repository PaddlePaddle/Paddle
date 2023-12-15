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

import unittest

import numpy as np

import paddle
from paddle.framework import core
from paddle.pir_utils import test_with_pir_api

paddle.enable_static()

RTOL = {'float32': 1e-03, 'float64': 1e-05}
ATOL = {'float32': 1e-03, 'float64': 1e-05}


def numpy_ref(x, value, axis=0, start=None, stop=None, step=1):
    _x = np.copy(x)

    start = 0 if start is None else start
    stop = _x.shape[axis] if stop is None else stop

    index = range(start, stop, step)
    exp_shape = [
        *([1] * _x.ndim)[:axis],
        len(index),
        *([1] * _x.ndim)[axis + 1 :],
    ]

    np.put_along_axis(
        _x, np.arange(start, stop, step).reshape(exp_shape), value, axis=axis
    )

    return _x


class TestSliceScatterApi(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)

        self.init_dtype()
        self.init_shape()

        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.value_np = np.random.random(self.value_shape).astype(self.dtype)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def init_dtype(self):
        self.dtype = 'float64'

    def init_shape(self):
        self.x_shape = [8, 6]
        self.value_shape = [8, 2]
        self.axis = 1
        self.start = 2
        self.stop = 6
        self.step = 2

    @test_with_pir_api
    def test_api_static(self):
        paddle.enable_static()

        for place in self.place:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('x', self.x_shape, self.dtype)
                value = paddle.static.data(
                    'value', self.value_shape, self.dtype
                )

                out = paddle.slice_scatter(
                    x,
                    value,
                    axis=self.axis,
                    start=self.start,
                    stop=self.stop,
                    step=self.step,
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
                axis=self.axis,
                start=self.start,
                stop=self.stop,
                step=self.step,
            )

            np.testing.assert_allclose(
                res, out_ref, rtol=RTOL[self.dtype], atol=ATOL[self.dtype]
            )

    def test_api_dygraph(self):
        for place in self.place:
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.slice_scatter(
                x_tensor,
                value_tensor,
                axis=self.axis,
                start=self.start,
                stop=self.stop,
                step=self.step,
            )
            out_ref = numpy_ref(
                self.x_np,
                self.value_np,
                axis=self.axis,
                start=self.start,
                stop=self.stop,
                step=self.step,
            )

            np.testing.assert_allclose(
                out.numpy(),
                out_ref,
                rtol=RTOL[self.dtype],
                atol=ATOL[self.dtype],
            )

            paddle.enable_static()


class TestSliceScatterApiFloat32(TestSliceScatterApi):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApiNoneStartStop(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [6, 8]
        self.value_shape = [2, 8]
        self.axis = 0
        self.start = None
        self.stop = None
        self.step = 3


class TestSliceScatterApi3D(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 6, 3]
        self.value_shape = [8, 2, 3]
        self.axis = 1
        self.start = 2
        self.stop = 6
        self.step = 2


class TestSliceScatterApi3DFloat32(TestSliceScatterApi3D):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApi4D(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 6, 3, 5]
        self.value_shape = [8, 2, 3, 5]
        self.axis = 1
        self.start = 2
        self.stop = 6
        self.step = 2


class TestSliceScatterApi4DFloat32(TestSliceScatterApi4D):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApi4DAxis3(TestSliceScatterApi):
    def init_shape(self):
        self.x_shape = [8, 6, 3, 9]
        self.value_shape = [8, 6, 3, 2]
        self.axis = 3
        self.start = 2
        self.stop = 6
        self.step = 2


class TestSliceScatterApi4DAxis3Float32(TestSliceScatterApi4DAxis3):
    def init_dtype(self):
        self.dtype = 'float32'


class TestSliceScatterApiError(unittest.TestCase):
    def test_error_ndim(self):
        with self.assertRaises(ValueError):
            x = np.random.rand(8, 6, 3)
            value = np.random.rand(8, 3)
            _ = paddle.slice_scatter(x, value)

    def test_error_index(self):
        with self.assertRaises(ValueError):
            x = np.random.rand(8, 6)
            value = np.random.rand(8, 3)
            _ = paddle.slice_scatter(x, value, axis=1, step=1)

        with self.assertRaises(ValueError):
            x = np.random.rand(8, 6)
            value = np.random.rand(2, 6)
            _ = paddle.slice_scatter(x, value)


if __name__ == '__main__':
    unittest.main()
