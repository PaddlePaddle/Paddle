#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle.base import core

np.random.seed(2021)


def tensordot_np(x, y, axes):
    if isinstance(axes, paddle.base.framework.Variable):
        axes = axes.tolist()

    # np.tensordot does not support empty axes
    if not axes:
        axes = 0
    if isinstance(axes, (tuple, list)):
        if all(np.issubdtype(type(i), np.integer) for i in axes):
            axes = [axes, axes]
        else:
            axes_x = axes[0]
            if len(axes) > 1:
                axes_y = axes[1]
            else:
                axes_y = axes_x
            len_axes_x, len_axes_y = len(axes_x), len(axes_y)
            if len_axes_x < len_axes_y:
                axes_x = axes_x + axes_y[len_axes_x:]
            elif len_axes_y < len_axes_x:
                axes_y = axes_y + axes_x[len_axes_y:]
            axes = [axes_x, axes_y]

    # np.tensordot does not support broadcast
    if isinstance(axes, (tuple, list)):
        axes_x, axes_y = axes
    else:
        axes_x = list(range(x.ndim - axes, x.ndim))
        axes_y = list(range(axes))
    shape_x, shape_y = list(np.shape(x)), list(np.shape(y))
    for i in range(len(axes_x)):
        dim_x, dim_y = axes_x[i], axes_y[i]
        sx, sy = shape_x[dim_x], shape_y[dim_y]
        if sx == 1:
            shape_y[dim_y] = 1
            y = np.sum(y, dim_y)
            y = np.reshape(y, shape_y)
        elif sy == 1:
            shape_x[dim_x] = 1
            x = np.sum(x, dim_x)
            x = np.reshape(x, shape_x)

    return np.tensordot(x, y, axes)


class TestTensordotAPI(unittest.TestCase):
    def setUp(self):
        self.set_place()
        self.set_dtype()
        self.set_input_shape()
        self.set_input_data()
        self.set_test_axes()

    def set_place(self):
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(core.CUDAPlace(0))

    def set_dtype(self):
        self.dtype = np.float32

    def set_input_shape(self):
        self.x_shape = [5, 5, 5, 5]
        self.y_shape = [5, 5, 5, 5]

    def set_input_data(self):
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)

    def set_test_axes(self):
        self.all_axes = [
            [[3, 2], [3]],
            [[2, 1, 0], [2, 1]],
            [[1, 2, 0], [1, 3, 2]],
            [3, 0],
            [[], [0, 3, 1]],
            [[2, 1, 0, 3], [2, 0, 1, 3]],
            [[3, 1, 2], [1, 3, 2, 0]],
            [[2, 1], [0, 2]],
            [[2, 0, 1, 3], [2]],
            [[1, 2, 0, 3], [0, 2, 1]],
            [[2, 1, 3, 0], [1, 2, 3]],
            [[2, 0, 1, 3], [3, 1, 0, 2]],
            [[0, 3], [0, 3, 2, 1]],
            [[1, 3, 2, 0], [2, 1, 0, 3]],
            [[1, 3, 2, 0], [1, 3, 2, 0]],
            [[1, 0, 2], [0, 1]],
            [[2, 3, 0], [3, 1]],
            [[1, 3, 2, 0], [3, 0, 1, 2]],
            [[3, 2, 1], [2, 0, 1]],
            [[0], []],
            [[2, 3, 0], [1, 2, 0]],
            [[3, 0, 2, 1], [2, 1, 0, 3]],
            [[3, 1, 2], [2, 3, 1]],
            [[1, 0, 2, 3], []],
            [[1, 2], [1, 2, 3]],
            [[2, 0, 1, 3], [2, 0, 1]],
            [[3, 1, 2], [1, 3, 2]],
            [[3, 1, 2, 0], [1, 2, 3, 0]],
            [[0, 2, 3], [0, 1, 2]],
            [[3, 2, 0], [2, 0, 3, 1]],
            [[2, 1, 0, 3], [3, 1, 2, 0]],
            [[1, 2, 3, 0], [1, 3, 0, 2]],
            [[3, 0], [2, 1]],
            [[0, 1, 3, 2], [0, 2, 1, 3]],
            [[1, 0], [2, 1, 3]],
            [[1, 0, 3, 2], [2, 3, 0, 1]],
            [[1, 2], [3]],
            [[1, 2, 3, 0], [3, 2, 1, 0]],
            [[0, 3, 2, 1], [2, 1, 3, 0]],
            [0],
            [[0, 2, 3], [3, 2, 0, 1]],
            [[1, 2, 3, 0], [3, 2, 1, 0]],
            [[3, 1], [3]],
            [[3, 2, 0, 1], [3, 2, 0]],
            [[2, 3, 0, 1], [0, 3, 2]],
            [[1], [1, 3]],
            [[1, 2], [2, 1, 0]],
            [[3, 1, 2], [3, 1, 0]],
            [[1, 3], [3, 1, 2]],
            [[2, 0, 1, 3], [3, 1, 0, 2]],
            [[1, 3, 0], [1, 3]],
            [[2, 3, 1], [1, 0, 2]],
            [[1, 2, 0, 3], [0, 2, 1, 3]],
            [[2], [0, 1, 3]],
            [[1], [1, 2]],
            [[1, 0, 2, 3], [3, 0, 1, 2]],
            [[0, 1, 3, 2], [1, 3, 0, 2]],
            [[3, 0, 2, 1], [0, 2, 3]],
            [[1, 2, 0], [1, 2, 3]],
            [[1, 0, 3], [2, 3, 0]],
            [[2, 3, 0], [3, 1, 0]],
            [[1, 3], [1, 0]],
            [[2, 1, 0, 3], [2, 0, 3, 1]],
            [[3, 2, 0], [2, 1, 0]],
            [[0, 1, 3], [0, 3, 1]],
            [[3, 1, 0], [3, 2, 1]],
            [[3, 2], [3, 1]],
            [[3], [2, 1, 0]],
            [[1, 2, 3, 0], []],
            [[1, 3, 2, 0], [3, 1, 2]],
            [[1], [0, 2]],
            [[3, 2, 0], [3, 2, 0]],
            [[3], []],
            [[1, 0, 3], [2, 1]],
            [[3, 1, 0, 2], [2, 3, 1, 0]],
            [[0, 1], [0, 3, 2]],
            [[0, 2, 3], [0, 2, 1]],
            [[1, 3, 0], [3, 0, 2]],
            [[3, 1, 2], [1, 2, 3]],
            [[3, 1, 2], [3, 1, 0]],
            [[0, 3, 1, 2], [3, 2, 1, 0]],
            [[0, 3], [3, 2, 1]],
            [[2, 3], [1, 3, 0]],
            [[0, 3, 2], [2, 0, 3, 1]],
            [[2, 3], [1, 3]],
            [[3, 1, 2, 0], [2, 3, 1, 0]],
            [[1, 0, 3, 2], [3, 0, 1, 2]],
            [[3, 2, 1, 0], [0, 1, 3, 2]],
            [[3, 1, 2], [3]],
            [[0, 1, 3, 2], [2, 3, 0, 1]],
            [[1, 2, 3, 0], [1, 3, 0, 2]],
            [3, 1, 2],
            [[3, 1, 2], [0, 3, 2]],
            [[2, 3, 0], [1, 2, 0]],
            [[2, 0, 3], [2, 0]],
            [[3, 1, 0, 2], [3, 1, 0, 2]],
            [[0, 1, 2], [2, 0, 1]],
            [[1, 0, 3], [2, 3, 0]],
            [[2, 0, 1], [0, 1, 3]],
            [[2, 1], [0, 1, 3]],
        ]

    def test_dygraph(self):
        paddle.disable_static()
        for axes in self.all_axes:
            for place in self.places:
                x = paddle.to_tensor(self.x, place=place)
                y = paddle.to_tensor(self.y, place=place)
                paddle_res = paddle.tensordot(x, y, axes)
                np_res = tensordot_np(self.x, self.y, axes)
                np.testing.assert_allclose(paddle_res, np_res, rtol=1e-6)

    def test_static(self):
        paddle.enable_static()
        for axes in self.all_axes:
            for place in self.places:
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(
                        name='x', shape=self.x_shape, dtype=self.dtype
                    )
                    y = paddle.static.data(
                        name='y', shape=self.y_shape, dtype=self.dtype
                    )
                    z = paddle.tensordot(x, y, axes)
                    exe = paddle.static.Executor(place)
                    paddle_res = exe.run(
                        feed={'x': self.x, 'y': self.y}, fetch_list=[z]
                    )
                    np_res = tensordot_np(self.x, self.y, axes)
                    np.testing.assert_allclose(paddle_res[0], np_res, rtol=1e-6)

    def test_fp16_with_gpu(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            for axes in self.all_axes:
                place = paddle.CUDAPlace(0)
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    input_x = np.random.random([5, 5, 5, 5]).astype("float16")
                    x = paddle.static.data(
                        name="x", shape=[5, 5, 5, 5], dtype="float16"
                    )

                    input_y = np.random.random([5, 5, 5, 5]).astype("float16")
                    y = paddle.static.data(
                        name="y", shape=[5, 5, 5, 5], dtype="float16"
                    )

                    z = paddle.tensordot(x, y, axes)
                    exe = paddle.static.Executor(place)

                    paddle_res = exe.run(
                        feed={'x': input_x, 'y': input_y}, fetch_list=[z]
                    )
                    np_res = tensordot_np(input_x, input_y, axes)
                    np.testing.assert_allclose(paddle_res[0], np_res, rtol=1)


class TestTensordotAPIFloat64(TestTensordotAPI):
    def set_dtype(self):
        self.dtype = np.float64


class TestTensordotAPIBroadcastCase1(TestTensordotAPIFloat64):
    def set_input_shape(self):
        self.x_shape = [1, 1, 1, 5]
        self.y_shape = [1, 5, 1, 1]


class TestTensordotAPIBroadcastCase2(TestTensordotAPIFloat64):
    def set_input_shape(self):
        self.x_shape = [1, 5, 5, 5]
        self.y_shape = [1, 1, 1, 5]


class TestTensordotAPIBroadcastCase3(TestTensordotAPIFloat64):
    def set_input_shape(self):
        self.x_shape = [5, 5, 5, 1]
        self.y_shape = [5, 5, 1, 5]


class TestTensordotAPIBroadcastCase4(TestTensordotAPIFloat64):
    def set_input_shape(self):
        self.x_shape = [5, 5, 5, 1]
        self.y_shape = [1, 1, 1, 1]


class TestTensordotAPIBroadcastCase5(TestTensordotAPIFloat64):
    def set_input_shape(self):
        self.x_shape = [1, 1, 5, 5]
        self.y_shape = [5, 5, 1, 5]


class TestTensordotAPIAxesType(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [3, 4, 4]
        self.y_shape = [4, 4, 5]

    def set_test_axes(self):
        self.all_axes = [
            0,
            1,
            2,
            (1,),
            [1],
            ((1,),),
            ([1],),
            ((2, 1), (0,)),
            ((1, 2), (0, 1)),
            ([1, 2], [0, 1]),
            ([1, 2], [0, 1]),
            [[1, 2], [0, 1]],
        ]

    def test_tensor_axes(self):
        # The 'axes' with type 'Tensor' in tensordot is not available in static graph mode
        paddle.disable_static()
        tensor_axes = [
            paddle.to_tensor([1]),
            (paddle.to_tensor([1])),
            (paddle.to_tensor([1, 2]), paddle.to_tensor([0, 1])),
            [paddle.to_tensor([1, 2]), paddle.to_tensor([0, 1])],
            paddle.to_tensor([[1, 2], [0, 1]]),
        ]

        for place in self.places:
            for axes in tensor_axes:
                x = paddle.to_tensor(self.x, place=place)
                y = paddle.to_tensor(self.y, place=place)
                paddle_res = paddle.tensordot(x, y, axes)
                np_res = tensordot_np(self.x, self.y, axes)
                np.testing.assert_allclose(paddle_res, np_res, rtol=1e-6)

    def test_error(self):
        self.all_axes = [
            [[[0], [1]]],
            0.1,
            -1,
            100,
            [[1, 2], [0, 0]],
            [[1, 2], [0, -1]],
            [0, 1, 2, 3],
        ]
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)

        with self.assertRaises(TypeError):
            paddle.tensordot(x, y, axes=self.all_axes[0])
        with self.assertRaises(TypeError):
            paddle.tensordot(x, y, axes=self.all_axes[1])
        with self.assertRaises(AssertionError):
            paddle.tensordot(x, y, axes=self.all_axes[2])
        with self.assertRaises(IndexError):
            paddle.tensordot(x, y, axes=self.all_axes[3])
        with self.assertRaises(ValueError):
            paddle.tensordot(x, y, axes=self.all_axes[4])
        with self.assertRaises(AssertionError):
            paddle.tensordot(x, y, axes=self.all_axes[5])
        with self.assertRaises(AssertionError):
            paddle.tensordot(x, y, axes=self.all_axes[6])


class TestTensordotAPIAxesTypeFloat64(TestTensordotAPIAxesType):
    def set_dtype(self):
        self.dtype = np.float64


class TestTensordotAPIZeroSize(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [0, 5, 5, 5]
        self.y_shape = [0, 5, 5, 5]

    def set_input_data(self):
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)

    def set_test_axes(self):
        self.all_axes = [
            [[], []],
        ]


class TestTensordotAPIFloat64ZeroSize(TestTensordotAPIZeroSize):
    def set_dtype(self):
        self.dtype = np.float64


class TestTensordotAPIZeroSize(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [0, 5, 5, 5]
        self.y_shape = [0, 5, 5, 5]

    def set_input_data(self):
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)

    def set_test_axes(self):
        self.all_axes = [
            [[], []],
        ]

    def set_dtype(self):
        self.dtype = np.float64


class TestTensordotAPIZeroSizeMultipleDims1(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [0, 0, 5, 5]
        self.y_shape = [0, 0, 5, 5]

    def set_test_axes(self):
        self.all_axes = [
            [[], []],
        ]


class TestTensordotAPIZeroSizeMultipleDims2(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [5, 0, 5, 0]
        self.y_shape = [5, 0, 5, 0]

    def set_test_axes(self):
        self.all_axes = [
            [[], []],
        ]


class TestTensordotAPIZeroSizeDifferentDims1(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [5, 5, 0, 5]
        self.y_shape = [5, 5, 0, 5]

    def set_test_axes(self):
        self.all_axes = [
            [[], []],
        ]


class TestTensordotAPIZeroSizeDifferentDims2(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [5, 5, 5, 0]
        self.y_shape = [5, 5, 5, 0]

    def set_test_axes(self):
        self.all_axes = [
            [[], []],
        ]


class TestTensordotAPISingleElementAndZeroSize(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [1, 5, 5, 5]
        self.y_shape = [0, 5, 5, 5]

    def set_test_axes(self):
        self.all_axes = [
            [[], []],
        ]


class TestBroadcastWithZeroSize1(unittest.TestCase):
    def setUp(self):
        self.x_shape = [5, 0, 3]
        self.y_shape = [3, 4, 0]

    def set_test_axes(self):
        self.all_axes = [[], []]


class TestBroadcastWithZeroSize2(unittest.TestCase):
    def setUp(self):
        self.x_shape = [5, 0, 3]
        self.y_shape = [3, 0]

    def set_test_axes(self):
        self.all_axes = [[], []]


if __name__ == "__main__":
    unittest.main()
