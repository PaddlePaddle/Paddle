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

import paddle
import unittest
import paddle.fluid.core as core
import numpy as np
import itertools as it

np.set_printoptions(threshold=np.inf)


def tensordot_np(x, y, axes):
    if isinstance(axes, paddle.fluid.framework.Variable):
        axes = axes.tolist()

    # np.tensordot does not support empty axes
    if not axes:
        axes = 0
    if (isinstance(axes, (tuple, list))):
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
    if (isinstance(axes, (tuple, list))):
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
        self.set_dtype()
        self.set_input_shape()
        self.set_input_data()

    def set_dtype(self):
        self.dtype = np.float32

    def set_input_shape(self):
        self.x_shape = [5, 5, 5, 5]
        self.y_shape = [5, 5, 5, 5]

    def set_input_data(self):
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.all_axes = [2]

    def run_dygraph(self, place):
        paddle.disable_static()
        x = paddle.to_tensor(self.x, place=place)
        y = paddle.to_tensor(self.y, place=place)
        paddle_res = paddle.tensordot(x, y, self.axes)
        np_res = tensordot_np(self.x, self.y, self.axes)
        np.testing.assert_allclose(paddle_res, np_res, rtol=1e-6)

    def run_static(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x = paddle.static.data(
                name='x', shape=self.x_shape, dtype=self.dtype)
            y = paddle.static.data(
                name='y', shape=self.y_shape, dtype=self.dtype)
            z = paddle.tensordot(x, y, self.axes)
            exe = paddle.static.Executor(place)
            paddle_res = exe.run(feed={'x': self.x,
                                       'y': self.y},
                                 fetch_list=[z])
            np_res = tensordot_np(self.x, self.y, self.axes)
            np.testing.assert_allclose(paddle_res[0], np_res, rtol=1e-6)

    def test_cases(self):
        self.all_axes = []
        axial_index = range(4)
        all_permutations = list(it.permutations(axial_index, 0)) + list(
            it.permutations(axial_index, 1)) + list(
                it.permutations(axial_index, 2)) + list(
                    it.permutations(axial_index, 3)) + list(
                        it.permutations(axial_index, 4))
        self.all_axes.extend(list(i) for i in all_permutations)

        for axes_x in all_permutations:
            for axes_y in all_permutations:
                if len(axes_x) < len(axes_y):
                    supplementary_axes_x = axes_x + axes_y[len(axes_x):]
                    if any(
                            supplementary_axes_x.count(i) > 1
                            for i in supplementary_axes_x):
                        continue
                elif len(axes_y) < len(axes_x):
                    supplementary_axes_y = axes_y + axes_x[len(axes_y):]
                    if any(
                            supplementary_axes_y.count(i) > 1
                            for i in supplementary_axes_y):
                        continue
                self.all_axes.append([list(axes_x), list(axes_y)])

        self.all_axes.extend(range(5))

        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for axes in self.all_axes:
            self.axes = axes
            for place in places:
                self.run_dygraph(place)
                self.run_static(place)


class TestTensordotAPIFloat64(TestTensordotAPI):
    def set_dtype(self):
        self.dtype = np.float64


class TestTensordotAPIAxesType(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [3, 4, 4]
        self.y_shape = [4, 4, 5]

    def test_cases(self):
        self.all_axes = [
            0, 1, 2, (1, ), [1], ((1, ), ), ([1], ), ((2, 1), (0, )), (
                (1, 2), (0, 1)), ([1, 2], [0, 1]), ([1, 2], [0, 1]),
            [[1, 2], [0, 1]]
        ]

        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for axes in self.all_axes:
            self.axes = axes
            for place in places:
                self.run_dygraph(place)
                self.run_static(place)

        # The 'axes' with type 'Tensor' in tensordot is not available in static mode
        paddle.disable_static()
        for place in places:
            self.all_axes = [
                paddle.to_tensor([1]), (paddle.to_tensor([1])),
                (paddle.to_tensor([1, 2]), paddle.to_tensor([0, 1])),
                [paddle.to_tensor([1, 2]), paddle.to_tensor([0, 1])],
                paddle.to_tensor([[1, 2], [0, 1]])
            ]
            for axes in self.all_axes:
                self.axes = axes
                for place in places:
                    self.run_dygraph(place)

    def test_error(self):
        self.all_axes = [[[[0], [1]]], 0.1, -1, 100, [[1, 2], [0, 0]],
                         [[1, 2], [0, -1]], [0, 1, 2, 3]]
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        for axes in self.all_axes:
            with self.assertRaises(BaseException):
                paddle.tensordot(x, y, axes)


class TestTensordotAPIAxesTypeFloat64(TestTensordotAPIAxesType):
    def set_dtype(self):
        self.dtype = np.float64


class TestTensordotAPIBroadcastCase1(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [1, 1, 1, 5]
        self.y_shape = [1, 5, 1, 1]


class TestTensordotAPIBroadcastCase2(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [1, 5, 5, 5]
        self.y_shape = [1, 1, 1, 5]


class TestTensordotAPIBroadcastCase3(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [5, 5, 5, 1]
        self.y_shape = [5, 5, 1, 5]


class TestTensordotAPIBroadcastCase4(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [5, 5, 5, 1]
        self.y_shape = [1, 1, 1, 1]


class TestTensordotAPIBroadcastCase5(TestTensordotAPI):
    def set_input_shape(self):
        self.x_shape = [1, 1, 5, 5]
        self.y_shape = [5, 5, 1, 5]


if __name__ == "__main__":
    unittest.main()
