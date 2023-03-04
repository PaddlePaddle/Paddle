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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard


class TestTrapezoidAPI(unittest.TestCase):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 1.0
        self.axis = -1

    def get_output(self):
        self.output = np.trapz(y=self.y, x=self.x, dx=self.dx, axis=self.axis)

    def setUp(self):
        self.set_args()
        self.get_output()
        self.places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def func_dygraph(self):
        for place in self.places:
            paddle.disable_static()
            y = paddle.to_tensor(self.y, place=place)
            if self.x is not None:
                self.x = paddle.to_tensor(self.x, place=place)
            if self.dx is not None:
                self.dx = paddle.to_tensor(self.dx, place=place)
            out = paddle.trapezoid(y=y, x=self.x, dx=self.dx, axis=self.axis)
            np.testing.assert_allclose(out, self.output, rtol=1e-05)
            # self.assertTrue((out.numpy() == self.output).all(), True)

    def test_dygraph(self):
        with _test_eager_guard():
            self.setUp()
            self.func_dygraph()
        self.setUp()
        self.func_dygraph()

    def test_static(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                y = paddle.fluid.data(
                    name="y", shape=self.y.shape, dtype=self.y.dtype
                )
                x = None
                dx = None
                if self.x is not None:
                    x = paddle.fluid.data(
                        name="x", shape=self.x.shape, dtype=self.x.dtype
                    )
                if self.dx is not None:
                    dx = paddle.fluid.data(
                        name="dx", shape=[1], dtype='float32'
                    )

                exe = fluid.Executor(place)
                out = paddle.trapezoid(y=y, x=x, dx=dx, axis=self.axis)
                fetches = exe.run(
                    fluid.default_main_program(),
                    feed={
                        "y": self.y,
                        "x": self.x,
                        "dx": self.dx,
                        "axis": self.axis,
                    },
                    fetch_list=[out],
                )
                np.testing.assert_allclose(fetches[0], self.output, rtol=1e-05)
                # self.assertTrue((fetches[0] == self.output).all(), True)


class TestTrapezoidAPIX(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = np.array([[1, 2, 3], [3, 4, 5]]).astype('float32')
        self.dx = None
        self.axis = -1


class TestTrapezoidAPIAxis(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 1.0
        self.axis = 0


class TestTrapezoidXdx(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 3.0
        self.axis = -1


class TestTrapezoidfloat64(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float64')
        self.x = np.array([[1, 2, 3], [3, 4, 5]]).astype('float64')
        self.dx = None
        self.axis = -1


class TestTrapezoidBroadcast(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.random.random((3, 3, 4)).astype('float32')
        self.x = np.random.random((3)).astype('float32')
        self.dx = None
        self.axis = 1


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
