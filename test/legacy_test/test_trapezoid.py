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

import os
import unittest

import numpy as np

import paddle


def get_ref_api():
    return (
        np.trapezoid
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0"
        else np.trapz  # noqa: NPY201
    )


class TestTrapezoidAPI(unittest.TestCase):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = None
        self.axis = -1

    def get_output(self):
        if self.x is None and self.dx is None:
            self.output = self.ref_api(
                y=self.y, x=self.x, dx=1.0, axis=self.axis
            )
        else:
            self.output = self.ref_api(
                y=self.y, x=self.x, dx=self.dx, axis=self.axis
            )

    def set_api(self):
        self.ref_api = get_ref_api()
        self.paddle_api = paddle.trapezoid

    def setUp(self):
        self.set_api()
        self.set_args()
        self.get_output()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if paddle.device.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def func_dygraph(self):
        for place in self.places:
            paddle.disable_static()
            y = paddle.to_tensor(self.y, place=place)
            if self.x is not None:
                self.x = paddle.to_tensor(self.x, place=place)
            if self.dx is not None:
                self.dx = paddle.to_tensor(self.dx, place=place)
            out = self.paddle_api(y=y, x=self.x, dx=self.dx, axis=self.axis)
            np.testing.assert_allclose(out, self.output, rtol=1e-05)

    def test_dygraph(self):
        self.setUp()
        self.func_dygraph()

    def test_static(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                y = paddle.static.data(
                    name="y", shape=self.y.shape, dtype=self.y.dtype
                )
                x = None
                dx = None
                if self.x is not None:
                    x = paddle.static.data(
                        name="x", shape=self.x.shape, dtype=self.x.dtype
                    )
                if self.dx is not None:
                    dx = paddle.static.data(
                        name="dx", shape=[], dtype='float32'
                    )

                exe = paddle.static.Executor(place)
                out = self.paddle_api(y=y, x=x, dx=dx, axis=self.axis)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "y": self.y,
                        "x": self.x,
                        "dx": self.dx,
                        "axis": self.axis,
                    },
                    fetch_list=[out],
                )
                np.testing.assert_allclose(fetches[0], self.output, rtol=1e-05)


class TestTrapezoidWithX(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = np.array([[1, 2, 3], [3, 4, 5]]).astype('float32')
        self.dx = None
        self.axis = -1


class TestTrapezoidAxis(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 1.0
        self.axis = 0


class TestTrapezoidWithDx(TestTrapezoidAPI):
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


class TestTrapezoidWithOutDxX(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float64')
        self.x = None
        self.dx = None
        self.axis = -1


class TestTrapezoidBroadcast(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.random.random((3, 3, 4)).astype('float32')
        self.x = np.random.random(3).astype('float32')
        self.dx = None
        self.axis = 1


class TestTrapezoidAxis1(TestTrapezoidAPI):
    def set_args(self):
        self.y = np.random.random((3, 3, 4)).astype('float32')
        self.x = None
        self.dx = 1.0
        self.axis = 1


class TestTrapezoidError(unittest.TestCase):
    # test error
    def set_api(self):
        self.paddle_api = paddle.trapezoid

    def test_errors(self):
        self.set_api()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_y_dtype():
                y = paddle.static.data(
                    name='y',
                    shape=[4, 4],
                    dtype="int64",
                )
                x = paddle.static.data(name='x', shape=[4, 4], dtype="float32")
                dx = None
                self.paddle_api(y, x, dx)

            self.assertRaises(TypeError, test_y_dtype)

            def test_x_dtype():
                y1 = paddle.static.data(
                    name='y1',
                    shape=[4, 4],
                    dtype="float32",
                )
                x1 = paddle.static.data(name='x1', shape=[4, 4], dtype="int64")
                dx1 = None
                self.paddle_api(y1, x1, dx1)

            self.assertRaises(TypeError, test_x_dtype)

            def test_dx_dim():
                y2 = paddle.static.data(
                    name='y2',
                    shape=[4, 4],
                    dtype="float32",
                )
                x2 = None
                dx2 = paddle.static.data(
                    name='dx2', shape=[4, 4], dtype="float32"
                )
                self.paddle_api(y2, x2, dx2)

            self.assertRaises(ValueError, test_dx_dim)

            def test_xwithdx():
                y3 = paddle.static.data(
                    name='y3',
                    shape=[4, 4],
                    dtype="float32",
                )
                x3 = paddle.static.data(
                    name='x3', shape=[4, 4], dtype="float32"
                )
                dx3 = 1.0
                self.paddle_api(y3, x3, dx3)

            self.assertRaises(ValueError, test_xwithdx)


class Testfp16Trapezoid(TestTrapezoidAPI):
    def set_api(self):
        self.paddle_api = paddle.trapezoid
        self.ref_api = get_ref_api()

    def test_fp16_with_gpu(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_y = np.random.random([4, 4]).astype("float16")
                y = paddle.static.data(name="y", shape=[4, 4], dtype="float16")

                input_x = np.random.random([4, 4]).astype("float16")
                x = paddle.static.data(name="x", shape=[4, 4], dtype="float16")

                exe = paddle.static.Executor(place)
                out = self.paddle_api(y=y, x=x, dx=self.dx, axis=self.axis)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "y": input_y,
                        "x": input_x,
                        "dx": self.dx,
                        "axis": self.axis,
                    },
                    fetch_list=[out],
                )

    def test_fp16_func_dygraph(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            paddle.disable_static()
            input_y = np.random.random([4, 4])
            y = paddle.to_tensor(input_y, dtype='float16', place=place)
            input_x = np.random.random([4, 4])
            x = paddle.to_tensor(input_x, dtype='float16', place=place)
            out = self.paddle_api(y=y, x=x)

    def test_fp16_dygraph(self):
        self.func_dygraph()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
