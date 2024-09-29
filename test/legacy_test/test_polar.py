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

# import torch
import numpy as np

import paddle
from paddle.base import core

np.random.seed(10)


def numpy_polar(abs, angle):
    real = np.multiply(abs, np.cos(angle))
    imag = np.multiply(abs, np.sin(angle))
    return real + imag * 1j


class TestPolarAPI(unittest.TestCase):
    def setUp(self):
        self.abs = np.array([1, 2]).astype("float64")
        self.angle = np.array([np.pi / 2, 5 * np.pi / 4]).astype("float64")
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                abs = paddle.static.data(
                    'abs',
                    shape=self.abs.shape,
                    dtype="float64",
                )
                angle = paddle.static.data(
                    'angle', shape=self.angle.shape, dtype="float64"
                )
                out1 = paddle.polar(abs, angle)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={'abs': self.abs, 'angle': self.angle},
                    fetch_list=[out1],
                )
            out_ref = numpy_polar(self.abs, self.angle)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            abs = paddle.to_tensor(self.abs)
            angle = paddle.to_tensor(self.angle)
            out1 = paddle.polar(abs, angle)

            out_ref1 = numpy_polar(self.abs, self.angle)
            np.testing.assert_allclose(out_ref1, out1.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_out_complex64(self):
        paddle.disable_static()
        abs = paddle.to_tensor(self.abs, dtype=paddle.float32)
        angle = paddle.to_tensor(self.angle, dtype=paddle.float32)
        out = paddle.polar(abs, angle)
        self.assertTrue(out.type, 'complex64')

    def test_out_complex128(self):
        paddle.disable_static()
        abs = paddle.to_tensor(self.abs, dtype=paddle.float64)
        angle = paddle.to_tensor(self.angle, dtype=paddle.float64)
        out = paddle.polar(abs, angle)
        self.assertTrue(out.type, 'complex128')

    def test_empty_input_error(self):
        for place in self.place:
            paddle.disable_static(place)
            abs = paddle.to_tensor(self.abs)
            angle = paddle.to_tensor(self.angle)
            self.assertRaises(AttributeError, paddle.polar, None, angle)
            self.assertRaises(AttributeError, paddle.polar, abs, None)


if __name__ == "__main__":
    unittest.main()
