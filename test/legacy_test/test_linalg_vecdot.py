#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import sys
import unittest

import numpy as np

import paddle
from paddle.base import core

if sys.platform == 'win32':
    RTOL = {'float32': 1e-02, 'float64': 1e-04}
    ATOL = {'float32': 1e-02, 'float64': 1e-04}
else:
    RTOL = {'float32': 1e-06, 'float64': 1e-15}
    ATOL = {'float32': 1e-06, 'float64': 1e-15}


class VecDotTestCase(unittest.TestCase):
    def setUp(self):
        self.init_config()
        self.generate_input()
        self.generate_expected_output()
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def generate_input(self):
        np.random.seed(123)
        self.x = np.random.random(self.input_shape).astype(self.dtype)
        self.y = np.random.random(self.input_shape).astype(self.dtype)

    def generate_expected_output(self):
        self.expected_output = np.sum(self.x * self.y, axis=self.axis)

    def init_config(self):
        self.dtype = 'float64'
        self.input_shape = (3, 4)
        self.axis = -1

    def test_dygraph(self):
        for place in self.places:
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x, dtype=self.dtype, place=place)
            y_tensor = paddle.to_tensor(self.y, dtype=self.dtype, place=place)
            result = paddle.vecdot(x_tensor, y_tensor, axis=self.axis)

            np.testing.assert_allclose(
                result.numpy(),
                self.expected_output,
                rtol=RTOL[self.dtype],
                atol=ATOL[self.dtype],
            )

    def test_static(self):
        paddle.enable_static()
        for place in self.places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="x", shape=self.input_shape, dtype=self.dtype
                )
                y = paddle.static.data(
                    name="y", shape=self.input_shape, dtype=self.dtype
                )

                result = paddle.vecdot(x, y, axis=self.axis)
                exe = paddle.static.Executor(place)
                output = exe.run(
                    feed={"x": self.x, "y": self.y},
                    fetch_list=[result],
                )[0]

            np.testing.assert_allclose(
                output,
                self.expected_output,
                rtol=RTOL[self.dtype],
                atol=ATOL[self.dtype],
            )


class VecDotTestCaseFloat32(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.input_shape = (3, 4)
        self.axis = -1


class VecDotTestCaseHigherDim(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.input_shape = (2, 3, 4)
        self.axis = -1


class VecDotTestCaseAxis(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.input_shape = (3, 4, 5)
        self.axis = 1


class VecDotTestCaseZeroSize2D(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.input_shape = (0, 4)
        self.axis = -1


class VecDotTestCaseZeroSize3D(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.input_shape = (2, 0, 4)
        self.axis = -1


class VecDotTestCaseZeroSize3DAxis1(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.input_shape = (2, 0, 0)
        self.axis = 0


class VecDotTestCaseZeroSize3DAxis2(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.input_shape = (2, 0, 0)
        self.axis = 1


class VecDotTestCaseError(unittest.TestCase):
    def test_axis_mismatch(self):
        with self.assertRaises(ValueError):
            x = paddle.rand([3, 4], dtype="float32")
            y = paddle.rand([3, 5], dtype="float32")
            paddle.vecdot(x, y, axis=-1)

    @unittest.skipIf(
        core.is_compiled_with_xpu(),
        "Skip XPU for not support uniform(dtype=int)",
    )
    def test_dtype_mismatch(self):
        with self.assertRaises(TypeError):
            x = paddle.rand([3, 4], dtype="float32")
            y = paddle.rand([3, 4], dtype="int32")
            paddle.vecdot(x, y, axis=-1)


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for not support uniform(dtype=int)",
)
class VecDotTestCaseComplex(unittest.TestCase):
    def run_test_dynamic(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype="complex64"
        )
        y = paddle.to_tensor(
            [[9 + 1j, 8 + 2j], [7 + 3j, 6 + 4j]], dtype="complex64"
        )
        result = paddle.vecdot(x, y, axis=-1)
        expected = np.sum((x.numpy().conj() * y.numpy()), axis=-1)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def run_test_static(self):
        paddle.enable_static()
        place = paddle.CPUPlace()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[2, 2], dtype="complex64")
            y = paddle.static.data(name="y", shape=[2, 2], dtype="complex64")
            result = paddle.vecdot(x, y, axis=-1)
            exe = paddle.static.Executor(place)
            output = exe.run(
                feed={
                    "x": np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]).astype(
                        "complex64"
                    ),
                    "y": np.array([[9 + 1j, 8 + 2j], [7 + 3j, 6 + 4j]]).astype(
                        "complex64"
                    ),
                },
                fetch_list=[result],
            )[0]
        expected = np.sum(
            np.conj(np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])).astype(
                "complex64"
            )
            * np.array([[9 + 1j, 8 + 2j], [7 + 3j, 6 + 4j]]).astype(
                "complex64"
            ),
            axis=-1,
        )
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

    def test_complex_conjugate(self):
        self.run_test_dynamic()
        self.run_test_static()


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for not support uniform(dtype=int)",
)
class VecDotTestCaseTypePromotion1(unittest.TestCase):
    def test_float32_float64_promotion(self):
        paddle.disable_static()
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype="float64")
        result = paddle.vecdot(x, y, axis=-1)

        expected = np.sum(x.numpy().astype("float64") * y.numpy(), axis=-1)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-6, atol=1e-6
        )


@unittest.skipIf(
    core.is_compiled_with_xpu(),
    "Skip XPU for not support uniform(dtype=int)",
)
class VecDotTestCaseTypePromotion2(unittest.TestCase):
    def test_float64_complex64_promotion(self):
        paddle.disable_static()
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
        y = paddle.to_tensor(
            [[5 + 6j, 7 + 8j], [9 + 1j, 2 + 3j]], dtype="complex64"
        )
        result = paddle.vecdot(x, y, axis=-1)

        expected = np.sum(x.numpy().astype("complex64") * y.numpy(), axis=-1)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )


class VecDotTestCaseBroadcast0DTensor(unittest.TestCase):
    def test_0d_tensor_broadcast(self):
        paddle.disable_static()
        x = paddle.to_tensor(2.0, dtype="float32")
        y = paddle.to_tensor(3.0, dtype="float32")
        result = paddle.vecdot(x, y)

        expected = x.numpy() * y.numpy()
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-6, atol=1e-6
        )


class VecDotTestCaseBroadcast1DTensor(unittest.TestCase):
    def test_1d_tensor_broadcast(self):
        paddle.disable_static()
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        y = paddle.to_tensor([4.0, 5.0, 6.0], dtype="float32")
        result = paddle.vecdot(x, y)

        expected = np.dot(x.numpy(), y.numpy())
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-6, atol=1e-6
        )


class VecDotTestCaseBroadcast1DNDTensor(unittest.TestCase):
    def test_1d_nd_tensor_broadcast(self):
        paddle.disable_static()
        x = paddle.to_tensor([1.0, 2.0], dtype="float32")
        y = paddle.to_tensor([[3.0, 4.0], [5.0, 6.0]], dtype="float32")
        result = paddle.vecdot(x, y, axis=-1)

        expected = np.sum(x.numpy() * y.numpy(), axis=-1)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-6, atol=1e-6
        )


class VecDotTestCaseBroadcastNDTensor(unittest.TestCase):
    def test_nd_nd_tensor_broadcast(self):
        paddle.disable_static()
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        y = paddle.to_tensor([5.0, 6.0], dtype="float32")
        result = paddle.vecdot(x, y, axis=-1)

        expected = np.sum(x.numpy() * y.numpy(), axis=-1)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-6, atol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
