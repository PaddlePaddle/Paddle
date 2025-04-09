#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.framework import core


class TestZeroSizeParameter(unittest.TestCase):
    def setUp(self):
        self.places = [
            "cpu",
        ]
        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            self.places.append("gpu")

        self.parameter_dtypes = [
            'float16',
            'float32',
            'float64',
        ]
        self.zero_size_shapes = [
            [0, 4],
            [0, 0],
            [4, 0],
            [0, 5, 6],
            [6, 5, 0, 0],
            [0, 0, 0, 12],
        ]

    def test_create_parameter(self):
        for place in self.places:
            paddle.device.set_device(place)
            for parameter_dtype in self.parameter_dtypes:
                for zero_size_shape in self.zero_size_shapes:

                    class Model(paddle.nn.Layer):
                        def __init__(self) -> None:
                            super().__init__()
                            self.dummy_linear = paddle.nn.Linear(3, 4)
                            self.w = self.create_parameter(
                                shape=zero_size_shape, dtype=parameter_dtype
                            )

                    model = Model()
                    model = model
                    self.assertEqual(
                        model.w.shape,
                        zero_size_shape,
                        msg=f"Check failed at: {parameter_dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        model.w.data_ptr(),
                        0,
                        msg=f"Check failed at: {parameter_dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        str(model.w.place),
                        str(model.dummy_linear.weight.place),
                        msg=f"Check failed at: {parameter_dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        model.w.strides,
                        zero_size_shape,
                        msg=f"Check failed at: {parameter_dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        model.w.is_contiguous(),
                        True,
                        msg=f"Check failed at: {parameter_dtype}, {zero_size_shape}",
                    )


class TestZeroSizeForward(unittest.TestCase):
    def setUp(self):
        self.places = [
            "cpu",
        ]
        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            self.places.append("gpu")

        self.dtypes = [
            'bool',
            'uint8',
            'int8',
            'int16',
            'int32',
            'int64',
            'float16',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ]
        self.zero_size_shapes = [
            [0, 4],
            [0, 0],
            [4, 0],
            [0, 5, 6],
            [6, 5, 0, 0],
            [0, 0, 0, 12],
        ]

    def test_forward_eager(self):
        """Test for simple API call"""
        for place in self.places:
            paddle.device.set_device(place)
            for dtype in self.dtypes:
                for zero_size_shape in self.zero_size_shapes:
                    x = paddle.ones(zero_size_shape, dtype=dtype)
                    self.assertEqual(x.data_ptr(), 0)

                    if x.dtype == paddle.bool:
                        y = ~x
                    else:
                        y = x + 1

                    self.assertEqual(
                        y.shape,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.data_ptr(),
                        0,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.strides,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        str(y.place),
                        str(x.place),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.dtype,
                        x.dtype,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.is_contiguous(),
                        x.is_contiguous(),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )

    def test_forward_static(self):
        """Test for simple API call"""

        def forward_func(x):
            if x.dtype == paddle.bool:
                y = ~x
            else:
                y = x + 1
            return y

        for place in self.places:
            paddle.device.set_device(place)
            static_forward_func = paddle.jit.to_static(
                forward_func, full_graph=True
            )

            for dtype in self.dtypes:
                for zero_size_shape in self.zero_size_shapes:
                    x = paddle.ones(zero_size_shape, dtype=dtype)
                    self.assertEqual(x.data_ptr(), 0)
                    y = static_forward_func(x)

                    self.assertEqual(
                        y.shape,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.data_ptr(),
                        0,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.strides,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        str(y.place),
                        str(x.place),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.dtype,
                        x.dtype,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        y.is_contiguous(),
                        x.is_contiguous(),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )


@unittest.skipIf(core.is_compiled_with_xpu(), "Skip XPU for xpu place issue")
class TestZeroSizeBackward(unittest.TestCase):
    def setUp(self):
        self.places = [
            "cpu",
        ]
        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            self.places.append("gpu")

        # Only floating and complex needs gradient
        self.dtypes = [
            'float16',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ]
        self.zero_size_shapes = [
            [0, 4],
            [0, 0],
            [4, 0],
            [0, 5, 6],
            [6, 5, 0, 0],
            [0, 0, 0, 12],
        ]

    def test_backward_eager(self):
        """Test for simple API call"""
        for place in self.places:
            paddle.device.set_device(place)
            for dtype in self.dtypes:
                for zero_size_shape in self.zero_size_shapes:
                    x = paddle.ones(zero_size_shape, dtype=dtype)
                    x.stop_gradient = False
                    self.assertEqual(x.data_ptr(), 0)

                    y = x * 2 + 1

                    (x_grad,) = paddle.grad(y, x, create_graph=True)

                    self.assertEqual(
                        x_grad.shape,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.data_ptr(),
                        0,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.strides,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        str(x_grad.place),
                        str(x.place),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.dtype,
                        x.dtype,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.is_contiguous(),
                        x.is_contiguous(),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )

    def test_backward_static(self):
        """Test for simple API call"""

        def gradient_func(x):
            y = x * 2 + 1
            return paddle.grad(y, x)

        for place in self.places:
            paddle.device.set_device(place)
            for dtype in self.dtypes:
                for zero_size_shape in self.zero_size_shapes:
                    x = paddle.ones(zero_size_shape, dtype=dtype)
                    x.stop_gradient = False
                    self.assertEqual(x.data_ptr(), 0)

                    static_gradient_func = paddle.jit.to_static(
                        gradient_func, full_graph=True
                    )
                    (x_grad,) = static_gradient_func(x)

                    self.assertEqual(
                        x_grad.shape,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.data_ptr(),
                        0,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.strides,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        str(x_grad.place),
                        str(x.place),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.dtype,
                        x.dtype,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.is_contiguous(),
                        x.is_contiguous(),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )


@unittest.skipIf(core.is_compiled_with_xpu(), "Skip XPU for xpu place issue")
class TestZeroSizeBackwardWithGradientAccumulation(unittest.TestCase):
    def setUp(self):
        self.places = [
            "cpu",
        ]
        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            self.places.append("gpu")

        # Only floating and complex needs gradient
        self.dtypes = [
            # 'float16',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ]
        self.zero_size_shapes = [
            [0, 4],
            [4, 0],
            [0, 5, 6],
            [6, 12, 0, 0],
            [0, 0, 0, 12],
        ]

    def test_backward_eager(self):
        """Test for simple API call"""
        for place in self.places:
            paddle.device.set_device(place)
            for dtype in self.dtypes:
                for zero_size_shape in self.zero_size_shapes:
                    x = paddle.ones(zero_size_shape, dtype=dtype)
                    x.stop_gradient = False
                    self.assertEqual(x.data_ptr(), 0)

                    def forward_func(x):
                        y1 = x / 2
                        y2 = x + 1
                        return y1 + y2

                    (x_grad,) = paddle.grad(
                        forward_func(x), x, create_graph=True
                    )

                    self.assertEqual(
                        x_grad.shape,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.data_ptr(),
                        0,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.strides,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        str(x_grad.place),
                        str(x.place),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.dtype,
                        x.dtype,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.is_contiguous(),
                        x.is_contiguous(),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )

    def test_backward_static(self):
        """Test for simple API call"""

        def gradient_func(x):
            y1 = x / 2
            y2 = x + 1
            out = y1 + y2
            return paddle.grad(out, x)

        for place in self.places:
            paddle.device.set_device(place)
            for dtype in self.dtypes:
                for zero_size_shape in self.zero_size_shapes:
                    x = paddle.ones(zero_size_shape, dtype=dtype)
                    x.stop_gradient = False
                    self.assertEqual(x.data_ptr(), 0)

                    static_gradient_func = paddle.jit.to_static(
                        gradient_func, full_graph=True
                    )
                    (x_grad,) = static_gradient_func(x)

                    self.assertEqual(
                        x_grad.shape,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.data_ptr(),
                        0,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.strides,
                        zero_size_shape,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        str(x_grad.place),
                        str(x.place),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.dtype,
                        x.dtype,
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )
                    self.assertEqual(
                        x_grad.is_contiguous(),
                        x.is_contiguous(),
                        msg=f"Check failed at: {dtype}, {zero_size_shape}",
                    )


if __name__ == '__main__':
    unittest.main()
