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

import unittest

import numpy as np

import paddle
from paddle import _legacy_C_ops


class TestQuantizeLinerAPI(unittest.TestCase):
    """
    test for quantize_linear and dequantize_linear
    """

    def setUp(self):
        np.random.seed(2025)
        paddle.disable_static()

    def run_case(self, function_name, xshape, axis, bit_length, qmin, qmax):
        func = getattr(_legacy_C_ops, function_name, None)
        if func is None:
            raise ValueError(
                f"No function named '{function_name}' found in _legacy_C_ops."
            )

        x_np = np.random.uniform(-0.1, 0.1, xshape).astype("float32")

        x_paddle = paddle.to_tensor(
            x_np, dtype="float32", place=paddle.XPUPlace(0)
        )
        x_paddle_cpu = paddle.to_tensor(
            x_np, dtype="float32", place=paddle.CPUPlace()
        )

        zero_paddle = paddle.to_tensor(
            [0], dtype="float32", place=paddle.XPUPlace(0)
        )
        zero_paddle_cpu = paddle.to_tensor(
            [0], dtype="float32", place=paddle.CPUPlace()
        )

        if axis == -1:
            scale_paddle = paddle.to_tensor(
                [0.5], dtype="float32", place=paddle.XPUPlace(0)
            )
            scale_paddle_cpu = paddle.to_tensor(
                [0.5], dtype="float32", place=paddle.CPUPlace()
            )
        elif axis == 0:
            scale_np = np.random.uniform(-0.1, 0.1, xshape[0]).astype("float32")
            scale_paddle = paddle.to_tensor(
                scale_np, dtype="float32", place=paddle.XPUPlace(0)
            )
            scale_paddle_cpu = paddle.to_tensor(
                scale_np, dtype="float32", place=paddle.CPUPlace()
            )
        elif axis == 1:
            scale_np = np.random.uniform(-0.1, 0.1, xshape[1]).astype("float32")
            scale_paddle = paddle.to_tensor(
                scale_np, dtype="float32", place=paddle.XPUPlace(0)
            )
            scale_paddle_cpu = paddle.to_tensor(
                scale_np, dtype="float32", place=paddle.CPUPlace()
            )
        else:
            raise AssertionError(
                "quant axis other than -1, 0, 1 is not supported in XPU"
            )

        paddle.set_device("xpu")
        y_xpu = func(
            x_paddle,
            scale_paddle,
            zero_paddle,
            "quant_axis",
            axis,
            "bit_length",
            bit_length,
            "qmin",
            qmin,
            "qmax",
            qmax,
        )
        paddle.set_device("cpu")
        y_cpu = func(
            x_paddle_cpu,
            scale_paddle_cpu,
            zero_paddle_cpu,
            "quant_axis",
            axis,
            "bit_length",
            bit_length,
            "qmin",
            qmin,
            "qmax",
            qmax,
        )
        np.testing.assert_allclose(y_xpu.numpy(), y_cpu.numpy(), atol=0, rtol=0)

    def test_quantize(self):
        for axis in [-1, 0, 1]:
            self.run_case("quantize_linear", [3, 5], axis, 4, -8, 7)
            self.run_case("quantize_linear", [10, 12], axis, 4, -8, 7)
            self.run_case("quantize_linear", [10, 12], axis, 8, -128, 127)
            self.run_case("quantize_linear", [10, 12, 15], axis, 4, -8, 7)
            self.run_case("quantize_linear", [10, 12, 15], axis, 8, -128, 127)

    def test_dequantize(self):
        for axis in [-1, 0, 1]:
            self.run_case("dequantize_linear", [3, 5], axis, 4, -8, 7)
            self.run_case("dequantize_linear", [10, 12], axis, 4, -8, 7)
            self.run_case("dequantize_linear", [10, 12], axis, 8, -128, 127)
            self.run_case("dequantize_linear", [10, 12, 15], axis, 4, -8, 7)
            self.run_case("dequantize_linear", [10, 12, 15], axis, 8, -128, 127)


if __name__ == "__main__":
    unittest.main()
