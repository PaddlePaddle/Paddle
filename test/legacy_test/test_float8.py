# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_cuda_version, get_device, is_custom_device

import paddle
from paddle.base import core

# define the e4m3/e5m2 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0


def check_fp8_support() -> bool:
    """Return if fp8 support is available"""
    gpu_arch = (
        paddle.device.cuda.get_device_capability()[0] * 10
        + paddle.device.cuda.get_device_capability()[1]
    )
    if gpu_arch >= 90:  # hopper and above
        return True
    # Device compute capability 8.9 or higher required for FP8 execution.
    if gpu_arch < 89:  # pre-ada
        return False
    if get_cuda_version() < 12010:
        return False
    return True


class TestFP8CastOp(unittest.TestCase):
    def setUp(self):
        if paddle.framework.use_pir_api():
            self.dtype_dict = {
                "float8_e4m3fn": core.DataType.FLOAT8_E4M3FN,
                "float8_e5m2": core.DataType.FLOAT8_E5M2,
            }
        else:
            self.dtype_dict = {
                "float8_e4m3fn": core.VarDesc.VarType.FP8_E4M3FN,
                "float8_e5m2": core.VarDesc.VarType.FP8_E5M2,
            }
        self.shape = (16, 16)

    def test_cast(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            for self.device in ["cpu", get_device()]:
                paddle.device.set_device(self.device)
                for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                    # test fp32 to fp8 (dtype)
                    input = paddle.full(self.shape, 100000.0)
                    input1 = input.astype(self.dtype)
                    self.assertTrue(input1.dtype == self.dtype_dict[self.dtype])
                    # test fp8 to fp32 (dtype)
                    input2 = input1.astype("float32")
                    if paddle.framework.use_pir_api():
                        self.assertTrue(input2.dtype == core.DataType.FLOAT32)
                    else:
                        self.assertTrue(
                            input2.dtype == core.VarDesc.VarType.FP32
                        )
                    # test fp32 to fp8 (value clip)
                    expect = paddle.full(
                        self.shape,
                        (
                            E4M3_MAX_POS
                            if self.dtype == "float8_e4m3fn"
                            else E5M2_MAX_POS
                        ),
                    )
                    self.assertTrue(paddle.equal_all(input2, expect))
        else:
            self.device = "cpu"
            paddle.device.set_device(self.device)
            for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                # test fp32 to fp8 (dtype)
                input = paddle.full(self.shape, 100000.0)
                input1 = input.astype(self.dtype)
                self.assertTrue(input1.dtype == self.dtype_dict[self.dtype])
                # test fp8 to fp32 (dtype)
                input2 = input1.astype("float32")
                if paddle.framework.use_pir_api():
                    self.assertTrue(input2.dtype == core.DataType.FLOAT32)
                else:
                    self.assertTrue(input2.dtype == core.VarDesc.VarType.FP32)
                # test fp32 to fp8 (value clip)
                expect = paddle.full(
                    self.shape,
                    (
                        E4M3_MAX_POS
                        if self.dtype == "float8_e4m3fn"
                        else E5M2_MAX_POS
                    ),
                )
                self.assertTrue(paddle.equal_all(input2, expect))


class TestFP8FullOp(unittest.TestCase):
    def setUp(self):
        if paddle.framework.use_pir_api():
            self.dtype_dict = {
                "float8_e4m3fn": core.DataType.FLOAT8_E4M3FN,
                "float8_e5m2": core.DataType.FLOAT8_E5M2,
            }
        else:
            self.dtype_dict = {
                "float8_e4m3fn": core.VarDesc.VarType.FP8_E4M3FN,
                "float8_e5m2": core.VarDesc.VarType.FP8_E5M2,
            }

    def test_ones(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            for self.device in ["cpu", get_device()]:
                paddle.device.set_device(self.device)
                for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                    input = paddle.ones([1, 2], dtype=self.dtype)
                    self.assertTrue(input.dtype == self.dtype_dict[self.dtype])
                    input_fp32 = input.astype("float32")
                    expect = paddle.to_tensor([[1, 1]]).astype("float32")
                    self.assertTrue(paddle.equal_all(expect, input_fp32))
        else:
            self.device = "cpu"
            paddle.device.set_device(self.device)
            for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                input = paddle.ones([1, 2], dtype=self.dtype)
                self.assertTrue(input.dtype == self.dtype_dict[self.dtype])
                input_fp32 = input.astype("float32")
                expect = paddle.to_tensor([[1, 1]]).astype("float32")
                self.assertTrue(paddle.equal_all(expect, input_fp32))

    def test_zeros(self):
        if core.is_compiled_with_cuda() or is_custom_device():
            for self.device in ["cpu", get_device()]:
                paddle.device.set_device(self.device)
                for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                    input = paddle.zeros([1, 2], dtype=self.dtype)
                    self.assertTrue(input.dtype == self.dtype_dict[self.dtype])
                    input_fp32 = input.astype("float32")
                    expect = paddle.to_tensor([[0, 0]]).astype("float32")
                    self.assertTrue(paddle.equal_all(expect, input_fp32))
        else:
            self.device = "cpu"
            paddle.device.set_device(self.device)
            for self.dtype in ["float8_e4m3fn", "float8_e5m2"]:
                input = paddle.zeros([1, 2], dtype=self.dtype)
                self.assertTrue(input.dtype == self.dtype_dict[self.dtype])
                input_fp32 = input.astype("float32")
                expect = paddle.to_tensor([[0, 0]]).astype("float32")
                self.assertTrue(paddle.equal_all(expect, input_fp32))


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not check_fp8_support(),
    "Fp8 matmul requires CUDA >= 12.1 on Ada arch or hopper arch",
)
class TestFP8MatmulOp(unittest.TestCase):
    def gelu(self, x):
        return (
            0.5
            * x
            * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * np.power(x, 3))))
        )

    def setUp(self):
        self.dtype_dict = {
            "float8_e4m3fn": core.VarDesc.VarType.FP8_E4M3FN,
            "float8_e5m2": core.VarDesc.VarType.FP8_E5M2,
        }

    def test_matmul(self):
        for self.device in [get_device()]:
            paddle.device.set_device(self.device)
            for self.dtype in ["float8_e4m3fn"]:
                input1 = paddle.ones([4, 16, 32], dtype=self.dtype)
                input2 = paddle.ones([4, 64, 32], dtype=self.dtype)

                bias_fp16 = paddle.ones([64], dtype="float16")
                bias_bf16 = paddle.ones([64], dtype="bfloat16")

                input3 = np.ones((4, 64, 32)).astype("float32")
                input4 = np.ones((4, 32, 64)).astype("float32")

                bias_float32 = paddle.ones([64], dtype="float32")

                output_fp16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype="float16",
                )

                output_bf16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype="bfloat16",
                )

                output_bias_fp16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    bias=bias_fp16,
                    scale=1.0,
                    output_dtype="float16",
                )

                output_bias_bf16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    bias=bias_bf16,
                    scale=1.0,
                    output_dtype="bfloat16",
                )

                output_gelu_fp16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    scale=1.0,
                    act="gelu",
                    output_dtype="float16",
                )

                output_gelu_bf16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    scale=1.0,
                    act="gelu",
                    output_dtype="bfloat16",
                )

                output_bias_gelu_fp16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    bias=bias_fp16,
                    scale=1.0,
                    act="gelu",
                    output_dtype="float16",
                )

                output_bias_gelu_bf16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    bias=bias_bf16,
                    scale=1.0,
                    act="gelu",
                    output_dtype="bfloat16",
                )

                output_bias_relu_fp16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    bias=bias_fp16,
                    scale=1.0,
                    act="relu",
                    output_dtype="float16",
                )

                output_bias_relu_bf16 = paddle.linalg.fp8_fp8_half_gemm_fused(
                    input1,
                    input2,
                    transpose_x=False,
                    transpose_y=True,
                    bias=bias_bf16,
                    scale=1.0,
                    act="relu",
                    output_dtype="bfloat16",
                )

                expect_result = np.matmul(input3, input4)
                if self.device == "gpu":
                    self.assertTrue(
                        paddle.equal_all(
                            paddle.cast(output_fp16, "float32"),
                            paddle.cast(output_bf16, "float32"),
                            paddle.to_tensor(expect_result),
                        )
                    )
                    self.assertTrue(
                        paddle.equal_all(
                            paddle.cast(output_gelu_fp16, "float32"),
                            paddle.cast(output_gelu_bf16, "float32"),
                            paddle.to_tensor(self.gelu(expect_result)),
                        )
                    )
                    self.assertTrue(
                        paddle.equal_all(
                            paddle.cast(output_bias_fp16, "float32"),
                            paddle.cast(output_bias_bf16, "float32"),
                            paddle.to_tensor(expect_result + bias_float32),
                        )
                    )
                    self.assertTrue(
                        paddle.equal_all(
                            paddle.cast(output_bias_gelu_fp16, "float32"),
                            paddle.cast(output_bias_gelu_bf16, "float32"),
                            paddle.to_tensor(
                                self.gelu(expect_result) + bias_float32
                            ),
                        )
                    )
                    self.assertTrue(
                        paddle.equal_all(
                            paddle.cast(output_bias_relu_fp16, "float32"),
                            paddle.cast(output_bias_relu_bf16, "float32"),
                            paddle.to_tensor(
                                np.maximum(expect_result, 0) + bias_float32
                            ),
                        )
                    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "float8 kernels below are GPU only"
)
class TestFP8IndexAndCompareOp(unittest.TestCase):
    """Data-movement / comparison ops on float8 tensors (GPU)."""

    def setUp(self):
        paddle.disable_static()
        paddle.device.set_device(get_device())
        self.dtypes = ["float8_e4m3fn", "float8_e5m2"]
        # Values exactly representable in both float8 formats.
        self.x_fp32 = paddle.to_tensor(
            [
                [0.0, 1.0, -2.0, 4.0],
                [0.5, -0.25, 8.0, 16.0],
                [3.0, -1.5, 2.0, 1.0],
            ]
        )

    def to_fp32(self, t):
        return t.astype("float32").numpy()

    def test_equal_not_equal(self):
        for dtype in self.dtypes:
            x = self.x_fp32.astype(dtype)
            y = x.clone()
            y[0, 1] = 0.0
            np.testing.assert_array_equal(
                paddle.equal(x, y).numpy(),
                self.to_fp32(x) == self.to_fp32(y),
            )
            np.testing.assert_array_equal(
                (x != y).numpy(), self.to_fp32(x) != self.to_fp32(y)
            )
            # broadcast
            row = x[1:2]
            np.testing.assert_array_equal(
                (x == row).numpy(), self.to_fp32(x) == self.to_fp32(row)
            )

    def test_equal_all(self):
        for dtype in self.dtypes:
            x = self.x_fp32.astype(dtype)
            self.assertTrue(paddle.equal_all(x, x.clone()).item())
            y = x.clone()
            y[2, 3] = 0.0
            self.assertFalse(paddle.equal_all(x, y).item())
            self.assertFalse(paddle.equal_all(x, x[:2]).item())

    def test_index_get(self):
        idx = paddle.to_tensor([2, 0, 2])
        for dtype in self.dtypes:
            x = self.x_fp32.astype(dtype)
            out = x[idx]
            self.assertEqual(out.dtype, x.dtype)
            np.testing.assert_array_equal(
                self.to_fp32(out), self.to_fp32(x)[idx.numpy()]
            )
            mask = paddle.to_tensor([True, False, True])
            np.testing.assert_array_equal(
                self.to_fp32(x[mask]), self.to_fp32(x)[mask.numpy()]
            )

    def test_index_put(self):
        # NOTE: only the tensor-value form of advanced-index assignment is
        # covered. Scalar assignment (``x[idx] = 2.0`` / ``x[i, j] = 2.0``)
        # routes through ``set_value`` / scalar ``index_elementwise_put``,
        # which are not yet registered for float8 and are out of scope here.
        idx = paddle.to_tensor([0, 2])
        for dtype in self.dtypes:
            x = self.x_fp32.astype(dtype)
            expect = self.to_fp32(x)
            value = self.x_fp32[1:2].astype(dtype).expand([2, 4])
            x[idx] = value
            expect[idx.numpy()] = self.to_fp32(value)
            np.testing.assert_array_equal(self.to_fp32(x), expect)

    def test_masked_fill(self):
        mask = paddle.to_tensor([[True, False, False, True]]).expand([3, 4])
        for dtype in self.dtypes:
            x = self.x_fp32.astype(dtype)
            out = x.masked_fill(mask, 0.0)
            self.assertEqual(out.dtype, x.dtype)
            expect = self.to_fp32(x)
            expect[mask.numpy()] = 0.0
            np.testing.assert_array_equal(self.to_fp32(out), expect)

    def test_pad(self):
        for dtype in self.dtypes:
            x = self.x_fp32.astype(dtype)
            out = paddle.nn.functional.pad(
                x, [0, 2], mode='constant', value=0.0
            )
            self.assertEqual(out.dtype, x.dtype)
            expect = np.pad(self.to_fp32(x), ((0, 0), (0, 2)), mode='constant')
            np.testing.assert_array_equal(self.to_fp32(out), expect)


if __name__ == "__main__":
    unittest.main()
