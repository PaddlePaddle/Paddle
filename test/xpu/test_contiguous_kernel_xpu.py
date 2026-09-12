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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16

import paddle

paddle.enable_static()


class XPUTestContiguousKernelXPU(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "cast"

    class TestContiguousKernel(unittest.TestCase):
        def setUp(self):
            if not paddle.is_compiled_with_xpu():
                self.skipTest("Paddle is not compiled with XPU.")

            self._orig_device = paddle.device.get_device()
            self.addCleanup(lambda: paddle.set_device(self._orig_device))
            self._orig_stride_flag = paddle.get_flags(
                ["FLAGS_use_stride_kernel"]
            )["FLAGS_use_stride_kernel"]

            def _restore_flags():
                if self._orig_stride_flag is not None:
                    paddle.set_flags(
                        {"FLAGS_use_stride_kernel": self._orig_stride_flag}
                    )

            self.addCleanup(_restore_flags)

            paddle.disable_static()
            self.addCleanup(paddle.enable_static)

            paddle.set_device("xpu:0")
            paddle.set_flags({"FLAGS_use_stride_kernel": 1})

        def _elem_bytes(self):
            return int(np.dtype(self.in_type).itemsize)

        def _assert_equal(self, got, expected):
            if self.in_type_str in ["float16", "float32", "float64"]:
                np.testing.assert_allclose(
                    got, expected, atol=0.0, rtol=0.0, equal_nan=True
                )
            else:
                np.testing.assert_array_equal(got, expected)

        def _to_tensor(self, np_array):
            return paddle.to_tensor(np_array)

        def _make_base_1d_np(self):
            if self.in_type_str == "bool":
                return np.array([True, False, True, False], dtype=np.bool_)
            if self.in_type_str == "bfloat16":
                data_fp32 = np.array([1, 2, 3, 4], dtype=np.float32)
                return convert_float_to_uint16(data_fp32)
            if self.in_type_str in ["float16", "float32", "float64"]:
                return np.array(
                    [1, 2, 3, 4], dtype=getattr(np, self.in_type_str)
                )
            if self.in_type_str in ["int8", "int16", "int32", "int64"]:
                return np.array(
                    [1, -2, 3, -4], dtype=getattr(np, self.in_type_str)
                )
            if self.in_type_str == "uint8":
                return np.array([1, 2, 3, 255], dtype=np.uint8)
            self.skipTest(
                f"Unsupported dtype for this test: {self.in_type_str}"
            )

        def _make_base_2d_np(self, shape):
            size = int(np.prod(shape))
            if self.in_type_str == "bool":
                data = (np.arange(size, dtype=np.int64) % 2 == 0).reshape(shape)
                return data.astype(np.bool_)
            if self.in_type_str == "bfloat16":
                data_fp32 = np.arange(size, dtype=np.float32).reshape(shape)
                return convert_float_to_uint16(data_fp32)
            if self.in_type_str in ["float16", "float32", "float64"]:
                return np.arange(
                    size, dtype=getattr(np, self.in_type_str)
                ).reshape(shape)
            if self.in_type_str in ["int8", "int16", "int32", "int64"]:
                return np.arange(
                    size, dtype=getattr(np, self.in_type_str)
                ).reshape(shape)
            if self.in_type_str == "uint8":
                return np.arange(size, dtype=np.uint8).reshape(shape)
            self.skipTest(
                f"Unsupported dtype for this test: {self.in_type_str}"
            )

        def test_numel_0(self):
            base0 = paddle.empty([2, 2], dtype=self.in_type_str)
            v0 = paddle.as_strided(base0, shape=[0, 2], stride=[1, 1], offset=0)
            out0 = v0.contiguous()
            self.assertTrue(out0.is_contiguous())
            self.assertEqual(out0.numel(), 0)
            self.assertEqual(list(out0.shape), [0, 2])

        def test_numel_1(self):
            # Cover: input.numel() == 1 -> xpu::copy
            base_np = self._make_base_1d_np()
            base = self._to_tensor(base_np)
            v1 = paddle.as_strided(base, shape=[1], stride=[2], offset=0)
            self.assertFalse(v1.is_contiguous())
            out1 = v1.contiguous()
            self.assertTrue(out1.is_contiguous())
            self._assert_equal(out1.numpy(), base_np[0:1])

        def test_numel_gt1(self):
            # Cover: input.numel() > 1 -> xpu::as_strided
            base_np = self._make_base_2d_np([2, 256])
            base = self._to_tensor(base_np)

            offset_elems = 10
            offset_bytes = offset_elems * self._elem_bytes()
            v = paddle.as_strided(
                base,
                shape=[2, 64],
                stride=[256, 1],
                offset=offset_bytes,
            )
            self.assertFalse(v.is_contiguous())
            out = v.contiguous()
            self.assertTrue(out.is_contiguous())
            self._assert_equal(
                out.numpy(), base_np[:, offset_elems : offset_elems + 64]
            )


class XPUTestContiguousComplexKernelXPU(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "real"

    class TestContiguousComplexKernel(unittest.TestCase):
        def setUp(self):
            if not paddle.is_compiled_with_xpu():
                self.skipTest("Paddle is not compiled with XPU.")

            self._orig_device = paddle.device.get_device()
            self.addCleanup(lambda: paddle.set_device(self._orig_device))
            self._orig_stride_flag = paddle.get_flags(
                ["FLAGS_use_stride_kernel"]
            )["FLAGS_use_stride_kernel"]

            def _restore_flags():
                if self._orig_stride_flag is not None:
                    paddle.set_flags(
                        {"FLAGS_use_stride_kernel": self._orig_stride_flag}
                    )

            self.addCleanup(_restore_flags)
            paddle.disable_static()
            self.addCleanup(paddle.enable_static)

            paddle.set_device("xpu:0")
            paddle.set_flags({"FLAGS_use_stride_kernel": 1})

        def _assert_allclose(self, got, expected):
            np.testing.assert_allclose(
                got, expected, atol=0.0, rtol=0.0, equal_nan=True
            )

        def test_complex_numel_0_and_1(self):
            if self.in_type_str not in ["complex64", "complex128"]:
                self.skipTest("complex-only.")

            dtype = self.in_type_str
            base0 = paddle.empty([2, 2], dtype=dtype)
            v0 = paddle.as_strided(base0, shape=[0, 2], stride=[1, 1], offset=0)
            self.assertFalse(
                v0.is_contiguous(), msg="expect non-contiguous view"
            )
            out0 = v0.contiguous()
            self.assertTrue(out0.is_contiguous())
            self.assertEqual(out0.numel(), 0)
            self.assertEqual(list(out0.shape), [0, 2])

            if dtype == "complex64":
                base_np = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
                expected_np = np.array([1.0 + 2.0j], dtype=np.complex64)
            else:
                base_np = np.array(
                    [1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex128
                )
                expected_np = np.array([1.0 + 2.0j], dtype=np.complex128)

            base = paddle.to_tensor(base_np)
            v1 = paddle.as_strided(base, shape=[1], stride=[2], offset=0)
            self.assertFalse(v1.is_contiguous())
            out1 = v1.contiguous()
            self.assertTrue(out1.is_contiguous())
            self._assert_allclose(out1.numpy(), expected_np)

        def test_complex_strided_slice_view_regression(self):
            if self.in_type_str not in ["complex64", "complex128"]:
                self.skipTest("complex-only.")

            if self.in_type_str == "complex64":
                real_dtype = np.float32
                complex_dtype = np.complex64
                imag_bias = np.float32(123.0)
                bsz, total_len, n1, n2 = 2, 32768, 1, 64
                start, end = 16384, 20480
            else:
                real_dtype = np.float64
                complex_dtype = np.complex128
                imag_bias = np.float64(123.0)
                bsz, total_len, n1, n2 = 2, 1024, 1, 64
                start, end = 512, 768

            real_np = np.arange(
                bsz * total_len * n1 * n2, dtype=real_dtype
            ).reshape([bsz, total_len, n1, n2])
            imag_np = real_np + imag_bias
            z_np = real_np.astype(complex_dtype) + 1j * imag_np.astype(
                complex_dtype
            )
            expected = z_np[:, start:end, :, :]

            z = paddle.to_tensor(z_np)

            stride0 = total_len * n1 * n2
            stride1 = n1 * n2
            stride2 = n2
            stride3 = 1
            offset_elems = start * n1 * n2
            offset_bytes = offset_elems * np.dtype(complex_dtype).itemsize
            t = paddle.as_strided(
                z,
                shape=[bsz, end - start, n1, n2],
                stride=[stride0, stride1, stride2, stride3],
                offset=offset_bytes,
            )
            self.assertFalse(t.is_contiguous())
            self._assert_allclose(t.numpy(), expected)
            self._assert_allclose(t.contiguous().numpy(), expected)
            self._assert_allclose(t.cpu().numpy(), expected)

        def test_complex_transpose_contiguous_regression(self):
            if self.in_type_str not in ["complex64", "complex128"]:
                self.skipTest("complex-only.")

            if self.in_type_str == "complex64":
                real_dtype = np.float32
                complex_dtype = np.complex64
                imag_bias = np.float32(100.0)
            else:
                real_dtype = np.float64
                complex_dtype = np.complex128
                imag_bias = np.float64(100.0)

            real_np = np.arange(6, dtype=real_dtype).reshape([2, 3])
            imag_np = real_np + imag_bias
            x_np = real_np.astype(complex_dtype) + 1j * imag_np.astype(
                complex_dtype
            )
            expected = x_np.transpose(1, 0)

            x = paddle.to_tensor(x_np)
            y = x.transpose([1, 0])
            self.assertFalse(y.is_contiguous())

            y_contig = y.contiguous()
            self.assertTrue(y_contig.is_contiguous())
            self._assert_allclose(y_contig.cpu().numpy(), expected)
            self._assert_allclose(y.cpu().numpy(), expected)


support_types_cast = get_xpu_op_support_types("cast")
support_types_real = get_xpu_op_support_types("real")

for stype in support_types_cast:
    if stype == "complex64":
        continue
    create_test_class(
        globals(),
        XPUTestContiguousKernelXPU,
        stype,
        test_grad=False,
    )

for stype in ["complex64", "complex128"]:
    if stype in support_types_real:
        create_test_class(
            globals(),
            XPUTestContiguousComplexKernelXPU,
            stype,
            test_grad=False,
        )


if __name__ == "__main__":
    unittest.main()
