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


class XPUTestStridedCopyKernelXPU(XPUOpTestWrapper):
    def __init__(self):
        # `strided_copy` is a phi kernel invoked by:
        #   - TransStride (stride-backup output copyback)
        #   - StridedTensorCopy (Tensor.copy_ when src/dst is non-contiguous)
        # It is not a standalone public op, so we use a "proxy" op name only to
        # follow the standard XPU test scaffolding and query supported dtypes.
        #
        # Use `cast` because it has a broad dtype coverage across XPU versions
        # and also reflects whether complex64 is enabled (depends on
        # `PADDLE_WITH_XPU_FFT`).
        self.op_name = "cast"

    class TestStridedCopyKernel(unittest.TestCase):
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

        def _assert_equal(self, got, expected):
            if self.in_type_str in [
                "float16",
                "float32",
                "float64",
                "complex64",
                "complex128",
            ]:
                np.testing.assert_allclose(
                    got, expected, atol=0.0, rtol=0.0, equal_nan=True
                )
            else:
                np.testing.assert_array_equal(got, expected)

        def _elem_bytes(self):
            return int(np.dtype(self.in_type).itemsize)

        def _to_tensor(self, np_array):
            # NOTE: for bfloat16, many XPU tests rely on an undocumented behavior
            # that `to_tensor(uint16)` constructs a bfloat16 tensor.
            return paddle.to_tensor(np_array)

        def _rand_np(self, shape, seed):
            rng = np.random.default_rng(seed)
            if self.in_type_str == "bool":
                return rng.integers(0, 2, size=shape).astype(np.bool_)
            if self.in_type_str == "bfloat16":
                data_fp32 = rng.standard_normal(shape).astype(np.float32)
                return convert_float_to_uint16(data_fp32)
            if self.in_type_str in ["float16", "float32", "float64"]:
                return rng.standard_normal(shape).astype(
                    getattr(np, self.in_type_str)
                )
            if self.in_type_str in ["int8", "int16", "int32", "int64"]:
                return rng.integers(-50, 50, size=shape).astype(
                    getattr(np, self.in_type_str)
                )
            if self.in_type_str == "uint8":
                return rng.integers(0, 256, size=shape).astype(np.uint8)
            if self.in_type_str == "complex64":
                real = rng.standard_normal(shape).astype(np.float32)
                imag = rng.standard_normal(shape).astype(np.float32)
                return (real + 1j * imag).astype(np.complex64)
            if self.in_type_str == "complex128":
                real = rng.standard_normal(shape).astype(np.float64)
                imag = rng.standard_normal(shape).astype(np.float64)
                return (real + 1j * imag).astype(np.complex128)
            self.skipTest(
                f"Unsupported dtype for this test: {self.in_type_str}"
            )

        def _fixed_np(self, shape):
            if self.in_type_str == "bool":
                data = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(
                    shape
                )
                return (data % 2).astype(np.bool_)
            if self.in_type_str == "bfloat16":
                data_fp32 = np.arange(
                    int(np.prod(shape)), dtype=np.float32
                ).reshape(shape) - np.float32(7.0)
                return convert_float_to_uint16(data_fp32)
            if self.in_type_str in ["float16", "float32", "float64"]:
                return (
                    np.arange(int(np.prod(shape)), dtype=np.float32).reshape(
                        shape
                    )
                    - np.float32(7.0)
                ).astype(getattr(np, self.in_type_str))
            if self.in_type_str in ["int8", "int16", "int32", "int64"]:
                data = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(
                    shape
                )
                return (data - 7).astype(getattr(np, self.in_type_str))
            if self.in_type_str == "uint8":
                data = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(
                    shape
                )
                return (data % 251).astype(np.uint8)
            if self.in_type_str == "complex64":
                real = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(
                    shape
                ) - np.float32(7.0)
                imag = real + np.float32(0.5)
                return (
                    real.astype(np.complex64) + 1j * imag.astype(np.complex64)
                ).astype(np.complex64)
            if self.in_type_str == "complex128":
                real = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(
                    shape
                ) - np.float64(7.0)
                imag = real + np.float64(0.5)
                return (
                    real.astype(np.complex128) + 1j * imag.astype(np.complex128)
                ).astype(np.complex128)
            self.skipTest(
                f"Unsupported dtype for this test: {self.in_type_str}"
            )

        def test_fixed_numel_0_and_1(self):
            # Cover:
            #   - input.numel() == 0 early-return
            #   - input.numel() == 1 fast copy path
            dtype = self.in_type_str

            # numel == 0 (force a non-contiguous zero-numel view)
            base0 = paddle.zeros([2, 2], dtype=dtype)
            out0 = paddle.as_strided(
                base0, shape=[0, 2], stride=[1, 1], offset=0
            )
            self.assertFalse(out0.is_contiguous())
            src0 = paddle.zeros([0, 2], dtype=dtype)
            out0.copy_(src0)
            self.assertEqual(out0.numel(), 0)
            self.assertEqual(list(out0.shape), [0, 2])

            # numel == 1
            base1 = paddle.zeros([2], dtype=dtype)
            out1 = paddle.as_strided(base1, shape=[1], stride=[2], offset=0)
            self.assertFalse(out1.is_contiguous())
            if dtype == "complex64":
                src_np = np.array([1.25 + 2.5j], dtype=np.complex64)
            elif dtype == "complex128":
                src_np = np.array([1.25 + 2.5j], dtype=np.complex128)
            elif dtype == "bfloat16":
                src_np = convert_float_to_uint16(
                    np.array([1.25], dtype=np.float32)
                )
            elif dtype in ["float16", "float32", "float64"]:
                src_np = np.array([1.25], dtype=getattr(np, dtype))
            else:
                src_np = np.array([7], dtype=self.in_type)
            src = self._to_tensor(src_np)
            paddle.assign(src, output=out1)
            self._assert_equal(out1.numpy(), src_np)

        def test_fixed_assign_out_to_strided_view(self):
            # Cover `xpu::strided_copy` on common strided-view patterns.
            dtype = self.in_type_str
            elem_bytes = self._elem_bytes()

            cases = [
                # (base_w, view_w, stride1, offset_elems)
                (256, 64, 1, 0),
                (256, 64, 1, 64),
                (256, 64, 2, 0),
                (256, 64, 2, 1),
            ]
            for base_w, view_w, stride1, offset_elems in cases:
                offset_bytes = offset_elems * elem_bytes
                base = paddle.zeros([2, base_w], dtype=dtype)
                out = paddle.as_strided(
                    base,
                    shape=[2, view_w],
                    stride=[base_w, stride1],
                    offset=offset_bytes,
                )
                self.assertFalse(out.is_contiguous())

                src_np = self._fixed_np([2, view_w])
                src = self._to_tensor(src_np)

                paddle.assign(src, output=out)
                self._assert_equal(out.numpy(), src_np)

        def test_fixed_copy_src_view_to_dst_contiguous(self):
            # Cover input-strided copy path via Tensor.copy_:
            #   src is non-contiguous view, dst is contiguous tensor.
            dtype = self.in_type_str
            elem_bytes = self._elem_bytes()

            for stride1 in [1, 2]:
                src_base_np = self._fixed_np([2, 256])
                src_base = self._to_tensor(src_base_np)

                offset_elems = 96
                offset_bytes = offset_elems * elem_bytes
                src_view = paddle.as_strided(
                    src_base,
                    shape=[2, 64],
                    stride=[256, stride1],
                    offset=offset_bytes,
                )
                self.assertFalse(src_view.is_contiguous())

                dst = paddle.empty([2, 64], dtype=dtype)
                dst.copy_(src_view)

                expected = src_base_np[
                    :, offset_elems : offset_elems + stride1 * 64 : stride1
                ]
                self._assert_equal(dst.numpy(), expected)

        def test_random_assign_and_copy_strided_views(self):
            # Randomized coverage for:
            #   - different shapes
            #   - different offsets
            #   - last-dim stride = 1 / 2
            dtype = self.in_type_str
            elem_bytes = self._elem_bytes()
            rng = np.random.default_rng(2025)
            for i in range(6):
                batch = int(rng.integers(1, 4))
                base_w = int(rng.integers(16, 192))
                stride1 = int(rng.choice([1, 2]))
                max_view_w = min(64, (base_w + stride1 - 1) // stride1)
                view_w = int(rng.integers(1, max_view_w + 1))
                max_offset = base_w - (view_w - 1) * stride1 - 1
                offset_elems = int(rng.integers(0, max_offset + 1))
                offset_bytes = offset_elems * elem_bytes

                # assign_out_ -> TransStride -> StridedCopyKernel
                base = paddle.zeros([batch, base_w], dtype=dtype)
                out = paddle.as_strided(
                    base,
                    shape=[batch, view_w],
                    stride=[base_w, stride1],
                    offset=offset_bytes,
                )
                self.assertFalse(out.is_contiguous())
                src_np = self._rand_np([batch, view_w], seed=1000 + i)
                src = self._to_tensor(src_np)
                paddle.assign(src, output=out)
                self._assert_equal(out.numpy(), src_np)

                # Tensor.copy_ -> StridedTensorCopy -> StridedCopyKernel
                src_base_np = self._rand_np([batch, base_w], seed=2000 + i)
                src_base = self._to_tensor(src_base_np)
                src_view = paddle.as_strided(
                    src_base,
                    shape=[batch, view_w],
                    stride=[base_w, stride1],
                    offset=offset_bytes,
                )
                self.assertFalse(src_view.is_contiguous())
                dst = paddle.empty([batch, view_w], dtype=dtype)
                dst.copy_(src_view)
                expected = src_base_np[
                    :, offset_elems : offset_elems + stride1 * view_w : stride1
                ]
                self._assert_equal(dst.numpy(), expected)

        def test_copy_different_dims_same_numel(self):
            # Regression for XPU StridedCopyKernel:
            # - strided_copy does not require input.dims == output.dims
            # - only requires numel match
            dtype = self.in_type_str

            src_np = self._fixed_np([2, 6])
            src = self._to_tensor(src_np)

            base = paddle.zeros([3, 8], dtype=dtype)
            dst_view = paddle.as_strided(
                base, shape=[3, 4], stride=[8, 2], offset=0
            )
            self.assertFalse(dst_view.is_contiguous())

            dst_view.copy_(src)
            expected = src_np.reshape([3, 4])
            self._assert_equal(dst_view.numpy(), expected)

        def test_complex_misaligned_offset_bytes(self):
            # Regression for complex specialization:
            # - CopyT is float32, so it should not require 8-byte alignment.
            # - The kernel should work even if the complex view has a 4-byte
            #   holder offset (valid alignment for float32).
            if self.in_type_str not in ["complex64", "complex128"]:
                self.skipTest("complex-only.")

            dtype = self.in_type_str

            base = paddle.zeros([2, 4], dtype=dtype)
            dst_view = paddle.as_strided(
                base, shape=[2, 3], stride=[4, 1], offset=4
            )
            self.assertFalse(dst_view.is_contiguous())

            src_np = self._rand_np([2, 3], seed=4242)
            src = self._to_tensor(src_np)
            dst_view.copy_(src)
            self._assert_equal(dst_view.numpy(), src_np)

        def test_complex_transpose_and_strided_views(self):
            # Cover complex64/complex128 XPU specialization:
            # - numel > 1 -> xpu::strided_copy path
            # - transpose-like strides (a common non-contiguous pattern)
            if self.in_type_str not in ["complex64", "complex128"]:
                self.skipTest("complex-only.")

            dtype = self.in_type_str
            elem_bytes = self._elem_bytes()

            # Copy from a transpose-like strided view -> contiguous dst.
            base_np = self._fixed_np([2, 3])
            base = self._to_tensor(base_np)
            src_view = paddle.as_strided(
                base, shape=[3, 2], stride=[1, 3], offset=0
            )
            self.assertFalse(src_view.is_contiguous())
            dst = paddle.empty([3, 2], dtype=dtype)
            dst.copy_(src_view)
            expected = base_np.transpose(1, 0)
            self._assert_equal(dst.numpy(), expected)

            # Copy to a transpose-like strided dst view from contiguous src.
            src_np = self._fixed_np([3, 2])
            src = self._to_tensor(src_np)
            base_dst = paddle.zeros([2, 3], dtype=dtype)
            dst_view = paddle.as_strided(
                base_dst, shape=[3, 2], stride=[1, 3], offset=0
            )
            self.assertFalse(dst_view.is_contiguous())
            dst_view.copy_(src)
            self._assert_equal(dst_view.numpy(), src_np)

            # Randomized transpose-like patterns (with random offsets), and cover:
            # - strided(src) -> contiguous(dst)
            # - contiguous(src) -> strided(dst)
            # - strided(src) -> strided(dst)
            rng = np.random.default_rng(2027)
            for i in range(5):
                base_m = int(rng.integers(2, 9))
                base_n = int(rng.integers(2, 9))
                view_m = int(rng.integers(1, base_m + 1))
                view_n = int(rng.integers(1, base_n + 1))
                r0 = int(rng.integers(0, base_m - view_m + 1))
                c0 = int(rng.integers(0, base_n - view_n + 1))
                offset_elems = r0 * base_n + c0
                offset_bytes = offset_elems * elem_bytes

                # src(strided transpose view) -> dst(contiguous)
                base_np = self._rand_np([base_m, base_n], seed=5000 + i)
                base = self._to_tensor(base_np)
                src_view = paddle.as_strided(
                    base,
                    shape=[view_n, view_m],
                    stride=[1, base_n],
                    offset=offset_bytes,
                )
                self.assertFalse(src_view.is_contiguous())
                dst = paddle.empty([view_n, view_m], dtype=dtype)
                dst.copy_(src_view)
                expected = base_np[
                    r0 : r0 + view_m, c0 : c0 + view_n
                ].transpose(1, 0)
                self._assert_equal(dst.numpy(), expected)

                # src(contiguous) -> dst(strided transpose view)
                src_np = self._rand_np([view_n, view_m], seed=6000 + i)
                src = self._to_tensor(src_np)
                base_dst = paddle.zeros([base_m, base_n], dtype=dtype)
                dst_view = paddle.as_strided(
                    base_dst,
                    shape=[view_n, view_m],
                    stride=[1, base_n],
                    offset=offset_bytes,
                )
                self.assertFalse(dst_view.is_contiguous())
                dst_view.copy_(src)
                base_dst_np = base_dst.numpy()
                expected_region = src_np.transpose(1, 0)
                got_region = base_dst_np[r0 : r0 + view_m, c0 : c0 + view_n]
                self._assert_equal(got_region, expected_region)

                # strided -> strided (same layout/offset), validate via base tensors
                src_base_np = self._rand_np([base_m, base_n], seed=7000 + i)
                src_base = self._to_tensor(src_base_np)
                src_view = paddle.as_strided(
                    src_base,
                    shape=[view_n, view_m],
                    stride=[1, base_n],
                    offset=offset_bytes,
                )
                dst_base = paddle.zeros([base_m, base_n], dtype=dtype)
                dst_view = paddle.as_strided(
                    dst_base,
                    shape=[view_n, view_m],
                    stride=[1, base_n],
                    offset=offset_bytes,
                )
                self.assertFalse(src_view.is_contiguous())
                self.assertFalse(dst_view.is_contiguous())
                dst_view.copy_(src_view)
                dst_base_np = dst_base.numpy()
                expected_region = src_base_np[
                    r0 : r0 + view_m, c0 : c0 + view_n
                ]
                got_region = dst_base_np[r0 : r0 + view_m, c0 : c0 + view_n]
                self._assert_equal(got_region, expected_region)

        def test_complex_transpose_unit_loop_counterexample(self):
            # Counterexample for an incorrect "no-unit-loop" idea:
            # If one tries to copy complex payload via a single float32
            # strided_copy by only tweaking strides (e.g., not multiplying the
            # last-dim stride by kCopyUnitsPerElem), transpose-like strides
            # would mix real/imag and produce wrong results.
            #
            # This test locks in the expected correct behavior on a tiny,
            # easy-to-inspect transpose view.
            if self.in_type_str not in ["complex64", "complex128"]:
                self.skipTest("complex-only.")

            dtype = self.in_type_str
            np_dtype = np.complex64 if dtype == "complex64" else np.complex128

            base_np = np.array(
                [
                    [0 + 100j, 1 + 101j, 2 + 102j],
                    [3 + 103j, 4 + 104j, 5 + 105j],
                ],
                dtype=np_dtype,
            )
            base = self._to_tensor(base_np)

            # B = A.T view:
            # - base shape [2, 3] -> view shape [3, 2]
            # - transpose-like stride [1, 3] (in complex elements)
            src_view = paddle.as_strided(
                base, shape=[3, 2], stride=[1, 3], offset=0
            )
            self.assertFalse(src_view.is_contiguous())

            dst = paddle.empty([3, 2], dtype=dtype)
            dst.copy_(src_view)
            expected = base_np.transpose(1, 0)
            self._assert_equal(dst.numpy(), expected)


# NOTE:
# `strided_copy` is a phi internal kernel and is not present in XPU op lists,
# so `get_xpu_op_support_types("strided_copy")` would be empty and no tests
# would be generated. We therefore query supported dtypes through a "proxy" op.
#
# `cast` is chosen because it has broad dtype coverage across XPU versions and
# also reflects whether complex64/complex128 are enabled in this build (depends
# on `PADDLE_WITH_XPU_FFT`).
# Use `cast` as the main proxy for scalar/real dtypes, and `real` as a proxy
# for complex dtypes (complex64/complex128 are not necessarily present in
# cast's support list on all XPU builds).
support_types = get_xpu_op_support_types("cast")
complex_support_types = get_xpu_op_support_types("real")
has_complex_support = (
    "complex64" in complex_support_types
    or "complex128" in complex_support_types
)
for stype in [
    "bool",
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "uint8",
    "int16",
    "int32",
    "int64",
    "complex64",
    "complex128",
]:
    if stype in support_types or (
        stype in ["complex64", "complex128"] and has_complex_support
    ):
        create_test_class(
            globals(),
            XPUTestStridedCopyKernelXPU,
            stype,
            test_grad=False,
        )


if __name__ == "__main__":
    unittest.main()
