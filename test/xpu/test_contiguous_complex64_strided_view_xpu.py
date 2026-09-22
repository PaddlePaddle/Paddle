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

import paddle
import paddle.distributed as dist

paddle.enable_static()


class XPUTestContiguousComplex64StridedViewXPU(XPUOpTestWrapper):
    def __init__(self):
        # `contiguous` is a phi kernel invoked by Trans2Contiguous/Copy, not a
        # standalone fluid op. We use `slice` as `op_name` only to follow the
        # standard XPU test scaffolding and to query supported dtypes.
        self.op_name = "slice"

    class TestContiguousKernelBranches(unittest.TestCase):
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

            # Use dygraph APIs (`Tensor.contiguous()` / `Tensor.cpu()`) to
            # exercise `paddle/phi/kernels/xpu/contiguous_kernel.cc`.
            paddle.disable_static()
            self.addCleanup(paddle.enable_static)

            paddle.set_device("xpu:0")
            paddle.set_flags({"FLAGS_use_stride_kernel": 1})

        def _assert_allclose(self, got, expected):
            np.testing.assert_allclose(
                got, expected, atol=0.0, rtol=0.0, equal_nan=True
            )

        def test_template_numel_0_1_gt1(self):
            # Cover template branches in `paddle/phi/kernels/xpu/contiguous_kernel.cc`
            # for a common dtype (float32).
            #   - out->numel() == 0
            #   - input.numel() == 1  -> xpu::copy
            #   - input.numel() > 1   -> xpu::as_strided
            if self.in_type_str != "float32":
                self.skipTest(
                    "Template-branch coverage is done on float32 only."
                )

            # out->numel() == 0
            # NOTE: some ops may treat zero-size results as already contiguous.
            # Here we use `as_strided` to force a zero-numel *non-contiguous*
            # view so that `Tensor.contiguous()` actually dispatches to the
            # contiguous kernel and hits the early-return branch.
            base0 = paddle.empty([2, 2], dtype="float32")
            v0 = paddle.as_strided(base0, shape=[0, 2], stride=[1, 1], offset=0)
            self.assertFalse(
                v0.is_contiguous(), msg="expect non-contiguous view"
            )
            out0 = v0.contiguous()
            self.assertTrue(out0.is_contiguous())
            self.assertEqual(out0.numel(), 0)
            self.assertEqual(list(out0.shape), [0, 2])

            # input.numel() == 1
            base1_np = np.array([10, 20, 30, 40], dtype=np.float32)
            base1 = paddle.to_tensor(base1_np)
            v1 = paddle.as_strided(base1, shape=[1], stride=[2], offset=0)
            self.assertFalse(v1.is_contiguous())
            out1 = v1.contiguous()
            self.assertTrue(out1.is_contiguous())
            self._assert_allclose(out1.numpy(), base1_np[0:1])

            # input.numel() > 1
            x_np = np.arange(2 * 256, dtype=np.float32).reshape([2, 256])
            x = paddle.to_tensor(x_np)
            v = paddle.as_strided(x, shape=[2, 64], stride=[256, 1], offset=0)
            self.assertFalse(v.is_contiguous())
            out = v.contiguous()
            self.assertTrue(out.is_contiguous())
            self._assert_allclose(out.numpy(), x_np[:, :64])

        def test_complex64_numel_0_and_1(self):
            # Cover complex64 specialization branches:
            #   - out->numel() == 0
            #   - input.numel() == 1  -> bytes xpu::copy<int8_t>
            if self.in_type_str != "complex64":
                self.skipTest("complex64-specific branches only.")

            # See the note in `test_template_numel_0_1_gt1` for why `as_strided`
            # is used for the zero-numel branch.
            base0 = paddle.empty([2, 2], dtype="complex64")
            v0 = paddle.as_strided(base0, shape=[0, 2], stride=[1, 1], offset=0)
            self.assertFalse(
                v0.is_contiguous(), msg="expect non-contiguous view"
            )
            out0 = v0.contiguous()
            self.assertTrue(out0.is_contiguous())
            self.assertEqual(out0.numel(), 0)
            self.assertEqual(list(out0.shape), [0, 2])

            base = paddle.to_tensor(
                np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
            )
            v1 = paddle.as_strided(base, shape=[1], stride=[2], offset=0)
            self.assertFalse(v1.is_contiguous())
            out1 = v1.contiguous()
            self.assertTrue(out1.is_contiguous())
            self._assert_allclose(
                out1.numpy(), np.array([1.0 + 2.0j], dtype=np.complex64)
            )

        def test_negative_stride_contiguous(self):
            # The XDNN gather/scatter primitives only walk an operand
            # forwards, so reversed views cannot be materialized or indexed
            # on XPU. Creating the view is metadata-only and must keep
            # working; every consuming kernel must raise instead of silently
            # returning wrong data.
            x_np = np.arange(8 * 6, dtype=np.float32).reshape([8, 6])
            if self.in_type_str == "complex64":
                x_np = x_np.astype(np.complex64) * (1.0 + 2.0j)
            x = paddle.to_tensor(x_np)

            for view_fn in (lambda t: t[::-1], lambda t: t[:, ::-1]):
                view = view_fn(x)
                self.assertFalse(view.is_contiguous())
                with self.assertRaises(NotImplementedError):
                    view.contiguous()

            if self.in_type_str != "float32":
                return
            idx = paddle.to_tensor(np.array([2, 5, 2, 0, 5], dtype=np.int64))
            # single __getitem__: the negative strides reach the gather
            # kernel through its stride attributes
            with self.assertRaises(NotImplementedError):
                x[::-1, idx]
            # chained __getitem__: the getitem fallback materializes the
            # reversed view first
            with self.assertRaises(NotImplementedError):
                x[:, ::-1][idx]
            # setitem: the scatter kernel walks the same reversed view
            with self.assertRaises(NotImplementedError):
                x[::-1, idx] = -7

        def test_complex64_strided_slice_regression(self):
            # Regression for: XPU complex64 strided-view materialization bug.
            #
            # Pre-fix symptom:
            #   - `t.numpy()` is correct (numpy view by strides/offset)
            #   - `t.cpu().numpy()` and `t.contiguous().numpy()` show ~50% mismatch
            #     (batch1 all wrong), because Trans2Contiguous used an invalid
            #     float real/imag materialization path for strided complex view.
            if self.in_type_str != "complex64":
                self.skipTest("complex64 regression only.")

            bsz, total_len, n1, n2 = 2, 32768, 1, 64
            start, end = 16384, 20480

            real_np = np.arange(
                bsz * total_len * n1 * n2, dtype=np.float32
            ).reshape([bsz, total_len, n1, n2])
            imag_np = real_np + np.float32(123.0)
            z_np = real_np.astype(np.complex64) + 1j * imag_np.astype(
                np.complex64
            )
            expected = z_np[:, start:end, :, :]

            z = paddle.to_tensor(z_np)

            t = paddle.slice(z, axes=[1], starts=[start], ends=[end])
            self.assertFalse(t.is_contiguous())

            # `numpy()` path should always be correct for the view itself.
            self._assert_allclose(t.numpy(), expected)

            # Both paths below trigger Trans2Contiguous -> XPU contiguous kernel.
            self._assert_allclose(t.contiguous().numpy(), expected)
            self._assert_allclose(t.cpu().numpy(), expected)

        def test_negative_stride_dist_tensor_advanced_index(self):
            if self.in_type_str != "float32":
                self.skipTest("DistTensor regression is covered on float32.")

            x_np = np.arange(8 * 6 * 6, dtype=np.float32).reshape([8, 6, 6])
            x = paddle.to_tensor(x_np)
            mesh = dist.ProcessMesh([0], dim_names=["x"])
            dist_x = dist.shard_tensor(x, mesh, [dist.Replicate()])
            idx_np = np.array([2, 5, 2, 0, 5], dtype=np.int64)
            idx = paddle.to_tensor(idx_np)

            # DistTensor strided_slice materializes the local result before
            # advanced indexing, so the XPU gather receives contiguous data.
            cases = (
                (lambda t, i: t[i, ::-1, :], lambda a: a[idx_np, ::-1, :]),
                (lambda t, i: t[::-1, i], lambda a: a[::-1, idx_np]),
                (
                    lambda t, i: t[::-1, i, ::-1],
                    lambda a: a[::-1, idx_np, ::-1],
                ),
            )
            for fn, expected_fn in cases:
                out = fn(dist_x, idx)
                self.assertTrue(out.is_dist())
                self.assertEqual(out.process_mesh, mesh)
                self.assertEqual(out.placements, dist_x.placements)
                self._assert_allclose(out.numpy(), expected_fn(x_np))


support_types = get_xpu_op_support_types("slice")
# We only need a minimal set of dtypes to cover all runtime branches in
# `paddle/phi/kernels/xpu/contiguous_kernel.cc` (template + complex64 specialization).
for stype in ["float32", "complex64"]:
    if stype in support_types:
        create_test_class(
            globals(),
            XPUTestContiguousComplex64StridedViewXPU,
            stype,
            test_grad=False,
        )


if __name__ == "__main__":
    unittest.main()
