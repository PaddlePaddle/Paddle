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

"""Transpose / contiguous / matmul on strided views with a non-zero offset.

Three groups of cases:

TestTransposeVecTile
    walks both sides of every selection gate of the vectorized 64x64 tile
    transpose kernel, so that neither the vectorized nor the scalar tiling can
    regress unnoticed. Transpose is pure data movement, so the reference is an
    exact bit comparison rather than a tolerance.

TestContiguousNonZeroOffset
    materializes views whose offset is non-zero. A freshly allocated output must
    always start at offset zero, whatever the input offset was; asserting that
    is what keeps the offset from leaking onto the output.

TestMatmulStrideOffset
    a transposed view is folded into the cuBLAS transpose flag instead of being
    materialized. Folding must not depend on the offset, so the result has to be
    identical, bit for bit, to the same call on a copy that starts at offset
    zero.

All of it needs FLAGS_use_stride_kernel, because that is what makes slicing and
transposing return a view instead of a fresh tensor. The transpose cases are run
with the flag off as well, since that reaches the same kernel through the dense
dispatch instead of through ContiguousKernel.
"""

import unittest

import numpy as np

import paddle


def _host_values(n, dtype, seed):
    """Values for a buffer of n elements, generated on the host.

    The dtype only decides which generator is used; correctness is later checked
    against the raw bits read back from the device, so no value here has to be
    exactly representable in the target dtype.
    """
    rs = np.random.RandomState(seed)
    if dtype == "bool":
        return rs.randint(0, 2, n).astype(np.bool_)
    if dtype == "int8":
        return rs.randint(-100, 100, n).astype(np.int8)
    if dtype in ("int16", "int32", "int64"):
        return rs.randint(-(2**20), 2**20, n).astype(np.int64)
    return (rs.randn(n) * 8.0).astype(np.float32)


def _buffer(n, dtype, seed):
    """A contiguous device buffer of n elements and its bits as a host array.

    Taking the reference from the device tensor's own bits, instead of from the
    host array it was built from, keeps the comparison exact for every dtype:
    .numpy() on bfloat16 hands back uint16, on float8 int8, so the reference is
    a permutation of bit patterns and never a rounded value.
    """
    t = paddle.to_tensor(_host_values(n, dtype, seed), dtype=dtype)
    return t, t.numpy()


def _bits(t):
    return t.numpy().tobytes()


def _numel(shape):
    return int(np.prod(shape))


@unittest.skipIf(
    not paddle.is_compiled_with_cuda(), "these kernels are GPU only"
)
class StrideFlagTestCase(unittest.TestCase):
    """Force FLAGS_use_stride_kernel on and put it back afterwards."""

    def setUp(self):
        paddle.set_device("gpu:0")
        self._saved = paddle.get_flags(
            ["FLAGS_use_stride_kernel", "FLAGS_use_stride_compute_kernel"]
        )
        paddle.set_flags(
            {
                "FLAGS_use_stride_kernel": True,
                "FLAGS_use_stride_compute_kernel": True,
            }
        )

    def tearDown(self):
        paddle.set_flags(self._saved)

    def check_transpose(self, shape, perm, dtype, seed):
        """Exact check of transpose(shape, perm) for one dtype."""
        t, bits = _buffer(_numel(shape), dtype, seed)
        t = t.reshape(shape)
        out = paddle.transpose(t, perm).contiguous()
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.shape, [shape[i] for i in perm])
        ref = np.ascontiguousarray(np.transpose(bits.reshape(shape), perm))
        np.testing.assert_array_equal(out.numpy(), ref)

    def sweep(self, cases, seed=7000):
        """Run every case with the stride kernel both on and off.

        Same kernel, two different ways in: with the flag on, transpose only
        builds a view and the kernel runs from ContiguousKernel; with it off,
        transpose dispatches the dense kernel directly.
        """
        for use_stride in (True, False):
            paddle.set_flags({"FLAGS_use_stride_kernel": use_stride})
            for i, (name, shape, perm, dtype) in enumerate(cases):
                with self.subTest(case=name, stride=use_stride):
                    self.check_transpose(shape, perm, dtype, seed + i)


class TestTransposeVecTile(StrideFlagTestCase):
    """Both sides of every gate guarding the vectorized tile transpose.

    The gates are: a 2-byte dtype, both transposed extents even, both base
    pointers 4-byte aligned, gridDim.z within 65535, and enough 64x64 tiles to
    fill the device. Whether a case ends up vectorized is not observable from
    the output, so the intent is recorded in the case name only; what is checked
    is that every case is exact.
    """

    @property
    def sm_count(self):
        return paddle.device.cuda.get_device_properties(0).multi_processor_count

    def test_inside_every_gate(self):
        # 2048x2048 is 1024 tiles of 64x64, far past the tile-count gate on any
        # real device, and small enough to stay cheap.
        self.sweep(
            [
                ("2d_bf16", [2048, 2048], [1, 0], "bfloat16"),
                ("2d_fp16", [2048, 2048], [1, 0], "float16"),
                ("3d_bf16", [4, 1024, 1024], [0, 2, 1], "bfloat16"),
                ("wide", [1024, 4096], [1, 0], "bfloat16"),
                ("tall", [4096, 1024], [1, 0], "bfloat16"),
                ("tile_multiple", [1280, 640], [1, 0], "bfloat16"),
            ]
        )

    def test_ragged_tiles(self):
        """Extents even but not multiples of 64.

        The edge tiles are partial, which is the one case the vectorized kernel
        handles element by element instead of with vector loads, so these cover
        its second half.
        """
        self.sweep(
            [
                ("both_ragged", [2050, 2050], [1, 0], "bfloat16"),
                ("cols_ragged", [2048, 2050], [1, 0], "bfloat16"),
                ("rows_ragged", [2050, 2048], [1, 0], "bfloat16"),
                ("ragged_3d", [4, 1026, 1026], [0, 2, 1], "bfloat16"),
            ],
            seed=7100,
        )

    def test_odd_extent_rejected(self):
        """An odd extent makes the 2-element vector loads illegal."""
        self.sweep(
            [
                ("odd_rows", [2049, 2048], [1, 0], "bfloat16"),
                ("odd_cols", [2048, 2049], [1, 0], "bfloat16"),
                ("odd_both", [2049, 2049], [1, 0], "bfloat16"),
                ("odd_3d", [4, 1025, 1025], [0, 2, 1], "bfloat16"),
            ],
            seed=7200,
        )

    def test_too_few_tiles(self):
        """Below the tile-count gate the scalar 32x32 tiling must be kept."""
        n = self.sm_count
        self.sweep(
            [
                ("tiles_eq_sm", [n, 64, 64], [0, 2, 1], "bfloat16"),
                ("tiles_lt_sm", [n - 1, 64, 64], [0, 2, 1], "bfloat16"),
                ("tiles_few", [1, 128, 128], [0, 2, 1], "bfloat16"),
                ("tiles_one", [1, 64, 64], [0, 2, 1], "bfloat16"),
            ],
            seed=7300,
        )

    def test_grid_z_overflow(self):
        """gridDim.z is limited to 65535, and dim0 maps onto it."""
        self.sweep(
            [
                ("z_at_limit", [65535, 16, 16], [0, 2, 1], "bfloat16"),
                ("z_over_limit", [65600, 16, 16], [0, 2, 1], "bfloat16"),
            ],
            seed=7400,
        )

    def test_small_extents(self):
        """Extents under 16 do not reach the tiling path at all."""
        self.sweep(
            [
                ("narrow", [1, 8, 4096], [0, 2, 1], "bfloat16"),
                ("narrow_t", [1, 4096, 8], [0, 2, 1], "bfloat16"),
                ("tiny", [1, 8, 8], [0, 2, 1], "bfloat16"),
                ("single", [1, 1, 1], [0, 2, 1], "bfloat16"),
            ],
            seed=7500,
        )

    def test_other_dtypes(self):
        """Only 2-byte dtypes are vectorized; the rest must be untouched."""
        self.sweep(
            [
                ("fp32", [1024, 1024], [1, 0], "float32"),
                ("fp64", [512, 512], [1, 0], "float64"),
                ("int8", [2048, 2048], [1, 0], "int8"),
                ("int32", [1024, 1024], [1, 0], "int32"),
                ("int64", [512, 512], [1, 0], "int64"),
                ("bool", [1024, 1024], [1, 0], "bool"),
            ],
            seed=7600,
        )

    def test_other_permutations(self):
        """Permutations that are not a trailing-pair swap use other kernels."""
        self.sweep(
            [
                ("perm_210", [64, 64, 64], [2, 1, 0], "bfloat16"),
                ("perm_102", [64, 32, 128], [1, 0, 2], "bfloat16"),
                ("perm_120", [32, 64, 128], [1, 2, 0], "bfloat16"),
                ("perm_0213", [8, 64, 32, 128], [0, 2, 1, 3], "bfloat16"),
                ("perm_3021", [4, 8, 16, 32], [3, 0, 2, 1], "bfloat16"),
                ("perm_5d", [2, 3, 4, 5, 6], [4, 3, 2, 1, 0], "bfloat16"),
                ("perm_identity", [16, 32, 64], [0, 1, 2], "bfloat16"),
                ("perm_1d", [1024], [0], "bfloat16"),
                ("perm_0213_fp32", [8, 64, 32, 128], [0, 2, 1, 3], "float32"),
            ],
            seed=7700,
        )

    def test_misaligned_base_pointer(self):
        """Break the alignment gate through the pointer instead of the shape.

        A bfloat16 view starting at an odd element offset has a 2-byte-aligned
        base, so 4-byte vector loads would fault and the gate has to reject it.
        Only reachable with the stride kernel on; otherwise the slice is
        materialized and the odd offset is lost.
        """
        shape = [1024, 1024]
        n = _numel(shape)
        buf, bits = _buffer(n + 1, "bfloat16", 7800)
        view = buf[1:].reshape(shape)
        self.assertEqual(view.data_ptr() % 4, 2)
        out = paddle.transpose(view, [1, 0]).contiguous()
        ref = np.ascontiguousarray(bits[1:].reshape(shape).T)
        np.testing.assert_array_equal(out.numpy(), ref)


class TestContiguousNonZeroOffset(StrideFlagTestCase):
    """Materializing a view whose offset is not zero.

    Element offset 1024 in bfloat16 is what a fused parameter buffer produces
    for every parameter but the first, so it is the shape of the case that
    matters in practice; the small offsets are there because they land in a
    different alignment class and must therefore take a different path inside
    the transpose kernel.
    """

    def _view(self, shape, dtype, off, seed):
        """A view of `shape` starting `off` elements into a larger buffer."""
        n = _numel(shape)
        buf, bits = _buffer(n + off, dtype, seed)
        view = buf[off : off + n].reshape(shape)
        self.assertEqual(view.offset, off * buf.element_size())
        self.assertTrue(view.is_contiguous())
        return buf, view, bits[off : off + n].reshape(shape)

    def _materialize(self, view, host, perm):
        """transpose(perm).contiguous() on a view, checked exactly."""
        strided = paddle.transpose(view, perm)
        out = strided.contiguous()
        # The output is a fresh allocation, so it has to start at offset zero
        # no matter what the input offset was. Inheriting the input offset would
        # read and write past the end of that allocation.
        self.assertEqual(out.offset, 0)
        self.assertTrue(out.is_contiguous())
        self.assertNotEqual(out.data_ptr(), view.data_ptr())
        ref = np.ascontiguousarray(np.transpose(host, perm))
        np.testing.assert_array_equal(out.numpy(), ref)
        return out

    def test_transposed_2d(self):
        cases = [
            # shape, dtype, element offset
            ([2048, 2048], "bfloat16", 0),
            ([2048, 2048], "bfloat16", 1),
            ([2048, 2048], "bfloat16", 4),
            ([2048, 2048], "bfloat16", 1024),
            ([1024, 3072], "bfloat16", 1024),
            ([2050, 2050], "bfloat16", 1024),
            ([2049, 2048], "bfloat16", 1024),
            ([65, 4096], "bfloat16", 1024),
            ([4096, 65], "bfloat16", 1024),
            ([2048, 2048], "float16", 1024),
            ([1024, 1024], "float32", 3),
            ([512, 512], "float64", 7),
            ([2048, 2048], "int8", 5),
            ([1024, 1024], "int32", 1024),
        ]
        for i, (shape, dtype, off) in enumerate(cases):
            with self.subTest(shape=shape, dtype=dtype, offset=off):
                _, view, host = self._view(shape, dtype, off, 8000 + i)
                self._materialize(view, host, [1, 0])

    def test_permuted_3d(self):
        """Pure permutations that are not a 2-D transpose, at an offset."""
        cases = [
            ([4, 1024, 1024], [0, 2, 1]),
            ([4, 1026, 1026], [0, 2, 1]),
            ([64, 64, 64], [2, 1, 0]),
            ([64, 32, 128], [1, 0, 2]),
            ([8, 16, 32, 64], [0, 2, 1, 3]),
            ([8, 16, 32, 64], [3, 2, 1, 0]),
        ]
        for i, (shape, perm) in enumerate(cases):
            with self.subTest(shape=shape, perm=perm):
                _, view, host = self._view(shape, "bfloat16", 1024, 8100 + i)
                self._materialize(view, host, perm)

    def test_non_permutation_views(self):
        """Strided views that are not a permutation, so they cannot be folded.

        These must be rejected and go to the generic strided copy, on top of a
        non-zero offset; the output still has to be exact and start at zero.
        """
        shape = [512, 512]
        _, view, host = self._view(shape, "bfloat16", 1024, 8200)
        for name, sub, ref in [
            ("row_step2", view[::2], host[::2]),
            ("col_step2", view[:, ::2], host[:, ::2]),
            ("row_slice_t", view[3:7].t(), host[3:7].T),
            ("col_slice", view[:, 5:9], host[:, 5:9]),
            ("both_step", view[1::3, 2::5], host[1::3, 2::5]),
        ]:
            with self.subTest(case=name):
                out = sub.contiguous()
                self.assertEqual(out.offset, 0)
                self.assertTrue(out.is_contiguous())
                np.testing.assert_array_equal(
                    out.numpy(), np.ascontiguousarray(ref)
                )

    def test_assign_materializes_the_same_bits(self):
        """paddle.assign has to materialize the view too, and agree."""
        _, view, host = self._view([1024, 1024], "bfloat16", 1024, 8300)
        strided = view.t()
        by_contiguous = strided.contiguous()
        by_assign = paddle.assign(strided).contiguous()
        self.assertEqual(_bits(by_assign), _bits(by_contiguous))
        np.testing.assert_array_equal(
            by_contiguous.numpy(), np.ascontiguousarray(host.T)
        )

    def test_output_does_not_alias_input(self):
        """The materialized copy must not be affected by later writes."""
        buf, view, host = self._view([256, 256], "bfloat16", 1024, 8400)
        out = view.t().contiguous()
        before = _bits(out)
        buf[:] = paddle.zeros_like(buf)
        self.assertEqual(_bits(out), before)
        np.testing.assert_array_equal(out.numpy(), np.ascontiguousarray(host.T))

    def test_offset_invariance(self):
        """The same values at different offsets must give the same bits."""
        shape = [1024, 1024]
        seen = {}
        for off in (0, 1, 2, 4, 8, 128, 1024):
            n = _numel(shape)
            vals = _host_values(n, "bfloat16", 8500)
            pad = np.zeros(off, dtype=vals.dtype)
            buf = paddle.to_tensor(
                np.concatenate([pad, vals]), dtype="bfloat16"
            )
            view = buf[off : off + n].reshape(shape)
            seen[off] = _bits(view.t().contiguous())
        ref = seen[0]
        for off, got in seen.items():
            self.assertEqual(got, ref, f"offset {off} differs from offset 0")


class TestMatmulStrideOffset(StrideFlagTestCase):
    """Folding a transposed operand into cuBLAS when its offset is not zero.

    The reference is the same call on a copy that starts at offset zero. Both go
    through the same fold and the same cuBLAS call, so the results must
    agree bit for bit; a difference means the offset changed the path taken.

    The offset used is a multiple of 1024 elements, the granularity a fused
    parameter buffer aligns to, so the base pointer keeps its alignment
    class and cuBLAS cannot pick a different kernel for that reason alone.
    """

    OFFSET = 1024

    def _pair(self, shape, dtype, seed):
        """The same values, once as an offset view and once as a tensor."""
        n = _numel(shape)
        vals = _host_values(n, dtype, seed)
        pad = np.zeros(self.OFFSET, dtype=vals.dtype)
        buf = paddle.to_tensor(np.concatenate([pad, vals]), dtype=dtype)
        view = buf[self.OFFSET : self.OFFSET + n].reshape(shape)
        plain = paddle.to_tensor(vals, dtype=dtype).reshape(shape)
        self.assertNotEqual(view.offset, 0)
        self.assertEqual(view.data_ptr() % 16, plain.data_ptr() % 16)
        # Same bits at both offsets, otherwise the comparison below is vacuous.
        self.assertEqual(_bits(view.contiguous()), _bits(plain))
        return view, plain

    def _same(self, fn, shape, dtype, seed):
        """fn applied to the offset view and to the copy must agree exactly."""
        view, plain = self._pair(shape, dtype, seed)
        got, ref = fn(view), fn(plain)
        self.assertEqual(got.shape, ref.shape)
        self.assertEqual(_bits(got), _bits(ref))
        return got

    def test_y_operand_folded(self):
        """matmul(x, w.t()): the transposed view is the second operand."""
        x = paddle.randn([512, 2048], dtype="float32").astype("bfloat16")
        u = paddle.randn([256, 1024], dtype="float32").astype("bfloat16")
        for name, fn in [
            ("wt", lambda w: paddle.matmul(x, w.t())),
            ("wt_ty", lambda w: paddle.matmul(u, w.t(), transpose_y=True)),
            ("w_ty", lambda w: paddle.matmul(x, w, transpose_y=True)),
            ("w_plain", lambda w: paddle.matmul(u, w)),
        ]:
            with self.subTest(case=name):
                self._same(fn, [1024, 2048], "bfloat16", 8600)

    def test_x_operand_folded(self):
        """matmul(w.t(), z): the transposed view is the first operand."""
        z = paddle.randn([1024, 256], dtype="float32").astype("bfloat16")
        u = paddle.randn([256, 1024], dtype="float32").astype("bfloat16")
        for name, fn in [
            ("wt_z", lambda w: paddle.matmul(w.t(), z)),
            ("wt_ut", lambda w: paddle.matmul(w.t(), u, transpose_y=True)),
            ("w_tx", lambda w: paddle.matmul(w, z, transpose_x=True)),
        ]:
            with self.subTest(case=name):
                self._same(fn, [1024, 2048], "bfloat16", 8700)

    def test_both_operands_folded(self):
        """Both operands are offset views of the same buffer."""
        n = 1024 * 512
        vals = _host_values(2 * n + self.OFFSET, "bfloat16", 8800)
        buf = paddle.to_tensor(vals, dtype="bfloat16")
        a = buf[self.OFFSET : self.OFFSET + n].reshape([512, 1024])
        b = buf[self.OFFSET + n : self.OFFSET + 2 * n].reshape([512, 1024])
        got = paddle.matmul(a.t(), b)
        ref = paddle.matmul(
            a.contiguous().t(), b.contiguous(), transpose_y=False
        )
        self.assertEqual(_bits(got), _bits(ref))

    def test_batched(self):
        """3-D operands, both a trailing swap and a permutation that is not."""
        g = paddle.randn([4, 256, 512], dtype="float32").astype("bfloat16")
        h = paddle.randn([64, 32, 4], dtype="float32").astype("bfloat16")
        for name, shape, fn in [
            (
                "swap_last2",
                [4, 512, 512],
                lambda w: paddle.matmul(g, w.transpose([0, 2, 1])),
            ),
            (
                "batched_x",
                [4, 512, 512],
                lambda w: paddle.matmul(
                    w.transpose([0, 2, 1]), g, transpose_y=True
                ),
            ),
            (
                "perm_102",
                [4, 64, 512],
                lambda w: paddle.matmul(h, w.transpose([1, 0, 2])),
            ),
        ]:
            with self.subTest(case=name):
                self._same(fn, shape, "bfloat16", 8900)

    def test_dtypes(self):
        """The fold is dtype independent."""
        for dtype in ("float16", "float32", "float64"):
            with self.subTest(dtype=dtype):
                x = paddle.randn([256, 512], dtype="float32").astype(dtype)
                self._same(
                    lambda w: paddle.matmul(x, w.t()), [256, 512], dtype, 9000
                )

    def test_against_float64_reference(self):
        """Guard against both paths being wrong in the same way.

        Bit-identity only says the offset did not change the answer. This checks
        the answer itself, in float64 on the host, with a tolerance set by the
        input dtype rather than by the accumulation order.
        """
        shape = [256, 512]
        view, _ = self._pair(shape, "float32", 9100)
        x = paddle.randn([128, 512], dtype="float32")
        got = paddle.matmul(x, view.t()).numpy()
        ref = (
            x.numpy().astype(np.float64)
            @ view.contiguous().numpy().astype(np.float64).T
        )
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_folded_result_is_used_by_backward(self):
        """The same fold runs in the gradient matmuls."""
        w = paddle.randn([1024, 512], dtype="float32")
        w.stop_gradient = False
        x = paddle.randn([256, 512], dtype="float32")
        x.stop_gradient = False
        out = paddle.matmul(x, w.t())
        out.backward()
        self.assertEqual(w.grad.shape, w.shape)
        self.assertEqual(x.grad.shape, x.shape)
        np.testing.assert_allclose(
            x.grad.numpy(),
            np.ones_like(out.numpy()) @ w.numpy(),
            rtol=1e-5,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
