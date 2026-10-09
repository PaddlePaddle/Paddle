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

import itertools
import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional import fp8


class TestFP8Quantization(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        self.m = 32768
        self.n = 7168
        self.x = paddle.randn((self.m, self.n), dtype=paddle.bfloat16)
        self.rmse_threshold = 3e-2
        self.quant_method_options = ["1x128", "128x128"]
        self.input_transpose_options = [True]  # return non-transpose afterall
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [True, False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def cal_all_rmse(self, x, x_qdq, transposed: bool):
        if transposed:
            diff_squared = (x_qdq.T - x.to(paddle.float32)) ** 2
        else:
            diff_squared = (x_qdq - x.to(paddle.float32)) ** 2
        rmse = paddle.sqrt(paddle.sum(diff_squared) / x.numel())
        return rmse

    def quant_verify_wrapper(
        self,
        x: paddle.Tensor,
        quant_method: str = "1x128",
        input_transpose: bool = False,
        output_scale_transpose: bool = False,
        return_transpose_only: bool = False,
        using_pow2_scale=True,
        using_ue8m0_scale=False,
    ):
        x = x.contiguous()
        x_q_valid = False
        x_t_q_valid = False
        if input_transpose:
            if return_transpose_only:
                x_t_q, scale_t = fp8.fp8_quant_blockwise(
                    x,
                    quant_method=quant_method,
                    input_transpose=input_transpose,
                    output_scale_transpose=output_scale_transpose,
                    using_pow2_scale=using_pow2_scale,
                    return_transpose_only=return_transpose_only,
                    using_ue8m0_scale=using_ue8m0_scale,
                )
                x_t_q_valid = True
            else:
                x_q, scale, x_t_q, scale_t = fp8.fp8_quant_blockwise(
                    x,
                    quant_method=quant_method,
                    input_transpose=input_transpose,
                    output_scale_transpose=output_scale_transpose,
                    using_pow2_scale=using_pow2_scale,
                    return_transpose_only=return_transpose_only,
                    using_ue8m0_scale=using_ue8m0_scale,
                )
                x_t_q_valid = True
                x_q_valid = True

        else:
            x_q, scale = fp8.fp8_quant_blockwise(
                x,
                quant_method=quant_method,
                input_transpose=input_transpose,
                output_scale_transpose=output_scale_transpose,
                using_pow2_scale=using_pow2_scale,
                return_transpose_only=return_transpose_only,
                using_ue8m0_scale=using_ue8m0_scale,
            )
            x_q_valid = True

        valid_test_list = []

        if x_q_valid:
            valid_test_list.append((False, x_q, scale))
        if x_t_q_valid:
            valid_test_list.append((True, x_t_q, scale_t))

        rmse = 0
        for verify_transpose, x_q_in, scale_in in valid_test_list:
            scale_in = scale_in.T if output_scale_transpose else scale_in
            if using_ue8m0_scale:
                # scale_in is int32 tensor packed with 4 float scales.
                # Explicitly cast to int32 to ensure correct unpacking behavior (4 bytes per element)
                # Ensure contiguous memory layout for view operation
                scale_np = np.ascontiguousarray(scale_in.numpy()).astype(
                    np.int32
                )
                # Unpack: (M, N/4) int32 -> (M, N) uint8
                scale_u8 = scale_np.view(np.uint8)
                # Recover scale value: 2^(exponent - 127)
                scale_float = 2.0 ** (scale_u8.astype(np.float32) - 127)
                scale_in = paddle.to_tensor(scale_float)

            scale_in = paddle.repeat_interleave(
                (
                    paddle.repeat_interleave(scale_in, repeats=128, axis=0)
                    if quant_method == "128x128" and not using_ue8m0_scale
                    else scale_in
                ),
                repeats=128,
                axis=1,
            )
            scale_in = scale_in[: x_q_in.shape[0], : x_q_in.shape[1]]
            self.assertEqual(scale_in.shape, x_q_in.shape)
            x_qdq = x_q_in.astype('float32') * scale_in
            rmse = rmse + self.cal_all_rmse(x, x_qdq, verify_transpose) / len(
                valid_test_list
            )
        return rmse

    def eval_all(
        self,
        x: paddle.Tensor,
    ):
        rmses = []
        for (
            quant_method,
            input_transpose,
            output_scale_transpose,
            using_pow2_scale,
            return_transpose_only,
            using_ue8m0_scale,
        ) in itertools.product(
            self.quant_method_options,
            self.input_transpose_options,
            self.output_scale_transpose_options,
            self.using_pow2_scale_options,
            self.return_transpose_only_options,
            self.using_ue8m0_scale_options,
        ):
            rmse = self.quant_verify_wrapper(
                x,
                quant_method=quant_method,
                input_transpose=input_transpose,
                output_scale_transpose=output_scale_transpose,
                return_transpose_only=return_transpose_only,
                using_pow2_scale=using_pow2_scale,
                using_ue8m0_scale=using_ue8m0_scale,
            )
            self.assertLessEqual(rmse, self.rmse_threshold)
            rmses.append(rmse)
        return rmses

    def test_tensor_shapes(self):
        self.assertEqual(self.x.shape, [self.m, self.n])
        self.assertEqual(self.x.dtype, paddle.bfloat16)

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)

    def test_quantization_consistency(self):
        rmses1 = self.eval_all(self.x)
        rmses2 = self.eval_all(self.x)
        for r1, r2 in zip(rmses1, rmses1):
            self.assertEqual(r1, r2)


class TestFP8QuantizationFP16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 128 * 12
        self.n = 4096
        self.x = paddle.randn((self.m, self.n), dtype=paddle.float16)
        self.rmse_threshold = 3e-2
        self.quant_method_options = ["1x128", "128x128"]
        self.input_transpose_options = [True]  # return non-transpose afterall
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [True, False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)

    def test_tensor_shapes(self):
        self.assertEqual(self.x.shape, [self.m, self.n])
        self.assertEqual(self.x.dtype, paddle.float16)


class TestFP8QuantizationUnalignedBF16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 80
        self.n = 4096
        self.dtype_options = paddle.bfloat16
        self.quant_method_options = ["1x128"]
        self.rmse_threshold = 3e-2
        self.using_ue8m0_scale_options = [True, False]

        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

        self.input_transpose_options = [False]
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [False]
        self.using_pow2_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)


class TestFP8QuantizationUnalignedFP16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 8184
        self.n = 2560
        self.dtype_options = paddle.float16
        self.quant_method_options = ["1x128"]

        self.rmse_threshold = 3e-2

        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

        self.input_transpose_options = [False]
        self.output_scale_transpose_options = [True, False]
        self.return_transpose_only_options = [False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)

    def test_tensor_shapes(self):
        self.assertEqual(self.x.shape, [self.m, self.n])
        self.assertEqual(self.x.dtype, paddle.float16)


class TestFP8QuantizatioUnalignedNBF16(TestFP8Quantization):
    def setUp(self):
        paddle.seed(42)
        self.m = 129
        self.n = 508
        self.dtype_options = paddle.bfloat16
        self.quant_method_options = ["1x128"]
        self.rmse_threshold = 3e-2

        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

        self.input_transpose_options = [False]
        self.return_transpose_only_options = [False]
        self.output_scale_transpose_options = [True, False]
        self.using_pow2_scale_options = [True, False]
        self.using_ue8m0_scale_options = [True, False]

    def test_quantization_accuracy(self):
        rmses = self.eval_all(self.x)
        for r in rmses:
            self.assertLessEqual(r, self.rmse_threshold)


# 0 size
class TestFP8QuantizationZeroSizeBF16(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        self.m = 0
        self.n = 0
        self.dtype_options = paddle.bfloat16
        self.x = paddle.randn((self.m, self.n), dtype=self.dtype_options)

    def test_fp8_quant_zero_size_tensor(self):
        x_q, scale = fp8.fp8_quant_blockwise(
            self.x,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
            using_pow2_scale=False,
            return_transpose_only=False,
            using_ue8m0_scale=False,
        )
        self.assertEqual(x_q.shape, [0, 0])
        self.assertEqual(x_q.dtype, paddle.float8_e4m3fn)
        self.assertEqual(scale.shape, [0, 0])
        self.assertEqual(scale.dtype, paddle.float32)


def _bits(t):
    """Raw bytes of a tensor, so the comparison is bit level.

    float8_e4m3fn has no numpy dtype but .numpy() hands back the raw bits as
    int8, and the float32 scales come back as float32; comparing bytes keeps the
    two NaN encodings distinguishable and works where == would not.
    """
    return np.ascontiguousarray(t.numpy()).tobytes()


def _quant(x, **kw):
    opts = {
        "epsilon": 0.0,
        "input_transpose": False,
        "output_scale_transpose": True,
        "return_transpose_only": False,
        "using_pow2_scale": True,
        "using_ue8m0_scale": False,
        "quant_method": "1x128",
        "output_type": "e4m3",
    }
    opts.update(kw)
    return fp8.fp8_quant_blockwise(x, **opts)


def _offset_view(values, shape, offset, dtype=paddle.bfloat16):
    """A view of `shape` starting `offset` elements into a larger buffer.

    The offset moves the base pointer without changing the values, which is the
    only thing the vectorized quantization path is gated on. Offsets below 4
    elements are deliberately absent: the scalar kernel already requires an
    8-byte-aligned base, so those fault on every build and are out of scope.
    """
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    pad = np.zeros(offset, dtype=np.float32)
    buf = paddle.to_tensor(np.concatenate([pad, flat]), dtype=dtype)
    view = buf[offset : offset + flat.size].reshape(shape)
    return buf, view


class TestFP8QuantBasePointerAlignment(unittest.TestCase):
    """The base pointer must not change the result.

    A 16-byte-aligned base allows 128-bit loads and a narrower one does not, so
    quantizing the same values at different offsets exercises both the wide and
    the narrow path and pins them to the same output.
    """

    def setUp(self):
        paddle.seed(42)
        rs = np.random.RandomState(1234)
        self.shape = [256, 1024]
        self.values = rs.randn(*self.shape).astype(np.float32)
        self.offsets = [0, 4, 8, 16, 64, 512, 1024]

    def _run(self, offset, **kw):
        _, view = _offset_view(self.values, self.shape, offset)
        self.assertTrue(view.is_contiguous())
        return [_bits(t) for t in _quant(view, **kw)]

    def test_offsets_agree(self):
        for p2, ue8 in itertools.product([True, False], repeat=2):
            with self.subTest(pow2=p2, ue8m0=ue8):
                ref = None
                for offset in self.offsets:
                    got = self._run(
                        offset, using_pow2_scale=p2, using_ue8m0_scale=ue8
                    )
                    if ref is None:
                        ref = got
                    self.assertEqual(got, ref, f"offset {offset} differs")

    def test_offsets_agree_scale_layouts(self):
        for ot in (True, False):
            with self.subTest(output_scale_transpose=ot):
                ref = None
                for offset in self.offsets:
                    got = self._run(offset, output_scale_transpose=ot)
                    if ref is None:
                        ref = got
                    self.assertEqual(got, ref, f"offset {offset} differs")

    def test_shapes(self):
        """Several tile counts, since the offset applies per row of blocks."""
        rs = np.random.RandomState(4321)
        for shape in ([128, 128], [128, 7168], [1024, 128], [512, 2048]):
            with self.subTest(shape=shape):
                self.values = rs.randn(*shape).astype(np.float32)
                self.shape = shape
                ref = None
                for offset in self.offsets:
                    got = self._run(offset)
                    if ref is None:
                        ref = got
                    self.assertEqual(got, ref, f"offset {offset} differs")


class TestFP8QuantVecMatchesScalar(unittest.TestCase):
    """Cross-check the two 1x128 implementations against each other.

    Asking for the transpose as well forces the scalar implementation, because
    the wide path only covers the non-transposed instantiation. The
    non-transposed halves of the two results must be identical: same blocks,
    same abs-max, same scale.
    """

    SHAPES = [[128, 128], [256, 1024], [1024, 512], [128, 7168], [2048, 1024]]

    def _pair(self, x, **kw):
        vec = _quant(x, input_transpose=False, **kw)[:2]
        scalar = _quant(x, input_transpose=True, **kw)[:2]
        return [_bits(t) for t in vec], [_bits(t) for t in scalar]

    def test_flag_matrix(self):
        paddle.seed(42)
        x = paddle.randn([256, 1024], dtype=paddle.bfloat16)
        for ot, p2, ue8 in itertools.product([True, False], repeat=3):
            with self.subTest(scale_transpose=ot, pow2=p2, ue8m0=ue8):
                vec, scalar = self._pair(
                    x,
                    output_scale_transpose=ot,
                    using_pow2_scale=p2,
                    using_ue8m0_scale=ue8,
                )
                self.assertEqual(vec, scalar)

    def test_shapes(self):
        paddle.seed(7)
        for shape in self.SHAPES:
            with self.subTest(shape=shape):
                x = paddle.randn(shape, dtype=paddle.bfloat16)
                vec, scalar = self._pair(x)
                self.assertEqual(vec, scalar)

    def test_dtypes(self):
        paddle.seed(11)
        for dtype in (paddle.bfloat16, paddle.float16):
            with self.subTest(dtype=dtype):
                x = paddle.randn([256, 1024], dtype=dtype)
                vec, scalar = self._pair(x)
                self.assertEqual(vec, scalar)

    def test_ragged_shapes(self):
        """Extents that are not multiples of 128 use a different kernel.

        They cannot be cross-checked against the transposing variant, which
        rejects them outright, so the invariant here is that the offset does not
        matter. Nothing changed for these shapes; they are present so that a
        stray template instantiation leaking into the ragged dispatch shows up.

        The transposed scale is padded up to a multiple of four rows and the
        padding is never written, so only the rows that carry data are compared.
        """
        rs = np.random.RandomState(13)
        for shape in ([4096, 4100], [4095, 4096], [129, 508], [80, 4096]):
            with self.subTest(shape=shape):
                values = rs.randn(*shape).astype(np.float32)
                ref = None
                for offset in (0, 4, 8, 512):
                    _, view = _offset_view(values, shape, offset)
                    q, s = _quant(view)
                    got = [_bits(q), _bits(s[:, : shape[0]])]
                    if ref is None:
                        ref = got
                    self.assertEqual(got, ref, f"offset {offset} differs")


class TestFP8QuantAmaxEdgeCases(unittest.TestCase):
    """Inputs built so that the order of the abs-max reduction would matter.

    Reordering the reduction is the only thing the wide path does
    differently, so these are the cases where it could show. Every row is
    1024 elements, that is eight independent 128-element blocks, and each
    pattern is laid out per block so that a wrong block boundary changes a
    scale instead of adding noise.
    """

    ROWS = 256
    COLS = 1024
    BLK = 128

    def _cases(self):
        rows, cols, blk = self.ROWS, self.COLS, self.BLK
        rs = np.random.RandomState(9100)
        cases = {}

        # One huge value per block, at a different position in each block, so no
        # reduction tree can always meet it first or last.
        a = rs.randn(rows, cols).astype(np.float32) * 1e-3
        for j in range(cols // blk):
            a[:, j * blk + (j * 37) % blk] = 4.0e4 if j % 2 else -4.0e4
        cases["outlier_position"] = a

        # max is NaN suppressing, so the surviving abs-max has to come from the
        # other elements whatever the order.
        a = rs.randn(rows, cols).astype(np.float32)
        a[np.arange(rows), (np.arange(rows) * 17) % cols] = np.nan
        cases["nan"] = a

        a = rs.randn(rows, cols).astype(np.float32)
        a[:, blk * 3 + 5] = np.inf
        a[:, blk * 5 + 9] = -np.inf
        cases["inf"] = a

        # Whole blocks of zeros, which is the branch where the scale would be a
        # division by zero, plus negative zero next to it.
        a = rs.randn(rows, cols).astype(np.float32)
        a[:, blk * 2 : blk * 4] = 0.0
        a[:, blk * 6 : blk * 7] = -0.0
        cases["zero_blocks"] = a

        # Values straddling the largest e4m3, so saturation happens inside a
        # block rather than at its edge.
        cases["saturating"] = rs.randn(rows, cols).astype(np.float32) * 448.0
        cases["tiny"] = rs.randn(rows, cols).astype(np.float32) * 1e-30
        cases["huge"] = rs.randn(rows, cols).astype(np.float32) * 1e30

        a = rs.randn(rows, cols).astype(np.float32)
        a[rs.rand(rows, cols) < 0.9] = 0.0
        cases["sparse"] = a

        a = np.zeros((rows, cols), np.float32)
        cases["all_zero"] = a
        return cases

    def test_vec_matches_scalar(self):
        for name, values in self._cases().items():
            x = paddle.to_tensor(values, dtype=paddle.bfloat16)
            for p2, ue8 in itertools.product([True, False], repeat=2):
                with self.subTest(case=name, pow2=p2, ue8m0=ue8):
                    kw = {"using_pow2_scale": p2, "using_ue8m0_scale": ue8}
                    vec = [_bits(t) for t in _quant(x, **kw)[:2]]
                    scalar = [
                        _bits(t)
                        for t in _quant(x, input_transpose=True, **kw)[:2]
                    ]
                    self.assertEqual(vec, scalar)

    def test_offsets_agree(self):
        for name, values in self._cases().items():
            with self.subTest(case=name):
                ref = None
                for offset in (0, 4, 8, 512):
                    _, view = _offset_view(
                        values, [self.ROWS, self.COLS], offset
                    )
                    got = [_bits(t) for t in _quant(view)]
                    if ref is None:
                        ref = got
                    self.assertEqual(got, ref, f"offset {offset} differs")

    def test_repeatable(self):
        for name, values in self._cases().items():
            with self.subTest(case=name):
                x = paddle.to_tensor(values, dtype=paddle.bfloat16)
                first = [_bits(t) for t in _quant(x)]
                second = [_bits(t) for t in _quant(x)]
                self.assertEqual(first, second)

    def test_finite_cases_stay_accurate(self):
        """Bit agreement alone would also hold if both paths were wrong."""
        for name in ("outlier_position", "saturating", "sparse"):
            values = self._cases()[name]
            with self.subTest(case=name):
                x = paddle.to_tensor(values, dtype=paddle.bfloat16)
                q, s = _quant(x, output_scale_transpose=False)
                deq = q.astype("float32") * paddle.repeat_interleave(
                    s, repeats=self.BLK, axis=1
                )
                ref = x.astype("float32")
                scale = float(paddle.abs(ref).max())
                rmse = float(
                    paddle.sqrt(paddle.mean((deq - ref) ** 2)) / max(scale, 1.0)
                )
                self.assertLess(rmse, 3e-2)


if __name__ == '__main__':
    unittest.main()
