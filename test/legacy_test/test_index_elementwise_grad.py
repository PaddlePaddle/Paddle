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


class TestIndexElementwiseGrad(unittest.TestCase):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"

    def setUp(self):
        self.init()

        if self.dtype in ["float32", "float64"]:
            self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        elif self.dtype in ["int32", "int8", "int64", "int16", "uint8"]:
            self.x_np = np.random.randint(
                100, size=self.x_shape, dtype=self.dtype
            )
        elif self.dtype == "float16":
            self.x_np = np.random.random(self.x_shape).astype("float16")

        self.index_np = np.random.randint(
            2, size=self.index_shape, dtype="bool"
        )

    def test_grad(self):
        paddle.disable_static()

        x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
        index = paddle.to_tensor(self.index_np).astype('bool')

        out = x[index]
        out_grad = paddle.ones_like(out)
        out.backward(out_grad)
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        x_grad_np = x.grad.numpy()
        expanded_index = np.expand_dims(
            self.index_np, axis=tuple(range(self.k, self.dim))
        )
        expanded_index = np.broadcast_to(expanded_index, self.x_shape)
        expected_grad = np.where(expanded_index, 1.0, 0.0).astype(self.dtype)

        atol = 1e-5 if self.dtype in ["float32", "float64"] else 1e-3
        rtol = 1e-5 if self.dtype in ["float32", "float64"] else 1e-3

        np.testing.assert_allclose(
            x_grad_np, expected_grad, atol=atol, rtol=rtol
        )

        paddle.enable_static()


class TestIndexElementwiseGrad3D(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad4D_k2(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad4D_k3(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad5D_k2(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad5D_k3(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGrad5D_k4(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 5
        self.x_shape = (2, 3, 4, 5, 6)
        self.k = 4
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"


class TestIndexElementwiseGradFloat64(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float64"


class TestIndexElementwiseGradFloat16(TestIndexElementwiseGrad):
    def init(self):
        self.dim = 4
        self.x_shape = (3, 4, 5, 6)
        self.k = 3
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float16"

    def setUp(self):
        self.init()
        self.x_np = np.random.random(self.x_shape).astype("float16")
        self.index_np = np.random.randint(
            2, size=self.index_shape, dtype="bool"
        )


class TestIndexElementwiseGradWithCustomOutGrad(unittest.TestCase):
    def init(self):
        self.dim = 3
        self.x_shape = (4, 5, 6)
        self.k = 2
        self.index_shape = self.x_shape[: self.k]
        self.dtype = "float32"

    def setUp(self):
        self.init()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.index_np = np.random.randint(
            2, size=self.index_shape, dtype="bool"
        )

    def test_custom_out_grad(self):
        paddle.disable_static()

        x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
        index = paddle.to_tensor(self.index_np).astype('bool')

        out = x[index]
        custom_grad = paddle.randn_like(out)
        out.backward(custom_grad)

        self.assertEqual(x.grad.shape, x.shape)
        paddle.enable_static()


class TestIndexElementwiseGradZeroIndex(unittest.TestCase):
    def test_zero_index(self):
        paddle.disable_static()

        x = paddle.randn([4, 5, 6], dtype='float32')
        x.stop_gradient = False
        index = paddle.zeros([4, 5], dtype='bool')
        out = x[index]
        self.assertEqual(out.numel(), 0)
        if out.numel() > 0:
            out.backward(paddle.ones_like(out))
            np.testing.assert_allclose(
                x.grad.numpy(), np.zeros_like(x.numpy()), atol=1e-5
            )

        paddle.enable_static()


class TestIndexElementwiseGradAllIndex(unittest.TestCase):
    def test_all_index(self):
        paddle.disable_static()

        x_np = np.random.random([4, 5, 6]).astype('float32')
        x = paddle.to_tensor(x_np, stop_gradient=False)
        index = paddle.ones([4, 5], dtype='bool')
        out = x[index]
        out.backward(paddle.ones_like(out))
        expected_grad = np.ones_like(x_np)
        np.testing.assert_allclose(
            x.grad.numpy(), expected_grad, atol=1e-5, rtol=1e-5
        )

        paddle.enable_static()


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda(), 'CUDA is required for this test.'
)
class TestIndexElementwiseGetGradStride1(unittest.TestCase):
    def test_duplicate_index_warp_boundaries(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))

        try:
            for num_duplicates in (1, 32, 33):
                with self.subTest(num_duplicates=num_duplicates):
                    index_np = np.array(
                        [11, *([7] * num_duplicates), 3], dtype=np.int64
                    )
                    out_grad_np = np.linspace(
                        0.25, 1.25, index_np.size, dtype=np.float32
                    )
                    expected_grad = np.zeros([16], dtype=np.float32)
                    np.add.at(expected_grad, index_np, out_grad_np)

                    x = paddle.zeros([16], dtype='float32')
                    x.stop_gradient = False
                    index = paddle.to_tensor(index_np)
                    out_grad = paddle.to_tensor(out_grad_np)

                    x[index].backward(out_grad)

                    np.testing.assert_allclose(
                        x.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-6
                    )
        finally:
            paddle.enable_static()


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and not paddle.device.is_compiled_with_rocm(),
    'The sorted-index backward kernels are built for CUDA only.',
)
class TestIndexElementwiseGetGradSmallStride(unittest.TestCase):
    """Duplicate indices with 1 < sliceSize <= 32 must reduce all duplicates in
    float32 and round to the low-precision dtype only once. Rounding on every
    duplicate would lose up to one ulp per accumulation step.

    ROCm/DCU is excluded because ``IndexPutWithSortKernel`` is guarded by
    ``PADDLE_WITH_CUDA``; those builds fall back to a bfloat16/float16
    ``CudaAtomicAdd`` that rounds on every duplicate.
    """

    def _reference_grad(self, rows, index_np, out_grad_np, dtype):
        expected = np.zeros([rows, out_grad_np.shape[1]], dtype=np.float32)
        np.add.at(expected, index_np, out_grad_np)
        return paddle.to_tensor(expected).astype(dtype).astype('float32')

    def test_low_precision_accumulation(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))

        try:
            rows = 16
            for dtype in ('float16', 'bfloat16'):
                for slice_size in (2, 8, 32):
                    with self.subTest(dtype=dtype, slice_size=slice_size):
                        rng = np.random.default_rng(2025)
                        index_np = rng.integers(
                            0, rows, size=(64,), dtype=np.int64
                        )
                        out_grad = paddle.to_tensor(
                            rng.uniform(-0.5, 0.5, (64, slice_size)),
                            dtype=dtype,
                        )
                        x = paddle.zeros([rows, slice_size], dtype=dtype)
                        x.stop_gradient = False

                        x[paddle.to_tensor(index_np)].backward(out_grad)

                        expected = self._reference_grad(
                            rows,
                            index_np,
                            out_grad.astype('float32').numpy(),
                            dtype,
                        )
                        np.testing.assert_array_equal(
                            x.grad.astype('float32').numpy(),
                            expected.numpy(),
                        )
        finally:
            paddle.enable_static()


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda(), 'CUDA is required for this test.'
)
class TestIndexElementwiseGetGradSlicedView(unittest.TestCase):
    """Advanced indexing applied to a basic slice.

    The slice is a view, so the backward kernel receives its byte offset inside
    ``x_grad`` (``slice_offset``) plus the view's strides, and has to scatter
    the reduced gradient back through them. A wrong offset or stride would move
    the whole gradient block, which shows up both as wrong values inside the
    slice and as non-zero gradient on rows the expression never read. The cases
    below deliberately include non-zero offsets (``x[1::2]``, ``x[2:7:2]``),
    and are run with ``FLAGS_use_stride_kernel`` both on and off, since only
    the former reaches ``index_elementwise_get_grad`` at all.
    """

    SHAPE = (8, 6)

    def _cases(self):
        return [
            ('x[::2, idx]', lambda t, i: t[::2, i], (slice(None, None, 2),)),
            ('x[1::2, idx]', lambda t, i: t[1::2, i], (slice(1, None, 2),)),
            ('x[2:7:2, idx]', lambda t, i: t[2:7:2, i], (slice(2, 7, 2),)),
            ('x[1:5, idx]', lambda t, i: t[1:5, i], (slice(1, 5),)),
            (
                'x[1::2, idx, None]',
                lambda t, i: t[1::2, i, None],
                (slice(1, None, 2),),
            ),
        ]

    def _expected(self, base_slice, index_np, out_grad_np, dtype):
        """Reduce in float32 and round once, which is what the sorted-index
        kernel does, then place the block back at the sliced rows."""
        rows = out_grad_np.shape[0]
        block = np.zeros([rows, self.SHAPE[1]], dtype=np.float32)
        np.add.at(block, (slice(None), index_np), out_grad_np)
        block = paddle.to_tensor(block).astype(dtype).astype('float32').numpy()
        expected = np.zeros(self.SHAPE, dtype=np.float32)
        expected[base_slice] = block
        return expected

    def _run(self, use_stride_kernel):
        # 7 duplicates of column 2 exercise the duplicate reduction; the
        # gradient of every untouched column stays exactly zero.  The gradient
        # values are quarters so that every partial sum is exact in float16 and
        # bfloat16 too, which keeps the comparison independent of the order the
        # kernel reduces duplicates in.
        index_np = np.array([2, 2, 2, 2, 2, 2, 2, 5], dtype=np.int64)
        for dtype in ('float32', 'float16', 'bfloat16'):
            for name, fn, base_slice in self._cases():
                with self.subTest(
                    use_stride_kernel=use_stride_kernel, dtype=dtype, expr=name
                ):
                    x = paddle.zeros(list(self.SHAPE), dtype=dtype)
                    x.stop_gradient = False
                    out = fn(x, paddle.to_tensor(index_np))

                    rows = out.shape[0]
                    steps = np.arange(rows * index_np.size, dtype=np.float32)
                    out_grad_np = ((steps % 8) + 1).reshape(
                        [rows, index_np.size]
                    ) / 4
                    out_grad = paddle.to_tensor(out_grad_np, dtype=dtype)
                    out.backward(out_grad.reshape(out.shape))

                    np.testing.assert_array_equal(
                        x.grad.astype('float32').numpy(),
                        self._expected(
                            base_slice, index_np, out_grad_np, dtype
                        ),
                    )

    def test_sliced_view_grad(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        original = paddle.get_flags('FLAGS_use_stride_kernel')[
            'FLAGS_use_stride_kernel'
        ]
        try:
            for use_stride_kernel in (True, False):
                paddle.set_flags({'FLAGS_use_stride_kernel': use_stride_kernel})
                self._run(use_stride_kernel)
        finally:
            paddle.set_flags({'FLAGS_use_stride_kernel': original})
            paddle.enable_static()


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and not paddle.device.is_compiled_with_rocm(),
    'The sorted-index backward kernels are built for CUDA only.',
)
class TestIndexElementwiseGetGradSlicedViewRounding(unittest.TestCase):
    """A basic slice is a strided view, so its gradient is reduced into a
    contiguous scratch buffer and scattered back through the view's strides.
    That reduction must still happen in float32 and round only once. Unlike
    ``TestIndexElementwiseGetGradSlicedView`` the gradient values here are not
    exactly representable, so a per-duplicate ``CudaAtomicAdd`` fallback would
    round on every step and fail the comparison.
    """

    SHAPE = (8, 6)

    def _cases(self):
        return [
            ('x[1::2, idx]', lambda t, i: t[1::2, i], (slice(1, None, 2),)),
            ('x[2:7:2, idx]', lambda t, i: t[2:7:2, i], (slice(2, 7, 2),)),
        ]

    def test_sliced_view_rounds_once(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        index_np = np.array([2, 2, 2, 2, 2, 2, 2, 5], dtype=np.int64)
        try:
            for dtype in ('float16', 'bfloat16'):
                for name, fn, base_slice in self._cases():
                    with self.subTest(dtype=dtype, expr=name):
                        rng = np.random.default_rng(2026)
                        x = paddle.zeros(list(self.SHAPE), dtype=dtype)
                        x.stop_gradient = False
                        out = fn(x, paddle.to_tensor(index_np))

                        out_grad = paddle.to_tensor(
                            rng.uniform(-0.5, 0.5, tuple(out.shape)),
                            dtype=dtype,
                        )
                        out.backward(out_grad)

                        block = np.zeros(
                            [out.shape[0], self.SHAPE[1]], dtype=np.float32
                        )
                        np.add.at(
                            block,
                            (slice(None), index_np),
                            out_grad.astype('float32').numpy(),
                        )
                        block = (
                            paddle.to_tensor(block)
                            .astype(dtype)
                            .astype('float32')
                            .numpy()
                        )
                        expected = np.zeros(self.SHAPE, dtype=np.float32)
                        expected[base_slice] = block

                        np.testing.assert_array_equal(
                            x.grad.astype('float32').numpy(), expected
                        )
        finally:
            paddle.enable_static()


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda(), 'CUDA is required for this test.'
)
class TestIndexElementwiseNegativeStrideView(unittest.TestCase):
    """Advanced indexing applied to a reversed view.

    A negative step makes the view's stride negative and puts its base offset
    at the *highest* address of that axis, so every element the kernel touches
    sits at a non-positive offset from ``slice_offset``. The offset calculators
    must therefore keep their offsets in a signed type; an unsigned one turns
    those offsets into huge positive numbers and the kernel reads or writes
    outside the tensor.

    numpy is the reference: torch rejects negative steps outright
    (``step must be greater than zero``), so there is nothing to compare
    against there. The backward reference is exact rather than approximate --
    running the same expression on an array of flat positions names the source
    element of every output element, which is exactly what the gradient
    scatters into.

    Ranks above two matter on their own: once an axis is left over after the
    indexed one, the iteration dimensions get sorted by stride, and ordering a
    negative stride as the smallest one permutes the output layout.
    """

    SHAPES = ((8, 6), (8, 6, 6), (8, 6, 33))

    def _cases(self, ndim):
        cases = [
            ('x[::-1, idx]', lambda t, i: t[::-1, i]),
            ('x[::-2, idx]', lambda t, i: t[::-2, i]),
            ('x[1::-1, idx]', lambda t, i: t[1::-1, i]),
            ('x[idx, ::-1]', lambda t, i: t[i, ::-1]),
            ('x[::-1][idx]', lambda t, i: t[::-1][i]),
            ('x[:, ::-1][idx]', lambda t, i: t[:, ::-1][i]),
            ('x[::-1, ::-1][idx]', lambda t, i: t[::-1, ::-1][i]),
            ('x[..., ::-1][idx]', lambda t, i: t[..., ::-1][i]),
            ('x[::-1, idx2]', lambda t, i: t[::-1, i]),
            # x[::-1][:, idx] is left out on purpose: advanced indexing on a
            # non-contiguous base with the indexed axis after a sliced one is
            # broken for positive steps too (x[::2][:, idx] returns the wrong
            # shape), so it is not a negative-stride issue.
        ]
        if ndim > 2:
            cases += [
                ('x[::-1, :, idx]', lambda t, i: t[::-1, :, i]),
                ('x[idx, ::-1, :]', lambda t, i: t[i, ::-1, :]),
                ('x[::-1, idx, ::-1]', lambda t, i: t[::-1, i, ::-1]),
            ]
        return cases

    def _run(self, dtype, shape):
        # Integral values keep every partial sum exact, so the comparison is
        # independent of the order duplicates are reduced in.
        x_np = (
            np.arange(np.prod(shape), dtype=np.float32)
            .reshape(shape)
            .astype(dtype)
        )
        positions = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
        idx_np = np.array([2, 5, 2, 0, 5], dtype=np.int64)
        idx2_np = np.array([[1, 3], [3, 1], [0, 0]], dtype=np.int64)

        for name, fn in self._cases(len(shape)):
            index_np = idx2_np if 'idx2' in name else idx_np
            with self.subTest(dtype=dtype, shape=shape, expr=name):
                expected = fn(x_np, index_np)
                selected = fn(positions, index_np)
                grad_np = (
                    np.arange(selected.size, dtype=np.float32).reshape(
                        selected.shape
                    )
                    % 8
                ) + 1
                expected_grad = np.zeros(positions.size, dtype=np.float32)
                np.add.at(
                    expected_grad, selected.reshape(-1), grad_np.reshape(-1)
                )
                expected_grad = expected_grad.reshape(shape)

                x = paddle.to_tensor(x_np, dtype=dtype)
                x.stop_gradient = False
                out = fn(x, paddle.to_tensor(index_np))
                self.assertEqual(list(out.shape), list(expected.shape))
                np.testing.assert_array_equal(out.numpy(), expected)

                out.backward(paddle.to_tensor(grad_np, dtype=dtype))
                np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_negative_stride_view_grad(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        try:
            for dtype in ('float32', 'float64'):
                for shape in self.SHAPES:
                    self._run(dtype, shape)
        finally:
            paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
