# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from test_bilinear_interp_accuracy_compatible import coordinates, round_to

import paddle
from paddle.base import core


def axes(shape, size, align_corners, scale_factor):
    return [
        coordinates(shape[k], size[k], align_corners, scale_factor[k])
        for k in range(3)
    ]


def broadcast(values, k):
    return values.reshape([-1 if i == k else 1 for i in range(3)])


def reference_forward(
    x, size, align_corners=False, scale_factor=(None, None, None)
):
    """Separate NumPy ufuncs preserve the decomposition's FP32 rounding."""
    (d0, d1, wd), (h0, h1, wh), (w0, w1, ww) = axes(
        x.shape[2:], size, align_corners, scale_factor
    )
    vs = [
        x[:, :, broadcast(d, 0), broadcast(h, 1), broadcast(w, 2)]
        for d, h, w in itertools.product([d0, d1], [h0, h1], [w0, w1])
    ]
    # Lerp W, then H, then D, as _upsample_linear does.
    for weight in [broadcast(ww, 2), broadcast(wh, 1), broadcast(wd, 0)]:
        vs = [a + (b - a) * weight for a, b in zip(vs[::2], vs[1::2])]
    return vs[0]


def reference_backward(
    shape, grad, align_corners=False, scale_factor=(None, None, None)
):
    """Independent stable-index reduction, including the 32-lane sum tree."""
    in_d, in_h, in_w = shape[2:]
    (d0, d1, wd), (h0, h1, wh), (w0, w1, ww) = axes(
        shape[2:], grad.shape[2:], align_corners, scale_factor
    )
    weights = [broadcast(wd, 0), broadcast(wh, 1), broadcast(ww, 2)]
    parts = []
    for corner, (d, h, w) in enumerate(
        itertools.product([d0, d1], [h0, h1], [w0, w1])
    ):
        contribution = grad
        for k, weight in enumerate(weights):
            upper = contribution * weight
            if (corner >> (2 - k)) & 1:
                contribution = upper
            else:
                contribution = contribution - upper
        indices = (
            (broadcast(d, 0) * in_h + broadcast(h, 1)) * in_w + broadcast(w, 2)
        ).reshape(-1)
        values = contribution.reshape(*shape[:2], -1)
        part = np.zeros((*shape[:2], in_d * in_h * in_w), np.float32)
        # Filtering the original row-major positions preserves stable-sort order.
        for index in np.unique(indices):
            selected = values[..., indices == index]
            count = selected.shape[-1]
            full = count // 32 * 32
            lanes = np.zeros((*shape[:2], 32), np.float32)
            for start in range(0, full, 32):
                lanes += selected[..., start : start + 32]
            for offset in (16, 8, 4, 2, 1):
                lanes[..., : 32 - offset] = (
                    lanes[..., : 32 - offset] + lanes[..., offset:]
                )
            total = lanes[..., 0].copy()
            for tail in range(full, count):
                total += selected[..., tail]
            part[..., index] = np.float32(0) + total
        parts.append(part.reshape(shape))
    result = parts[7]
    for part in reversed(parts[:7]):
        result = result + part
    return result


@unittest.skipUnless(
    core.is_compiled_with_cuda() and not core.is_compiled_with_rocm(),
    "The accuracy-compatible trilinear kernel requires CUDA",
)
class TestTrilinearAccuracyCompatible(unittest.TestCase):
    def setUp(self):
        self.flags = paddle.get_flags(['FLAGS_use_accuracy_compatible_kernel'])
        self.device = paddle.device.get_device()
        paddle.disable_static()
        paddle.set_device('gpu:0')
        paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': True})

    def tearDown(self):
        paddle.disable_static()
        paddle.set_flags(self.flags)
        paddle.set_device(self.device)

    def assert_bitwise_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(
            actual.view(np.uint32), expected.view(np.uint32)
        )

    def evaluate(
        self, data, size, grad, align_corners=False, align_mode=0, **kwargs
    ):
        x = paddle.to_tensor(data, stop_gradient=False)
        y = paddle.nn.functional.interpolate(
            x,
            size=size,
            mode='trilinear',
            align_corners=align_corners,
            align_mode=align_mode,
            data_format='NCDHW',
            **kwargs,
        )
        dx = paddle.grad(y, x, grad_outputs=paddle.to_tensor(grad))[0]
        return y.numpy(), dx.numpy()

    def check(self, source, size, align_corners=False, seed=0, repeats=1):
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((2, 3, *source)).astype(np.float32)
        grad = rng.standard_normal((2, 3, *size)).astype(np.float32)
        # Sparse signed gradients exercise cancellation and zero sums.
        grad[..., ::2, ::2, ::2] = 0
        expected_y = reference_forward(data, size, align_corners)
        expected_dx = reference_backward(data.shape, grad, align_corners)
        for _ in range(repeats):
            y, dx = self.evaluate(data, size, grad, align_corners)
            self.assert_bitwise_equal(y, expected_y)
            self.assert_bitwise_equal(dx, expected_dx)

    def test_random_gradients_and_repeats(self):
        for source, size in [
            ((4, 8, 8), (8, 16, 16)),
            ((5, 7, 9), (3, 4, 5)),
            ((6, 6, 6), (7, 11, 13)),
            ((8, 16, 16), (4, 8, 8)),
        ]:
            with self.subTest(source=source, size=size):
                self.check(source, size, repeats=3)

    def test_boundaries_and_large_reductions(self):
        for source, size in [
            ((1, 1, 1), (3, 4, 5)),
            ((1, 2, 3), (2, 1, 5)),
            # More than 2x along every axis takes the warp reduction path.
            ((2, 2, 3), (9, 20, 30)),
            ((3, 4, 5), (1, 1, 1)),
            ((3, 4, 5), (3, 4, 2)),
        ]:
            with self.subTest(source=source, size=size):
                self.check(source, size, seed=7)

    def test_align_corners(self):
        for source, size in [
            ((8, 16, 16), (4, 8, 8)),
            ((4, 8, 8), (8, 16, 16)),
            ((1, 1, 1), (3, 4, 5)),
            ((2, 2, 3), (9, 20, 30)),
            ((3, 4, 5), (1, 1, 1)),
        ]:
            with self.subTest(source=source, size=size):
                self.check(source, size, align_corners=True, seed=11)
        # align_mode is ignored when align_corners=True.
        rng = np.random.default_rng(12)
        data = rng.standard_normal((1, 2, 3, 4, 5)).astype(np.float32)
        grad = rng.standard_normal((1, 2, 5, 6, 7)).astype(np.float32)
        y, dx = self.evaluate(data, (5, 6, 7), grad, True, align_mode=1)
        self.assert_bitwise_equal(y, reference_forward(data, (5, 6, 7), True))
        self.assert_bitwise_equal(
            dx, reference_backward(data.shape, grad, True)
        )

    def test_scale_factor(self):
        # Without recompute, torch maps coordinates by 1 / scale_factor.
        rng = np.random.default_rng(43)
        source = (4, 5, 7)
        data = rng.standard_normal((2, 3, *source)).astype(np.float32)
        for scale in [
            2.0,
            (1.5, 0.75, 2.25),
            # Output size equals input size, but coordinates still move.
            (1.1, 1.1, 1.1),
            # Slightly above 2 while the output stays within 2x the input.
            (2.05, 2.05, 2.1),
            # Beyond 2x the warp reduction path is used.
            (2.5, 3.0, 2.0),
        ]:
            factors = (scale,) * 3 if isinstance(scale, float) else scale
            size = [int(dim * f) for dim, f in zip(source, factors)]
            grad = rng.standard_normal((2, 3, *size)).astype(np.float32)
            for align_corners in [False, True]:
                expected_y = reference_forward(
                    data, size, align_corners, factors
                )
                expected_dx = reference_backward(
                    data.shape, grad, align_corners, factors
                )
                for recompute in [False, None]:
                    with self.subTest(
                        scale=scale,
                        align_corners=align_corners,
                        recompute=recompute,
                    ):
                        y, dx = self.evaluate(
                            data,
                            None,
                            grad,
                            align_corners,
                            scale_factor=scale,
                            recompute_scale_factor=recompute,
                        )
                        self.assert_bitwise_equal(y, expected_y)
                        self.assert_bitwise_equal(dx, expected_dx)

    def test_recomputed_scale_matches_output_size(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 3, 4, 5, 7)).astype(np.float32)
        for scale in [1.5, (0.5, 0.75, 1.5)]:
            with self.subTest(scale=scale):
                factors = (scale,) * 3 if isinstance(scale, float) else scale
                size = [
                    int(np.floor(dim * f))
                    for dim, f in zip(data.shape[2:], factors)
                ]
                grad = rng.standard_normal((2, 3, *size)).astype(np.float32)
                y, dx = self.evaluate(
                    data,
                    None,
                    grad,
                    scale_factor=scale,
                    recompute_scale_factor=True,
                )
                self.assert_bitwise_equal(y, reference_forward(data, size))
                self.assert_bitwise_equal(
                    dx, reference_backward(data.shape, grad)
                )

    def test_scale_tensor_pir(self):
        # FP32-exact factors, since the Scale input is read as FP32.
        factors = (1.5, 0.75, 2.0)
        data = (
            np.random.default_rng(14)
            .standard_normal((1, 2, 4, 5, 7))
            .astype(np.float32)
        )
        size = [int(dim * f) for dim, f in zip(data.shape[2:], factors)]
        grad = (
            np.random.default_rng(15)
            .standard_normal((1, 2, *size))
            .astype(np.float32)
        )
        with paddle.pir_utils.IrGuard():
            paddle.enable_static()
            main, startup = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', data.shape, 'float32')
                x.stop_gradient = False
                g = paddle.static.data('g', grad.shape, 'float32')
                s = paddle.static.data('s', [3], 'float32')
                y = paddle.nn.functional.interpolate(
                    x,
                    scale_factor=s,
                    mode='trilinear',
                    align_corners=False,
                    data_format='NCDHW',
                )
                dx = paddle.static.gradients([y], [x], [g])[0]
            executor = paddle.static.Executor(paddle.CUDAPlace(0))
            executor.run(startup)
            actual_y, actual_dx = executor.run(
                main,
                feed={
                    'x': data,
                    'g': grad,
                    's': np.array(factors, np.float32),
                },
                fetch_list=[y, dx],
            )
        self.assert_bitwise_equal(
            actual_y, reference_forward(data, size, False, factors)
        )
        self.assert_bitwise_equal(
            actual_dx, reference_backward(data.shape, grad, False, factors)
        )

    def test_ndhwc_and_half_precision(self):
        # NDHWC only changes addressing; torch sums in logical index order.
        # FP16/BF16 are widened to FP32 and rounded back once.
        rng = np.random.default_rng(16)
        for dtype, layout, source, size, kwargs in [
            ('float32', 'NDHWC', (4, 8, 8), (8, 16, 16), {}),
            ('float32', 'NDHWC', (2, 2, 3), (9, 20, 30), {}),
            ('float32', 'NDHWC', (8, 6, 6), (4, 3, 5), {"align_corners": True}),
            ('float32', 'NDHWC', (4, 5, 7), None, {"scale_factor": 2.0}),
            ('float16', 'NCDHW', (4, 8, 8), (8, 16, 16), {}),
            ('bfloat16', 'NCDHW', (5, 7, 9), (3, 4, 5), {}),
            ('float16', 'NDHWC', (4, 5, 7), None, {"scale_factor": 1.5}),
        ]:
            with self.subTest(
                dtype=dtype, layout=layout, source=source, kwargs=kwargs
            ):
                align_corners = kwargs.get("align_corners", False)
                factor = kwargs.get("scale_factor")
                factors = (factor,) * 3
                out = size or [int(dim * factor) for dim in source]
                data = round_to(
                    rng.standard_normal((2, 5, *source)).astype(np.float32),
                    dtype,
                )
                grad = round_to(
                    rng.standard_normal((2, 5, *out)).astype(np.float32),
                    dtype,
                )
                ndhwc = layout == 'NDHWC'
                to_layout = (0, 2, 3, 4, 1) if ndhwc else (0, 1, 2, 3, 4)
                from_layout = (0, 4, 1, 2, 3) if ndhwc else (0, 1, 2, 3, 4)
                x = paddle.to_tensor(data.transpose(to_layout).copy())
                x = x.astype(dtype)
                x.stop_gradient = False
                y = paddle.nn.functional.interpolate(
                    x,
                    size=size,
                    mode='trilinear',
                    data_format=layout,
                    **kwargs,
                )
                dy = paddle.to_tensor(grad.transpose(to_layout).copy())
                dx = paddle.grad(y, x, grad_outputs=dy.astype(dtype))[0]
                self.assertEqual(y.dtype, x.dtype)
                self.assert_bitwise_equal(
                    y.astype('float32').numpy().transpose(from_layout).copy(),
                    round_to(
                        reference_forward(data, out, align_corners, factors),
                        dtype,
                    ),
                )
                self.assert_bitwise_equal(
                    dx.astype('float32').numpy().transpose(from_layout).copy(),
                    round_to(
                        reference_backward(
                            data.shape, grad, align_corners, factors
                        ),
                        dtype,
                    ),
                )

    def test_other_modes_keep_existing_path(self):
        data = (
            np.random.default_rng(1)
            .standard_normal((1, 2, 3, 4, 5))
            .astype(np.float32)
        )
        cases = [
            (data, {"size": [5, 6, 7], "mode": 'trilinear', "align_mode": 1}),
            (data, {"size": [5, 6, 7], "mode": 'nearest'}),
            (data.astype(np.float64), {"size": [5, 6, 7], "mode": 'trilinear'}),
            # Without a scale, an unchanged size is still a copy.
            (data, {"size": [3, 4, 5], "mode": 'trilinear'}),
        ]
        for values, kwargs in cases:
            with self.subTest(kwargs=kwargs, dtype=values.dtype):
                x = paddle.to_tensor(values)
                paddle.set_flags(
                    {'FLAGS_use_accuracy_compatible_kernel': False}
                )
                expected = paddle.nn.functional.interpolate(
                    x, data_format='NCDHW', **kwargs
                ).numpy()
                paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': True})
                actual = paddle.nn.functional.interpolate(
                    x, data_format='NCDHW', **kwargs
                ).numpy()
                self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == '__main__':
    unittest.main()
