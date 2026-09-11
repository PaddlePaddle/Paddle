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

import hashlib
import unittest

import numpy as np

import paddle
from paddle.base import core


def coordinates(in_size, out_size):
    scale = np.float32(in_size / out_size)
    source = np.maximum(
        (np.arange(out_size, dtype=np.float32) + np.float32(0.5)) * scale
        - np.float32(0.5),
        np.float32(0),
    )
    lower = source.astype(np.int64)
    upper = np.minimum(lower + 1, in_size - 1)
    weight = np.clip(source - lower.astype(np.float32), 0, 1)
    return lower, upper, weight


def reference_forward(x, size):
    """Separate NumPy ufuncs preserve the decomposition's FP32 rounding."""
    y0, y1, wy = coordinates(x.shape[2], size[0])
    x0, x1, wx = coordinates(x.shape[3], size[1])
    v00 = x[:, :, y0[:, None], x0]
    v01 = x[:, :, y0[:, None], x1]
    v10 = x[:, :, y1[:, None], x0]
    v11 = x[:, :, y1[:, None], x1]
    top = v00 + (v01 - v00) * wx
    bottom = v10 + (v11 - v10) * wx
    return top + (bottom - top) * wy[:, None]


def reference_backward(shape, grad):
    """Independent stable-index reduction, including the 32-lane sum tree."""
    y0, y1, wy = coordinates(shape[2], grad.shape[2])
    x0, x1, wx = coordinates(shape[3], grad.shape[3])
    bottom = grad * wy[:, None]
    top = grad - bottom
    contributions = [
        top - top * wx,
        top * wx,
        bottom - bottom * wx,
        bottom * wx,
    ]
    parts = []
    for (ys, xs), contribution in zip(
        [(y0, x0), (y0, x1), (y1, x0), (y1, x1)], contributions
    ):
        indices = (ys[:, None] * shape[3] + xs).reshape(-1)
        values = contribution.reshape(*shape[:2], -1)
        part = np.zeros((*shape[:2], shape[2] * shape[3]), np.float32)
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
    return ((parts[3] + parts[2]) + parts[1]) + parts[0]


def round_to_bfloat16(x):
    # Finite normal-distributed inputs, rounded to nearest with ties to even.
    bits = x.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def md5(x):
    return hashlib.md5(x.tobytes()).hexdigest()


@unittest.skipUnless(
    core.is_compiled_with_cuda() and not core.is_compiled_with_rocm(),
    "The accuracy-compatible bilinear kernel requires CUDA",
)
class TestBilinearAccuracyCompatible(unittest.TestCase):
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

    def evaluate(self, data, size, grad):
        x = paddle.to_tensor(data, stop_gradient=False)
        y = paddle.nn.functional.interpolate(
            x, size=size, mode='bilinear', align_corners=False, align_mode=0
        )
        dx = paddle.grad(y, x, grad_outputs=paddle.to_tensor(grad))[0]
        return y.numpy(), dx.numpy()

    def test_random_gradients_and_repeats(self):
        rng = np.random.default_rng(2026)
        for size in [(31, 68), (20, 35), (19, 37), (30, 45)]:
            with self.subTest(size=size):
                data = rng.standard_normal((2, 3, 52, 52)).astype(np.float32)
                grad = rng.standard_normal((2, 3, *size)).astype(np.float32)
                expected_y = reference_forward(data, size)
                expected_dx = reference_backward(data.shape, grad)
                for _ in range(3):
                    y, dx = self.evaluate(data, size, grad)
                    self.assert_bitwise_equal(y, expected_y)
                    self.assert_bitwise_equal(dx, expected_dx)

    def test_boundaries_and_large_reductions(self):
        rng = np.random.default_rng(7)
        for source, size in [
            ((1, 1), (20, 20)),
            ((1, 7), (1, 40)),
            ((7, 1), (40, 1)),
            ((2, 3), (120, 90)),
            ((3, 4), (6, 8)),
            ((3, 4), (7, 9)),
            ((7, 9), (1, 1)),
            ((7, 9), (3, 4)),
            ((7, 9), (7, 4)),
            ((7, 9), (3, 9)),
            ((7, 9), (7, 9)),
        ]:
            with self.subTest(source=source, size=size):
                data = rng.standard_normal((1, 2, *source)).astype(np.float32)
                grad = rng.standard_normal((1, 2, *size)).astype(np.float32)
                # Sparse signed gradients exercise cancellation and zero sums.
                grad[..., ::2, ::2] = 0
                y, dx = self.evaluate(data, size, grad)
                self.assert_bitwise_equal(y, reference_forward(data, size))
                self.assert_bitwise_equal(
                    dx, reference_backward(data.shape, grad)
                )

    def test_positional_embedding_md5(self):
        # Generated with torch 2.12.1+cu129 on CUDA, deterministic=True,
        # default_rng(0), BF16 -> FP32 input and ones_like(output) gradients.
        expected = {
            (31, 68): (
                '05f31786fbfd2f17bc3587a5872335f2',
                '7407cf8a481e119de7e43689d6b634a0',
            ),
            (20, 35): (
                'bfc8e76343b3b98acef27e373ef8f870',
                '1161b1d927bce439cdc7589cfecb6ca8',
            ),
            (19, 37): (
                'd116da26f32a919993158d9012a90cb1',
                '1327ab280c2d20460d5469cf54224c25',
            ),
            (30, 45): (
                'c8b2eec98469fbd185959fd10fac7cf5',
                'd749fc8113211af85d17d1756d9a681e',
            ),
        }
        data = round_to_bfloat16(
            np.random.default_rng(0)
            .standard_normal((1, 768, 52, 52))
            .astype(np.float32)
        )
        self.assertEqual(md5(data), 'e4db7bd4a16d58be5e501d444b17e21a')
        for size, (forward_hash, backward_hash) in expected.items():
            with self.subTest(size=size):
                y, dx = self.evaluate(
                    data, size, np.ones((1, 768, *size), np.float32)
                )
                self.assertEqual(md5(y), forward_hash)
                self.assertEqual(md5(dx), backward_hash)

    def test_pir(self):
        data = (
            np.random.default_rng(12)
            .standard_normal((1, 2, 7, 9))
            .astype(np.float32)
        )
        grad = (
            np.random.default_rng(13)
            .standard_normal((1, 2, 3, 4))
            .astype(np.float32)
        )
        with paddle.pir_utils.IrGuard():
            paddle.enable_static()
            main, startup = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', data.shape, 'float32')
                x.stop_gradient = False
                g = paddle.static.data('g', grad.shape, 'float32')
                y = paddle.nn.functional.interpolate(
                    x, size=[3, 4], mode='bilinear', align_corners=False
                )
                dx = paddle.static.gradients([y], [x], [g])[0]
            executor = paddle.static.Executor(paddle.CUDAPlace(0))
            executor.run(startup)
            actual_y, actual_dx = executor.run(
                main, feed={'x': data, 'g': grad}, fetch_list=[y, dx]
            )
        self.assert_bitwise_equal(actual_y, reference_forward(data, (3, 4)))
        self.assert_bitwise_equal(
            actual_dx, reference_backward(data.shape, grad)
        )

    def test_transposed_input(self):
        data = (
            np.random.default_rng(5)
            .standard_normal((1, 3, 52, 52))
            .astype(np.float32)
        )
        grad = (
            np.random.default_rng(6)
            .standard_normal((1, 3, 31, 68))
            .astype(np.float32)
        )
        base = paddle.to_tensor(
            data.transpose(0, 2, 3, 1).copy(), stop_gradient=False
        )
        x = base.transpose([0, 3, 1, 2])
        y = paddle.nn.functional.interpolate(
            x, size=[31, 68], mode='bilinear', align_corners=False
        )
        dx = paddle.grad(y, base, grad_outputs=paddle.to_tensor(grad))[0]
        self.assert_bitwise_equal(y.numpy(), reference_forward(data, (31, 68)))
        self.assert_bitwise_equal(
            dx.numpy().transpose(0, 3, 1, 2).copy(),
            reference_backward(data.shape, grad),
        )

    def test_linear_forward_unchanged(self):
        rng = np.random.default_rng(0)
        # Keep the same RNG sequence as the 2D/1D positional-embedding probe.
        rng.standard_normal((1, 768, 52, 52))
        data = round_to_bfloat16(
            rng.standard_normal((1, 768, 1500)).astype(np.float32)
        )
        x = paddle.to_tensor(data)
        expected = {
            1245: 'f2a1388fcc96a79ba91f19a8423e8a08',
            79: '0917f4636ef5910933559aa9096eb88f',
            501: 'fa2dda66f1fae6b39f50f5300f465e72',
            387: 'd1e3bc4f46eaab9de7ae5530db8a91f2',
        }
        for size, digest in expected.items():
            with self.subTest(size=size):
                result = paddle.nn.functional.interpolate(
                    x,
                    size=[size],
                    mode='linear',
                    align_corners=False,
                    data_format='NCW',
                )
                self.assertEqual(md5(result.numpy()), digest)

    def test_other_modes_keep_existing_path(self):
        data = (
            np.random.default_rng(1)
            .standard_normal((1, 2, 7, 9))
            .astype(np.float32)
        )
        cases = [
            (data, {"size": [3, 4], "mode": 'bilinear', "align_corners": True}),
            (data, {"size": [3, 4], "mode": 'bilinear', "align_mode": 1}),
            (data, {"size": [3, 4], "mode": 'nearest'}),
            (data, {"size": [3, 4], "mode": 'bicubic'}),
            (data, {"scale_factor": 0.75, "mode": 'bilinear'}),
            (
                data.transpose(0, 2, 3, 1).copy(),
                {"size": [3, 4], "mode": 'bilinear', "data_format": 'NHWC'},
            ),
            (
                data.astype(np.float64),
                {"size": [3, 4], "mode": 'bilinear'},
            ),
        ]
        for values, kwargs in cases:
            with self.subTest(kwargs=kwargs, dtype=values.dtype):
                x = paddle.to_tensor(values)
                paddle.set_flags(
                    {'FLAGS_use_accuracy_compatible_kernel': False}
                )
                expected = paddle.nn.functional.interpolate(x, **kwargs).numpy()
                paddle.set_flags({'FLAGS_use_accuracy_compatible_kernel': True})
                actual = paddle.nn.functional.interpolate(x, **kwargs).numpy()
                self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == '__main__':
    unittest.main()
