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

import paddle
from paddle.audio.functional.window import get_window


class TestWindowFunctions(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")

    def test_hamming_alpha_beta_transform_and_requires_grad(self):
        N = 16
        w0 = get_window('hamming', N, fftbins=True, dtype='float64')

        # Custom alpha/beta, verify linear transformation A + B * w0
        alpha, beta = 0.60, 0.40
        w = paddle.hamming_window(
            N,
            periodic=True,
            alpha=alpha,
            beta=beta,
            dtype='float64',
            requires_grad=True,
        )
        self.assertEqual(w.dtype, paddle.float64)
        self.assertFalse(w.stop_gradient)
        # Linear equivalence: w ≈ A + B * w0
        alpha0, beta0 = 0.54, 0.46
        B = beta / beta0
        A = alpha - B * alpha0
        self.assertTrue(paddle.allclose(w, A + B * w0, atol=1e-12))

    def test_hamming_layout_warning(self):
        N = 8
        # Pass layout != None to trigger warning branch (ignored)
        w = paddle.hamming_window(
            N,
            periodic=False,
            alpha=0.54,
            beta=0.46,
            dtype='float32',
            layout='strided',
            device='cpu',
            requires_grad=False,
        )
        self.assertEqual(w.dtype, paddle.float32)
        self.assertTrue(w.stop_gradient)
        self.assertEqual(list(w.shape), [N])

    def test_hamming_device_gpu_pin_memory(self):
        if paddle.is_compiled_with_cuda():
            N = 12
            # Explicitly set device to cuda:0 / gpu:0 should work (PlaceLike supports str)
            w = paddle.hamming_window(
                N,
                periodic=True,
                alpha=0.54,
                beta=0.46,
                dtype='float32',
                layout=None,
                device='gpu:0',
                pin_memory=True,
                requires_grad=None,
            )
            self.assertEqual(list(w.shape), [N])
            self.assertIn('gpu', str(w.place))

    def test_hann_basic_paths(self):
        N = 10
        # Pass layout=None; set requires_grad=True
        w = paddle.hann_window(
            N,
            periodic=True,
            dtype='float64',
            layout=None,
            device='cpu',
            requires_grad=True,
        )
        self.assertEqual(list(w.shape), [N])
        self.assertFalse(w.stop_gradient)

        # Test layout != None
        w2 = paddle.hann_window(
            N,
            periodic=False,
            dtype='float32',
            layout='strided',
            device='cpu',
            requires_grad=False,
        )
        self.assertEqual(w2.dtype, paddle.float32)
        self.assertTrue(w2.stop_gradient)

    def test_blackman_and_bartlett_basic(self):
        N = 9
        wb = paddle.blackman_window(
            N,
            periodic=True,
            dtype='float64',
            layout=None,
            device=None,
            requires_grad=None,
        )
        self.assertEqual(list(wb.shape), [N])

        wl = paddle.bartlett_window(
            N,
            periodic=False,
            dtype='float32',
            layout='strided',
            device='cpu',
            requires_grad=True,
        )
        self.assertEqual(list(wl.shape), [N])
        self.assertFalse(wl.stop_gradient)

    def test_kaiser_beta_and_paths(self):
        N = 7
        beta = 6.0
        w = paddle.kaiser_window(
            N,
            periodic=True,
            beta=beta,
            dtype='float64',
            layout=None,
            device=None,
            requires_grad=None,
        )
        self.assertEqual(list(w.shape), [N])

        # Test layout != None + requires_grad
        w2 = paddle.kaiser_window(
            N,
            periodic=False,
            beta=8.0,
            dtype='float32',
            layout='strided',
            device='cpu',
            requires_grad=False,
        )
        self.assertEqual(w2.dtype, paddle.float32)
        self.assertTrue(w2.stop_gradient)

    def test_hamming_periodic_vs_symmetric(self):
        # Test periodic True/False length handling (DFT symmetry/periodic)
        N = 11
        w_per = paddle.hamming_window(
            N, periodic=True, alpha=0.54, beta=0.46, dtype='float64'
        )
        w_sym = paddle.hamming_window(
            N, periodic=False, alpha=0.54, beta=0.46, dtype='float64'
        )
        self.assertEqual(list(w_per.shape), [N])
        self.assertEqual(list(w_sym.shape), [N])


if __name__ == '__main__':
    unittest.main()
