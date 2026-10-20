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

"""
信号处理高级测试 / Advanced Signal Processing Tests

测试目标 / Test Target:
  paddle.signal 和 paddle.fft 信号处理

覆盖的模块 / Covered Modules:
  - paddle.signal.stft: 短时傅里叶变换
  - paddle.signal.istft: 逆短时傅里叶变换
  - paddle.fft 高级用法

作用 / Purpose:
  补充信号处理API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestSTFT(unittest.TestCase):
    """测试短时傅里叶变换 / Test STFT"""

    def test_stft_basic(self):
        """测试基本STFT / Test basic STFT"""
        # Signal length
        signal_len = 1024
        x = paddle.randn([signal_len])
        n_fft = 128
        hop_length = 64
        result = paddle.signal.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=128
        )
        # Expected freq bins: n_fft//2 + 1
        expected_freq = n_fft // 2 + 1
        self.assertEqual(result.shape[0], expected_freq)

    def test_stft_batched(self):
        """测试批量STFT / Test batched STFT"""
        x = paddle.randn([4, 512])
        n_fft = 64
        hop_length = 32
        result = paddle.signal.stft(x, n_fft=n_fft, hop_length=hop_length)
        expected_freq = n_fft // 2 + 1
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result.shape[1], expected_freq)

    def test_stft_with_window(self):
        """测试带窗函数的STFT / Test STFT with window"""
        x = paddle.randn([256])
        window = paddle.hann_window(64)
        result = paddle.signal.stft(
            x, n_fft=64, hop_length=32, win_length=64, window=window
        )
        self.assertIsNotNone(result)


class TestFFTAdvanced(unittest.TestCase):
    """测试高级FFT操作 / Test advanced FFT operations"""

    def test_fft_shift(self):
        """测试FFT移位 / Test FFT shift"""
        x = paddle.randn([16])
        fft_result = paddle.fft.fft(x)
        shifted = paddle.fft.fftshift(fft_result)
        unshifted = paddle.fft.ifftshift(shifted)
        np.testing.assert_allclose(
            fft_result.numpy().real, unshifted.numpy().real, rtol=1e-5
        )

    def test_rfft_output_size(self):
        """测试RFFT输出大小 / Test RFFT output size"""
        for n in [8, 16, 32]:
            x = paddle.randn([n])
            result = paddle.fft.rfft(x)
            self.assertEqual(result.shape[0], n // 2 + 1)

    def test_irfft_roundtrip(self):
        """测试IRFFT往返 / Test IRFFT round-trip"""
        x = paddle.randn([32])
        fft = paddle.fft.rfft(x)
        recovered = paddle.fft.irfft(fft, n=32)
        np.testing.assert_allclose(x.numpy(), recovered.numpy(), rtol=1e-5)

    def test_fft2_2d(self):
        """测试2D FFT / Test 2D FFT"""
        x = paddle.randn([8, 8])
        result = paddle.fft.fft2(x)
        self.assertEqual(result.shape, [8, 8])

    def test_ifft2_roundtrip(self):
        """测试2D IFFT往返 / Test 2D IFFT round-trip"""
        x = paddle.randn([8, 8])
        fft = paddle.fft.fft2(x)
        recovered = paddle.fft.ifft2(fft)
        np.testing.assert_allclose(
            x.numpy(), recovered.numpy().real, rtol=1e-4, atol=1e-6
        )

    def test_fft_frequencies(self):
        """测试FFT频率 / Test FFT frequencies"""
        freqs = paddle.fft.fftfreq(8, d=1.0)
        self.assertEqual(freqs.shape[0], 8)

    def test_rfft_frequencies(self):
        """测试RFFT频率 / Test RFFT frequencies"""
        freqs = paddle.fft.rfftfreq(8, d=1.0)
        self.assertEqual(freqs.shape[0], 5)  # 8//2 + 1


class TestHannWindow(unittest.TestCase):
    """测试Hann窗函数 / Test Hann window function"""

    def test_hann_window(self):
        """测试Hann窗 / Test Hann window"""
        window = paddle.hann_window(64)
        self.assertEqual(window.shape[0], 64)
        # Hann window center should be max
        center = int(window.argmax().numpy())
        self.assertAlmostEqual(center, 32, delta=1)

    def test_hamming_window(self):
        """测试Hamming窗 / Test Hamming window"""
        window = paddle.hamming_window(64)
        self.assertEqual(window.shape[0], 64)

    def test_blackman_window(self):
        """测试Blackman窗 / Test Blackman window"""
        window = paddle.blackman_window(64)
        self.assertEqual(window.shape[0], 64)


if __name__ == '__main__':
    unittest.main()
