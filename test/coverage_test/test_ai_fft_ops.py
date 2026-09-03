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
FFT操作单元测试 / FFT Operations Unit Tests

测试目标 / Test Target:
  paddle.fft 模块 (python/paddle/fft.py, 覆盖率约57.8%)

覆盖的模块 / Covered Modules:
  - paddle.fft: FFT变换相关API
  - 包括 fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn等

作用 / Purpose:
  覆盖fast Fourier transform (快速傅里叶变换)及相关逆变换的代码路径，
  补充未被原有测试覆盖的频域变换函数。
"""

import unittest

import numpy as np

import paddle
import paddle.fft

paddle.disable_static()


class TestFFTBasic(unittest.TestCase):
    """测试基本FFT操作 / Test basic FFT operations"""

    def setUp(self):
        """初始化测试数据 / Initialize test data"""
        self.x_real = paddle.to_tensor(np.random.randn(8).astype('float32'))
        self.x_2d = paddle.to_tensor(np.random.randn(4, 8).astype('float32'))
        self.x_complex = paddle.to_tensor(
            (np.random.randn(8) + 1j * np.random.randn(8)).astype('complex64')
        )

    def test_fft_1d(self):
        """测试1D FFT变换 / Test 1D FFT transform"""
        result = paddle.fft.fft(self.x_complex)
        self.assertEqual(result.shape, self.x_complex.shape)
        self.assertTrue(result.dtype == paddle.complex64)

    def test_fft_with_n(self):
        """测试带长度参数的FFT / Test FFT with n parameter"""
        result = paddle.fft.fft(self.x_complex, n=16)
        self.assertEqual(result.shape[-1], 16)

    def test_fft_with_axis(self):
        """测试沿指定轴的FFT / Test FFT along specified axis"""
        x = paddle.to_tensor(
            (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(
                'complex64'
            )
        )
        result = paddle.fft.fft(x, axis=0)
        self.assertEqual(result.shape, x.shape)
        result2 = paddle.fft.fft(x, axis=1)
        self.assertEqual(result2.shape, x.shape)

    def test_ifft_1d(self):
        """测试1D IFFT逆变换 / Test 1D IFFT inverse transform"""
        fft_result = paddle.fft.fft(self.x_complex)
        recovered = paddle.fft.ifft(fft_result)
        self.assertEqual(recovered.shape, self.x_complex.shape)

    def test_rfft_1d(self):
        """测试实数FFT / Test real FFT"""
        result = paddle.fft.rfft(self.x_real)
        # rfft output has n//2+1 complex elements
        expected_len = self.x_real.shape[-1] // 2 + 1
        self.assertEqual(result.shape[-1], expected_len)
        self.assertTrue(result.dtype in [paddle.complex64, paddle.complex128])

    def test_irfft_1d(self):
        """测试实数IFFT逆变换 / Test real IFFT inverse transform"""
        rfft_result = paddle.fft.rfft(self.x_real)
        recovered = paddle.fft.irfft(rfft_result)
        self.assertEqual(recovered.shape[-1], self.x_real.shape[-1])
        self.assertTrue(recovered.dtype in [paddle.float32, paddle.float64])

    def test_rfft_with_n(self):
        """测试带长度参数的实数FFT / Test real FFT with n"""
        result = paddle.fft.rfft(self.x_real, n=16)
        self.assertEqual(result.shape[-1], 16 // 2 + 1)


class TestFFT2D(unittest.TestCase):
    """测试2D FFT操作 / Test 2D FFT operations"""

    def setUp(self):
        """初始化2D测试数据 / Initialize 2D test data"""
        self.x_real_2d = paddle.to_tensor(
            np.random.randn(4, 8).astype('float32')
        )
        self.x_complex_2d = paddle.to_tensor(
            (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(
                'complex64'
            )
        )

    def test_fft2(self):
        """测试2D FFT / Test 2D FFT"""
        result = paddle.fft.fft2(self.x_complex_2d)
        self.assertEqual(result.shape, self.x_complex_2d.shape)

    def test_ifft2(self):
        """测试2D IFFT / Test 2D IFFT"""
        fft_result = paddle.fft.fft2(self.x_complex_2d)
        recovered = paddle.fft.ifft2(fft_result)
        self.assertEqual(recovered.shape, self.x_complex_2d.shape)

    def test_rfft2(self):
        """测试2D 实数FFT / Test 2D real FFT"""
        result = paddle.fft.rfft2(self.x_real_2d)
        # Last dim: n//2+1
        expected_last = self.x_real_2d.shape[-1] // 2 + 1
        self.assertEqual(result.shape[-1], expected_last)
        self.assertEqual(result.shape[-2], self.x_real_2d.shape[-2])

    def test_irfft2(self):
        """测试2D 实数IFFT / Test 2D real IFFT"""
        rfft_result = paddle.fft.rfft2(self.x_real_2d)
        recovered = paddle.fft.irfft2(rfft_result)
        self.assertEqual(recovered.shape[-1], self.x_real_2d.shape[-1])
        self.assertEqual(recovered.shape[-2], self.x_real_2d.shape[-2])

    def test_fft2_with_s(self):
        """测试带s参数的2D FFT / Test 2D FFT with s parameter"""
        result = paddle.fft.fft2(self.x_complex_2d, s=[8, 16])
        self.assertEqual(result.shape[-2], 8)
        self.assertEqual(result.shape[-1], 16)


class TestFFTND(unittest.TestCase):
    """测试N维FFT操作 / Test N-dimensional FFT operations"""

    def setUp(self):
        """初始化多维测试数据 / Initialize multi-dimensional test data"""
        self.x_real_3d = paddle.to_tensor(
            np.random.randn(2, 4, 8).astype('float32')
        )
        self.x_complex_3d = paddle.to_tensor(
            (np.random.randn(2, 4, 8) + 1j * np.random.randn(2, 4, 8)).astype(
                'complex64'
            )
        )

    def test_fftn(self):
        """测试N维FFT / Test N-dimensional FFT"""
        result = paddle.fft.fftn(self.x_complex_3d)
        self.assertEqual(result.shape, self.x_complex_3d.shape)

    def test_ifftn(self):
        """测试N维IFFT / Test N-dimensional IFFT"""
        fft_result = paddle.fft.fftn(self.x_complex_3d)
        recovered = paddle.fft.ifftn(fft_result)
        self.assertEqual(recovered.shape, self.x_complex_3d.shape)

    def test_rfftn(self):
        """测试N维实数FFT / Test N-dimensional real FFT"""
        result = paddle.fft.rfftn(self.x_real_3d)
        expected_last = self.x_real_3d.shape[-1] // 2 + 1
        self.assertEqual(result.shape[-1], expected_last)

    def test_irfftn(self):
        """测试N维实数IFFT / Test N-dimensional real IFFT"""
        rfft_result = paddle.fft.rfftn(self.x_real_3d)
        recovered = paddle.fft.irfftn(rfft_result)
        self.assertEqual(recovered.shape[-1], self.x_real_3d.shape[-1])

    def test_fftn_with_axes(self):
        """测试带axes参数的N维FFT / Test N-dimensional FFT with axes parameter"""
        result = paddle.fft.fftn(self.x_complex_3d, axes=[1, 2])
        self.assertEqual(result.shape, self.x_complex_3d.shape)


class TestFFTFreq(unittest.TestCase):
    """测试FFT频率函数 / Test FFT frequency functions"""

    def test_fftfreq(self):
        """测试FFT频率 / Test FFT frequency"""
        result = paddle.fft.fftfreq(8)
        self.assertEqual(result.shape[0], 8)

    def test_fftfreq_with_d(self):
        """测试带采样间隔的FFT频率 / Test FFT frequency with sample spacing"""
        result = paddle.fft.fftfreq(8, d=0.5)
        self.assertEqual(result.shape[0], 8)

    def test_rfftfreq(self):
        """测试实数FFT频率 / Test real FFT frequency"""
        result = paddle.fft.rfftfreq(8)
        self.assertEqual(result.shape[0], 8 // 2 + 1)

    def test_rfftfreq_with_d(self):
        """测试带采样间隔的实数FFT频率 / Test real FFT frequency with sample spacing"""
        result = paddle.fft.rfftfreq(8, d=0.5)
        self.assertEqual(result.shape[0], 8 // 2 + 1)

    def test_fftshift(self):
        """测试FFT频率移位 / Test FFT frequency shift"""
        x = paddle.to_tensor(np.fft.fftfreq(8).astype('float32'))
        result = paddle.fft.fftshift(x)
        self.assertEqual(result.shape, x.shape)

    def test_ifftshift(self):
        """测试IFFT频率移位 / Test IFFT frequency shift"""
        x = paddle.to_tensor(
            np.fft.fftshift(np.fft.fftfreq(8)).astype('float32')
        )
        result = paddle.fft.ifftshift(x)
        self.assertEqual(result.shape, x.shape)

    def test_hfft(self):
        """测试Hermitian FFT / Test Hermitian FFT"""
        x = paddle.to_tensor(
            (np.random.randn(5) + 1j * np.random.randn(5)).astype('complex64')
        )
        result = paddle.fft.hfft(x)
        # hfft output real, length 2*(n-1)
        self.assertTrue(result.dtype in [paddle.float32, paddle.float64])

    def test_ihfft(self):
        """测试Hermitian IFFT / Test Hermitian IFFT"""
        x = paddle.to_tensor(np.random.randn(8).astype('float32'))
        result = paddle.fft.ihfft(x)
        self.assertTrue(result.dtype in [paddle.complex64, paddle.complex128])


if __name__ == '__main__':
    unittest.main()
