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


class TestLightweightFloatToComplex(unittest.TestCase):
    """Test casting from lightweight float formats (float8, float16, bfloat16) to complex types."""

    def test_float16_to_complex(self):
        """Test float16 to complex64/complex128 conversion."""
        paddle.set_device('cpu')

        r_fp16 = np.random.random(size=[10, 10]).astype('float16')
        r_fp16_t = paddle.to_tensor(r_fp16, dtype='float16')

        # Test dtype conversion
        self.assertEqual(r_fp16_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(r_fp16_t.cast('complex128').dtype, paddle.complex128)

        # Verify the real part is correct
        np.testing.assert_allclose(
            r_fp16_t.cast('complex64').real().numpy(),
            r_fp16.astype('float32'),
            rtol=1e-03,
        )
        np.testing.assert_allclose(
            r_fp16_t.cast('complex128').real().numpy(),
            r_fp16.astype('float64'),
            rtol=1e-03,
        )

        # Verify the imaginary part is zero
        np.testing.assert_array_equal(
            r_fp16_t.cast('complex64').imag().numpy(),
            np.zeros([10, 10], dtype='float32'),
        )
        np.testing.assert_array_equal(
            r_fp16_t.cast('complex128').imag().numpy(),
            np.zeros([10, 10], dtype='float64'),
        )

    def test_bfloat16_to_complex(self):
        """Test bfloat16 to complex64/complex128 conversion."""
        paddle.set_device('cpu')

        r_bf16 = np.random.random(size=[10, 10]).astype('float32')
        r_bf16_t = paddle.to_tensor(r_bf16, dtype='bfloat16')

        # Test dtype conversion
        self.assertEqual(r_bf16_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(r_bf16_t.cast('complex128').dtype, paddle.complex128)

        # Verify the real part is correct
        np.testing.assert_allclose(
            r_bf16_t.cast('complex64').real().numpy(),
            r_bf16_t.cast('float32').numpy(),
            rtol=1e-02,
        )
        np.testing.assert_allclose(
            r_bf16_t.cast('complex128').real().numpy(),
            r_bf16_t.cast('float64').numpy(),
            rtol=1e-02,
        )

        # Verify the imaginary part is zero
        np.testing.assert_array_equal(
            r_bf16_t.cast('complex64').imag().numpy(),
            np.zeros([10, 10], dtype='float32'),
        )
        np.testing.assert_array_equal(
            r_bf16_t.cast('complex128').imag().numpy(),
            np.zeros([10, 10], dtype='float64'),
        )

    def test_float8_e4m3fn_to_complex(self):
        """Test float8_e4m3fn to complex64/complex128 conversion."""
        paddle.set_device('cpu')

        r_fp32 = np.random.uniform(1.0, 10.0, size=[10, 10]).astype('float32')
        r_fp32_t = paddle.to_tensor(r_fp32)
        r_fp8_e4m3fn_t = r_fp32_t.astype('float8_e4m3fn')

        # Test dtype conversion
        self.assertEqual(
            r_fp8_e4m3fn_t.cast('complex64').dtype, paddle.complex64
        )
        self.assertEqual(
            r_fp8_e4m3fn_t.cast('complex128').dtype, paddle.complex128
        )

        # Verify the real part matches the float32 version
        np.testing.assert_allclose(
            r_fp8_e4m3fn_t.cast('complex64').real().numpy(),
            r_fp8_e4m3fn_t.cast('float32').numpy(),
            rtol=1e-02,
        )
        np.testing.assert_allclose(
            r_fp8_e4m3fn_t.cast('complex128').real().numpy(),
            r_fp8_e4m3fn_t.cast('float64').numpy(),
            rtol=1e-02,
        )

        # Verify the imaginary part is zero
        np.testing.assert_array_equal(
            r_fp8_e4m3fn_t.cast('complex64').imag().numpy(),
            np.zeros([10, 10], dtype='float32'),
        )
        np.testing.assert_array_equal(
            r_fp8_e4m3fn_t.cast('complex128').imag().numpy(),
            np.zeros([10, 10], dtype='float64'),
        )

    def test_float8_e5m2_to_complex(self):
        """Test float8_e5m2 to complex64/complex128 conversion."""
        paddle.set_device('cpu')

        r_fp32 = np.random.uniform(1.0, 10.0, size=[10, 10]).astype('float32')
        r_fp32_t = paddle.to_tensor(r_fp32)
        r_fp8_e5m2_t = r_fp32_t.astype('float8_e5m2')

        # Test dtype conversion
        self.assertEqual(r_fp8_e5m2_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(
            r_fp8_e5m2_t.cast('complex128').dtype, paddle.complex128
        )

        # Verify the real part matches the float32 version
        np.testing.assert_allclose(
            r_fp8_e5m2_t.cast('complex64').real().numpy(),
            r_fp8_e5m2_t.cast('float32').numpy(),
            rtol=1e-02,
        )
        np.testing.assert_allclose(
            r_fp8_e5m2_t.cast('complex128').real().numpy(),
            r_fp8_e5m2_t.cast('float64').numpy(),
            rtol=1e-02,
        )

        # Verify the imaginary part is zero
        np.testing.assert_array_equal(
            r_fp8_e5m2_t.cast('complex64').imag().numpy(),
            np.zeros([10, 10], dtype='float32'),
        )
        np.testing.assert_array_equal(
            r_fp8_e5m2_t.cast('complex128').imag().numpy(),
            np.zeros([10, 10], dtype='float64'),
        )


if __name__ == '__main__':
    unittest.main()
