# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestComplexCastOp(unittest.TestCase):
    def test_complex_to_real(self):
        r = np.random.random(size=[10, 10]) * 10
        i = np.random.random(size=[10, 10])

        c_t = paddle.to_tensor(r + i * 1j, dtype='complex64')

        self.assertEqual(c_t.cast('int64').dtype, paddle.int64)
        self.assertEqual(c_t.cast('int32').dtype, paddle.int32)
        self.assertEqual(c_t.cast('float32').dtype, paddle.float32)
        self.assertEqual(c_t.cast('float64').dtype, paddle.float64)
        self.assertEqual(c_t.cast('bool').dtype, paddle.bool)

        np.testing.assert_allclose(
            c_t.cast('int64').numpy(), r.astype('int64'), rtol=1e-05
        )
        np.testing.assert_allclose(
            c_t.cast('int32').numpy(), r.astype('int32'), rtol=1e-05
        )
        np.testing.assert_allclose(
            c_t.cast('float32').numpy(), r.astype('float32'), rtol=1e-05
        )
        np.testing.assert_allclose(
            c_t.cast('float64').numpy(), r.astype('float64'), rtol=1e-05
        )
        np.testing.assert_allclose(
            c_t.cast('bool').numpy(), r.astype('bool'), rtol=1e-05
        )

    def test_real_to_complex(self):
        r = np.random.random(size=[10, 10]) * 10
        r_t = paddle.to_tensor(r)

        self.assertEqual(r_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(r_t.cast('complex128').dtype, paddle.complex128)

        np.testing.assert_allclose(
            r_t.cast('complex64').real().numpy(), r, rtol=1e-05
        )
        np.testing.assert_allclose(
            r_t.cast('complex128').real().numpy(), r, rtol=1e-05
        )

    def test_complex64_complex128(self):
        r = np.random.random(size=[10, 10])
        i = np.random.random(size=[10, 10])

        c = r + i * 1j
        c_64 = paddle.to_tensor(c, dtype='complex64')
        c_128 = paddle.to_tensor(c, dtype='complex128')

        self.assertTrue(c_64.cast('complex128').dtype, paddle.complex128)
        self.assertTrue(c_128.cast('complex128').dtype, paddle.complex64)
        np.testing.assert_allclose(
            c_64.cast('complex128').numpy(), c_128.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            c_128.cast('complex128').numpy(), c_64.numpy(), rtol=1e-05
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "float16/bfloat16/float8 test runs only on CUDA",
    )
    def test_float16_bfloat16_to_complex(self):
        # Test float16 to complex64/complex128
        r_fp16 = np.random.random(size=[10, 10]).astype('float16')
        r_fp16_t = paddle.to_tensor(r_fp16, dtype='float16')

        self.assertEqual(r_fp16_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(r_fp16_t.cast('complex128').dtype, paddle.complex128)

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

        # Test bfloat16 to complex64/complex128
        r_bf16 = np.random.random(size=[10, 10]).astype('float32')
        r_bf16_t = paddle.to_tensor(r_bf16, dtype='bfloat16')

        self.assertEqual(r_bf16_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(r_bf16_t.cast('complex128').dtype, paddle.complex128)

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

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "float8 test runs only on CUDA",
    )
    def test_float8_to_complex(self):
        # Test float8_e4m3fn to complex64/complex128
        r_fp32 = np.random.uniform(1.0, 10.0, size=[10, 10]).astype('float32')
        r_fp32_t = paddle.to_tensor(r_fp32)
        r_fp8_e4m3fn_t = r_fp32_t.astype('float8_e4m3fn')

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

        # Test float8_e5m2 to complex64/complex128
        r_fp8_e5m2_t = r_fp32_t.astype('float8_e5m2')

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
