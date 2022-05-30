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

from __future__ import print_function, division

import unittest
import numpy as np

import paddle
from paddle.fluid.framework import _test_eager_guard


class TestComplexCastOp(unittest.TestCase):
    def test_complex_to_real(self):
        r = np.random.random(size=[10, 10]) * 10
        i = np.random.random(size=[10, 10])

        c_t = paddle.to_tensor(r + i * 1J, dtype='complex64')

        self.assertEqual(c_t.cast('int64').dtype, paddle.int64)
        self.assertEqual(c_t.cast('int32').dtype, paddle.int32)
        self.assertEqual(c_t.cast('float32').dtype, paddle.float32)
        self.assertEqual(c_t.cast('float64').dtype, paddle.float64)
        self.assertEqual(c_t.cast('bool').dtype, paddle.bool)

        self.assertTrue(
            np.allclose(c_t.cast('int64').numpy(), r.astype('int64')))
        self.assertTrue(
            np.allclose(c_t.cast('int32').numpy(), r.astype('int32')))
        self.assertTrue(
            np.allclose(c_t.cast('float32').numpy(), r.astype('float32')))
        self.assertTrue(
            np.allclose(c_t.cast('float64').numpy(), r.astype('float64')))
        self.assertTrue(np.allclose(c_t.cast('bool').numpy(), r.astype('bool')))

    def test_real_to_complex(self):
        r = np.random.random(size=[10, 10]) * 10
        r_t = paddle.to_tensor(r)

        self.assertEqual(r_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(r_t.cast('complex128').dtype, paddle.complex128)

        self.assertTrue(np.allclose(r_t.cast('complex64').real().numpy(), r))
        self.assertTrue(np.allclose(r_t.cast('complex128').real().numpy(), r))

    def test_complex64_complex128(self):
        r = np.random.random(size=[10, 10])
        i = np.random.random(size=[10, 10])

        c = r + i * 1J
        c_64 = paddle.to_tensor(c, dtype='complex64')
        c_128 = paddle.to_tensor(c, dtype='complex128')

        self.assertTrue(c_64.cast('complex128').dtype, paddle.complex128)
        self.assertTrue(c_128.cast('complex128').dtype, paddle.complex64)
        self.assertTrue(
            np.allclose(c_64.cast('complex128').numpy(), c_128.numpy()))
        self.assertTrue(
            np.allclose(c_128.cast('complex128').numpy(), c_64.numpy()))

    def test_eager(self):
        with _test_eager_guard():
            self.test_complex64_complex128()
            self.test_real_to_complex()
            self.test_complex_to_real()


if __name__ == '__main__':
    unittest.main()
