# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class TestIInfoAndFInfoAPI(unittest.TestCase):
    def test_invalid_input(self):
        for dtype in [
            paddle.float16,
            paddle.float32,
            paddle.float64,
            paddle.bfloat16,
            paddle.complex64,
            paddle.complex128,
            paddle.bool,
            'float16',
            'float32',
            'float64',
            'uint16',
            'complex64',
            'complex128',
            'bool',
        ]:
            if isinstance(dtype, paddle.base.core.DataType):
                dtype = paddle.pir.core.datatype_to_vartype[dtype]
            with self.assertRaises(ValueError):
                _ = paddle.iinfo(dtype)

    def test_iinfo(self):
        for paddle_dtype, np_dtype in [
            (paddle.int64, np.int64),
            (paddle.int32, np.int32),
            (paddle.int16, np.int16),
            (paddle.int8, np.int8),
            (paddle.uint8, np.uint8),
            ('int64', np.int64),
            ('int32', np.int32),
            ('int16', np.int16),
            ('int8', np.int8),
            ('uint8', np.uint8),
        ]:
            if isinstance(paddle_dtype, paddle.base.core.DataType):
                paddle_dtype = paddle.pir.core.datatype_to_vartype[paddle_dtype]
            xinfo = paddle.iinfo(paddle_dtype)
            xninfo = np.iinfo(np_dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.dtype, xninfo.dtype)

    def test_finfo(self):
        for paddle_dtype, np_dtype in [
            (paddle.float32, np.float32),
            (paddle.float64, np.float64),
            ('float32', np.float32),
            ('float64', np.float64),
        ]:
            xinfo = paddle.finfo(paddle_dtype)
            xninfo = np.finfo(np_dtype)
            self.assertEqual(xinfo.dtype, xninfo.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertAlmostEqual(xinfo.max, xninfo.max)
            self.assertAlmostEqual(xinfo.min, xninfo.min)
            self.assertAlmostEqual(xinfo.eps, xninfo.eps)
            self.assertAlmostEqual(xinfo.tiny, xninfo.tiny)
            self.assertAlmostEqual(xinfo.resolution, xninfo.resolution)
            if np.lib.NumpyVersion(np.__version__) >= "1.22.0":
                self.assertAlmostEqual(
                    xinfo.smallest_normal, xninfo.smallest_normal
                )

        for paddle_dtype, np_dtype in [
            (paddle.complex64, np.complex64),
            (paddle.complex128, np.complex128),
            ('complex64', np.complex64),
            ('complex128', np.complex128),
        ]:
            xinfo = paddle.finfo(paddle_dtype)
            xninfo = np.finfo(np_dtype)
            self.assertEqual(xinfo.dtype, xninfo.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertAlmostEqual(xinfo.max, xninfo.max, places=16)
            self.assertAlmostEqual(xinfo.min, xninfo.min, places=16)
            self.assertAlmostEqual(xinfo.eps, xninfo.eps, places=16)
            self.assertAlmostEqual(xinfo.tiny, xninfo.tiny, places=16)
            self.assertAlmostEqual(xinfo.resolution, xninfo.resolution)
            if np.lib.NumpyVersion(np.__version__) >= "1.22.0":
                self.assertAlmostEqual(
                    xinfo.smallest_normal, xninfo.smallest_normal, places=16
                )

        xinfo = paddle.finfo(paddle.float16)
        self.assertEqual(xinfo.dtype, "float16")
        self.assertEqual(xinfo.bits, 16)
        self.assertAlmostEqual(xinfo.max, 65504.0)
        self.assertAlmostEqual(xinfo.min, -65504.0)
        self.assertAlmostEqual(xinfo.eps, 0.0009765625)
        self.assertAlmostEqual(xinfo.tiny, 6.103515625e-05)
        self.assertAlmostEqual(xinfo.resolution, 0.001)
        self.assertAlmostEqual(xinfo.smallest_normal, 6.103515625e-05)

        xinfo = paddle.finfo('float16')
        self.assertEqual(xinfo.dtype, "float16")
        self.assertEqual(xinfo.bits, 16)
        self.assertAlmostEqual(xinfo.max, 65504.0)
        self.assertAlmostEqual(xinfo.min, -65504.0)
        self.assertAlmostEqual(xinfo.eps, 0.0009765625)
        self.assertAlmostEqual(xinfo.tiny, 6.103515625e-05)
        self.assertAlmostEqual(xinfo.resolution, 0.001)
        self.assertAlmostEqual(xinfo.smallest_normal, 6.103515625e-05)

        xinfo = paddle.finfo(paddle.bfloat16)
        self.assertEqual(xinfo.dtype, "bfloat16")
        self.assertEqual(xinfo.bits, 16)
        self.assertAlmostEqual(xinfo.max, 3.3895313892515355e38)
        self.assertAlmostEqual(xinfo.min, -3.3895313892515355e38)
        self.assertAlmostEqual(xinfo.eps, 0.0078125)
        self.assertAlmostEqual(xinfo.tiny, 1.1754943508222875e-38)
        self.assertAlmostEqual(xinfo.resolution, 0.01)
        self.assertAlmostEqual(xinfo.smallest_normal, 1.1754943508222875e-38)


if __name__ == '__main__':
    unittest.main()
