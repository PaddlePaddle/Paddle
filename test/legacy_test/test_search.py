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

import paddle


class TestSearchAPIs(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.con = None
        self.con_2D = None

    def setUp(self):
        self.con = paddle.to_tensor([0.4, 0.3, 0.6, 0.7], dtype="float32")
        self.con_2D = paddle.rand([4, 4], dtype='float32')

    def test_where_with_float16_scalar(self):
        # TODO(hanchoa): Do not support float16 with cpu.
        pass

    def test_where_with_bfloat16_scalar(self):
        # TODO(hanchoa): Do not support bfloat16 with cpu.
        pass

    def test_where_with_float32_scalar(self):
        x = paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype="float32")
        y = paddle.to_tensor([0.1, 0.1, 0.1, 0.1], dtype="float32")

        res = paddle.where(self.con > 0.5, x, y)
        self.assertEqual(res.dtype, paddle.float32)

        res = paddle.where(self.con > 0.5, 0.5, y)
        self.assertEqual(res.dtype, paddle.float32)

        res = paddle.where(self.con > 0.5, x, 0.6)
        self.assertEqual(res.dtype, paddle.float32)

        res = paddle.where(self.con > 0.5, 0.5, 0.6)
        self.assertEqual(res.dtype, paddle.float32)

    def test_where_with_float64_scalar(self):
        x = paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype="float64")
        y = paddle.to_tensor([0.1, 0.1, 0.1, 0.1], dtype="float64")

        res = paddle.where(self.con > 0.5, x, y)
        self.assertEqual(res.dtype, paddle.float64)

        res = paddle.where(self.con > 0.5, 0.5, y)
        self.assertEqual(res.dtype, paddle.float64)

        res = paddle.where(self.con > 0.5, x, 0.6)
        self.assertEqual(res.dtype, paddle.float64)

        res = paddle.where(self.con > 0.5, 0.5, 0.6)
        self.assertEqual(res.dtype, paddle.float32)

    def test_where_with_complex64_scalar(self):
        x = paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype="complex64")
        y = paddle.to_tensor([0.1, 0.1, 0.1, 0.1], dtype="complex64")

        res = paddle.where(self.con > 0.5, x, y)
        self.assertEqual(res.dtype, paddle.complex64)

        res = paddle.where(self.con > 0.5, 0.5, y)
        self.assertEqual(res.dtype, paddle.complex64)

        res = paddle.where(self.con > 0.5, x, 0.6)
        self.assertEqual(res.dtype, paddle.complex64)

        res = paddle.where(self.con > 0.5, 0.5, 0.6)
        self.assertEqual(res.dtype, paddle.float32)

    @unittest.skipIf(
        paddle.is_compiled_with_xpu(),
        "XPU cast kernel does not support complex128 yet.",
    )
    def test_where_with_complex128_scalar(self):
        x = paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype="complex128")
        y = paddle.to_tensor([0.1, 0.1, 0.1, 0.1], dtype="complex128")

        res = paddle.where(self.con > 0.5, x, y)
        self.assertEqual(res.dtype, paddle.complex128)

        res = paddle.where(self.con > 0.5, 0.5, y)
        self.assertEqual(res.dtype, paddle.complex128)

        res = paddle.where(self.con > 0.5, x, 0.6)
        self.assertEqual(res.dtype, paddle.complex128)

        res = paddle.where(self.con > 0.5, 0.5, 0.6)
        self.assertEqual(res.dtype, paddle.float32)

    def test_where_with_int_scalar(self):
        x = paddle.to_tensor([2, 2, 2, 2], dtype="int32")
        y = paddle.to_tensor([3, 3, 3, 3], dtype="int32")

        res = paddle.where(self.con > 0.5, x, y)
        self.assertEqual(res.dtype, paddle.int32)

        # TODO(hanchao): Do not support int type promotion yet.
        # res = paddle.where(self.con > 0.5, 3, y)
        # self.assertEqual(res.dtype, paddle.int32)

        # res = paddle.where(self.con > 0.5, x, 4)
        # self.assertEqual(res.dtype, paddle.int32)
        #
        # res = paddle.where(self.con > 0.5, 3, 4)
        # self.assertEqual(res.dtype, paddle.int32)

    def test_where_with_float32_scalar_2D(self):
        x = paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype="float32")
        y = paddle.to_tensor([0.1, 0.1, 0.1, 0.1], dtype="float32")

        res = paddle.where(self.con_2D > 0.5, x, y)
        self.assertEqual(res.dtype, paddle.float32)

        res = paddle.where(self.con_2D > 0.5, 0.5, y)
        self.assertEqual(res.dtype, paddle.float32)

        res = paddle.where(self.con_2D > 0.5, x, 0.6)
        self.assertEqual(res.dtype, paddle.float32)

        res = paddle.where(self.con_2D > 0.5, 0.5, 0.6)
        self.assertEqual(res.dtype, paddle.float32)


if __name__ == '__main__':
    unittest.main()
