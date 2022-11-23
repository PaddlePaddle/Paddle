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

import paddle
import unittest
import numpy as np


class TestIInfoAndFInfoAPI(unittest.TestCase):

    def test_invalid_input(self):
        for dtype in [
                paddle.float16, paddle.float32, paddle.float64, paddle.bfloat16,
                paddle.complex64, paddle.complex128, paddle.bool
        ]:
            with self.assertRaises(ValueError):
                _ = paddle.iinfo(dtype)

    def test_iinfo(self):
        for paddle_dtype, np_dtype in [(paddle.int64, np.int64),
                                       (paddle.int32, np.int32),
                                       (paddle.int16, np.int16),
                                       (paddle.int8, np.int8),
                                       (paddle.uint8, np.uint8)]:
            xinfo = paddle.iinfo(paddle_dtype)
            xninfo = np.iinfo(np_dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.dtype, xninfo.dtype)


if __name__ == '__main__':
    unittest.main()
