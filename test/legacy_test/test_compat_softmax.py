#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestCompatSoftmax(unittest.TestCase):
    def _compare_with_origin(self, input_tensor, axis):
        softmax_compat = paddle.compat.nn.Softmax(dim=axis)
        softmax_origin = paddle.nn.Softmax(axis=axis)

        expected_res = softmax_origin(input_tensor).numpy()
        np.testing.assert_allclose(
            softmax_compat(input_tensor).numpy(),
            expected_res,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_compare_with_origin(self):
        input_shape = (3, 4)
        input_tensor = paddle.randn(input_shape, dtype=paddle.float32)
        self._compare_with_origin(input_tensor, axis=0)
        self._compare_with_origin(input_tensor, axis=1)
        self._compare_with_origin(input_tensor, axis=-1)

        input_shape = (2, 3, 4)
        input_tensor = paddle.randn(input_shape, dtype=paddle.float64)
        self._compare_with_origin(input_tensor, axis=0)
        self._compare_with_origin(input_tensor, axis=1)
        self._compare_with_origin(input_tensor, axis=2)
        self._compare_with_origin(input_tensor, axis=-1)

        input_shape = (2, 3, 4, 5)
        input_tensor = paddle.randn(input_shape, dtype=paddle.float32)
        self._compare_with_origin(input_tensor, axis=1)
        self._compare_with_origin(input_tensor, axis=-2)

        input_tensor = paddle.randn((2, 3), dtype=paddle.float32)
        softmax_compat = paddle.compat.nn.Softmax()
        softmax_origin = paddle.nn.Softmax()
        expected_res = softmax_origin(input_tensor).numpy()
        np.testing.assert_allclose(
            softmax_compat(input_tensor).numpy(),
            expected_res,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_error_handling(self):
        x = paddle.randn([3, 9, 5])

        msg_gt_1 = "paddle.compat.nn.Softmax() received unexpected keyword argument 'axis'. \nDid you mean to use paddle.nn.Softmax() instead?"

        with self.assertRaises(TypeError) as cm:
            softmax = paddle.compat.nn.Softmax(axis=1)
        self.assertEqual(str(cm.exception), msg_gt_1)


if __name__ == "__main__":
    unittest.main()
