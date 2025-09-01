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
from utils import dygraph_guard, static_guard

import paddle
from paddle import enable_static


class TestSumOp_Compatibility(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4]
        self.axis = 0
        self.input_dtype = 'float32'
        self.test_dtypes = [
            "int32",
            "float32",
        ]

    def test_dygraph(self):
        with dygraph_guard():
            x_paddle = paddle.ones(shape=self.shape, dtype=self.input_dtype)
            for dtype_input in self.test_dtypes:
                numpy_result = np.sum(
                    x_paddle.numpy(),
                    axis=self.axis,
                    dtype=np.dtype(dtype_input),
                    keepdims=False,
                )

                # paddle test case
                paddle_result0 = paddle.sum(x_paddle, self.axis, dtype_input)
                np.testing.assert_allclose(paddle_result0, numpy_result)

                paddle_result1 = paddle.sum(
                    x_paddle, self.axis, dtype_input, False
                )
                np.testing.assert_allclose(paddle_result1, numpy_result)

                paddle_result2 = paddle.sum(
                    x=x_paddle, axis=self.axis, dtype=dtype_input, keepdim=False
                )
                np.testing.assert_allclose(paddle_result2, numpy_result)

                # torch test case
                paddle_result3 = paddle.sum(
                    input=x_paddle, dim=self.axis, keepdim=False
                )
                self.assertEqual(paddle_result3.dtype, paddle.float32)

                paddle_result4 = paddle.sum(
                    input=x_paddle,
                    dim=self.axis,
                    keepdim=False,
                    dtype=dtype_input,
                )
                np.testing.assert_allclose(paddle_result4, numpy_result)

                paddle_result5 = paddle.sum(
                    x_paddle, self.axis, keepdim=False, dtype=dtype_input
                )
                np.testing.assert_allclose(paddle_result5, numpy_result)

                paddle_result6 = paddle.sum(
                    x_paddle, self.axis, False, dtype=dtype_input
                )
                np.testing.assert_allclose(paddle_result6, numpy_result)

                paddle_result7 = paddle.sum(
                    x_paddle, self.axis, False, dtype_input
                )
                np.testing.assert_allclose(paddle_result7, numpy_result)

                paddle_result8 = paddle.sum(
                    x_paddle, self.axis, dtype_input, False
                )
                np.testing.assert_allclose(paddle_result8, numpy_result)

                paddle_result9 = paddle.sum(x_paddle, self.axis, False)
                self.assertEqual(paddle_result9.dtype, paddle.float32)

                paddle_result10 = paddle.sum(x_paddle, self.axis, dtype_input)
                np.testing.assert_allclose(paddle_result10, numpy_result)

    def test_static(self):
        self.test_dtypes = [
            paddle.int32,
            paddle.int64,
            paddle.float64,
            paddle.bool,
        ]
        with static_guard():
            for dtype_input in self.test_dtypes:
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x_paddle = paddle.static.data(
                        name='x', shape=self.shape, dtype=self.input_dtype
                    )

                    # paddle test case
                    paddle_result0 = paddle.sum(
                        x_paddle, axis=self.axis, dtype=dtype_input
                    )
                    self.assertEqual(paddle_result0.dtype, dtype_input)

                    paddle_result1 = paddle.sum(
                        x_paddle,
                        axis=self.axis,
                        dtype=dtype_input,
                        keepdim=False,
                    )
                    self.assertEqual(paddle_result1.dtype, dtype_input)

                    paddle_result2 = paddle.sum(
                        x=x_paddle,
                        axis=self.axis,
                        dtype=dtype_input,
                        keepdim=False,
                    )
                    self.assertEqual(paddle_result2.dtype, dtype_input)

                    # torch test case
                    paddle_result3 = paddle.sum(
                        input=x_paddle, dim=self.axis, keepdim=False
                    )
                    self.assertEqual(paddle_result3.dtype, paddle.float32)

                    paddle_result4 = paddle.sum(
                        input=x_paddle,
                        dim=self.axis,
                        keepdim=False,
                        dtype=dtype_input,
                    )
                    self.assertEqual(paddle_result4.dtype, dtype_input)

                    paddle_result5 = paddle.sum(
                        x_paddle, self.axis, keepdim=False, dtype=dtype_input
                    )
                    self.assertEqual(paddle_result5.dtype, dtype_input)

                    paddle_result6 = paddle.sum(
                        x_paddle, self.axis, False, dtype=dtype_input
                    )
                    self.assertEqual(paddle_result6.dtype, dtype_input)

                    paddle_result7 = paddle.sum(
                        x_paddle, self.axis, False, dtype_input
                    )
                    self.assertEqual(paddle_result7.dtype, dtype_input)

                    paddle_result8 = paddle.sum(
                        x_paddle, self.axis, dtype_input, False
                    )
                    self.assertEqual(paddle_result8.dtype, dtype_input)

                    paddle_result9 = paddle.sum(x_paddle, self.axis, False)
                    self.assertEqual(paddle_result9.dtype, paddle.float32)

                    paddle_result10 = paddle.sum(
                        x_paddle, self.axis, dtype_input
                    )
                    self.assertEqual(paddle_result10.dtype, dtype_input)


if __name__ == "__main__":
    enable_static()
    unittest.main()
