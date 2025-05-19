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


class TestError(unittest.TestCase):
    def test_exception(self):
        exception_regular_expression = (
            'Expected reduction axis(.)* to have non-zero size.'
        )

        self.assertRaisesRegex(
            IndexError,
            exception_regular_expression,
            lambda: paddle.argmin(
                paddle.zeros([1, 2, 3, 0], dtype='float32').argmin(3)
            ),
        )

        self.assertRaisesRegex(
            IndexError,
            exception_regular_expression,
            lambda: paddle.argmax(
                paddle.zeros([1, 2, 3, 0], dtype='float32').argmax(3)
            ),
        )
        self.assertRaisesRegex(
            IndexError,
            exception_regular_expression,
            lambda: paddle.argmax(
                paddle.zeros([1, 2, 3, 0], dtype='float32').argmax(-1)
            ),
        )
        self.assertRaisesRegex(
            IndexError,
            exception_regular_expression,
            lambda: paddle.argmax(
                paddle.zeros([1, 2, 0, 3, 4], dtype='int32').argmax(2)
            ),
        )
        self.assertRaisesRegex(
            IndexError,
            exception_regular_expression,
            lambda: paddle.argmax(
                paddle.zeros([1, 2, 0, 3, 4], dtype='int64').argmax(-3)
            ),
        )


def create_test_class(op_type, dtype, shape, axis):
    class TempCls(unittest.TestCase):
        def test_zero_size(self):
            numpy_tensor_1 = np.random.rand(*shape).astype(dtype)
            paddle_x = paddle.to_tensor(numpy_tensor_1)
            paddle_x.stop_gradient = False

            paddle_api = eval(f"paddle.{op_type}")
            paddle_out = paddle_api(paddle_x, axis=axis)
            numpy_api = eval(f"np.{op_type}")
            numpy_out = numpy_api(numpy_tensor_1, axis=axis)

            np.testing.assert_allclose(
                paddle_out.numpy(),
                numpy_out,
                1e-2,
                1e-2,
            )
            np.testing.assert_allclose(
                paddle_out.shape,
                numpy_out.shape,
                1e-2,
                1e-2,
            )

    cls_name = f"{op_type}{dtype}ZeroSizeTest"
    TempCls.__name__ = cls_name
    globals()[cls_name] = TempCls


create_test_class("argmax", "float32", [3, 4, 0], 0)
create_test_class("argmax", "float64", [3, 4, 0, 3, 4], -2)
create_test_class("argmax", "int32", [3, 4, 0], 0)
create_test_class("argmax", "int64", [3, 4, 0, 3, 4], -1)
create_test_class("argmin", "float32", [0, 3, 4], -2)
create_test_class("argmin", "float64", [3, 0, 4], 2)
create_test_class("argmin", "int32", [3, 4, 0, 3, 4], 1)
create_test_class("argmin", "int64", [3, 4, 0, 3, 4], 0)

if __name__ == '__main__':
    unittest.main()
