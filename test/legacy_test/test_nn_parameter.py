#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import textwrap
import unittest

import numpy as np

import paddle

#  Parameters
#     data (Tensor) – parameter tensor.
#     requires_grad (bool, optional) – if the parameter requires gradient. Default: True


class TestPaddleParameter(unittest.TestCase):
    def setUp(self):
        self.data_np = np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype='float32'
        )

    def test_case_1(self):
        paddle_code = textwrap.dedent(
            """
            import paddle
            x = paddle.to_tensor([[1., 2., 3.], [2., 3., 4.]])
            result = paddle.nn.Parameter(x)
            """
        )
        x = paddle.to_tensor(self.data_np)
        result = paddle.nn.Parameter(x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertEqual(result.trainable, True)  # Default requires grad

    def test_case_2(self):
        paddle_code = textwrap.dedent(
            """
            import paddle
            x = paddle.to_tensor([[1., 2., 3.], [2., 3., 4.]])
            result = paddle.nn.Parameter(x, requires_grad=False)
            """
        )
        x = paddle.to_tensor(self.data_np)
        result = paddle.nn.Parameter(x, requires_grad=False)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertEqual(result.trainable, False)

    def test_alias_case_1(self):
        paddle_code = textwrap.dedent(
            """
            import paddle
            x = paddle.to_tensor([[1., 2., 3.], [2., 3., 4.]])
            result = paddle.nn.parameter.Parameter(x)
            """
        )
        x = paddle.to_tensor(self.data_np)
        result = paddle.nn.parameter.Parameter(x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertEqual(result.trainable, True)

    def test_case_3(self):
        paddle_code = textwrap.dedent(
            """
            import paddle
            x = paddle.to_tensor([[1., 2., 3.], [2., 3., 4.]])
            result = paddle.nn.Parameter(x, False)
            """
        )
        x = paddle.to_tensor(self.data_np)
        result = paddle.nn.Parameter(x, False)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertEqual(result.trainable, False)

    def test_case_4(self):
        paddle_code = textwrap.dedent(
            """
            import paddle
            x = paddle.to_tensor([[1., 2., 3.], [2., 3., 4.]])
            result = paddle.nn.Parameter(data=x, requires_grad=False)
            """
        )
        x = paddle.to_tensor(self.data_np)
        result = paddle.nn.Parameter(data=x, requires_grad=False)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertEqual(result.trainable, False)

    def test_case_5(self):
        paddle_code = textwrap.dedent(
            """
            import paddle
            x = paddle.to_tensor([[1., 2., 3.], [2., 3., 4.]])
            result = paddle.nn.Parameter(requires_grad=False, data=x)
            """
        )
        x = paddle.to_tensor(self.data_np)
        result = paddle.nn.Parameter(requires_grad=False, data=x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertEqual(result.trainable, False)

    def test_case_6(self):
        paddle_code = textwrap.dedent(
            """
            import paddle
            result = paddle.nn.Parameter()
            """
        )
        result = paddle.nn.Parameter()
        self.assertEqual(result.shape, [0])  # Empty parameter
        self.assertEqual(result.trainable, True)

    def test_inheritance(self):
        """Test that Parameter is subclass of both Parameter and Tensor"""
        param = paddle.nn.Parameter()
        self.assertTrue(isinstance(param, paddle.Tensor))
        self.assertTrue(isinstance(param, paddle.nn.Parameter))
        self.assertEqual(type(param), paddle.nn.Parameter)

        class MyTensor(paddle.Tensor):
            _is_param = True

        a = MyTensor(self.data_np)
        self.assertTrue(isinstance(a, paddle.nn.Parameter))

        b = paddle.to_tensor(self.data_np)
        self.assertFalse(isinstance(b, paddle.nn.Parameter))

    def test_repr(self):
        """Test Parameter.__repr__() output"""
        x = paddle.to_tensor(self.data_np)
        x.stop_gradient = False
        param = paddle.nn.Parameter(x)

        expected_repr = f"Parameter containing:\n{x!s}"

        self.assertEqual(repr(param), expected_repr)
        self.assertEqual(str(param), expected_repr)


if __name__ == "__main__":
    unittest.main()
