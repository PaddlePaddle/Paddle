#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle


class TestPaddleAddNewFeatures(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([3, 5], dtype='float32')
        self.y_np = np.array([2, 3], dtype='float32')
        self.scalar = 2.0
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_paddle_add_with_alpha(self):
        """test paddle.add alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.add(x, y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_tensor_add_with_alpha(self):
        """test paddle.Tensor.add alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = x.add(y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_tensor_add_inplace_with_alpha(self):
        """test Tensor.add_ alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.add_(y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_consistency_between_apis(self):
        """test different APIs consistency for add with alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)

        out1 = paddle.add(x, y, alpha=2)
        out2 = x.add(y, alpha=2)
        x.add_(y, alpha=2)

        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out1.numpy(), expected)
        np.testing.assert_array_equal(out2.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_static_graph_add_with_alpha(self):
        """test static graph add with alpha and parameter aliases"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 2], dtype='float32')
            out1 = paddle.add(x, y, alpha=2)
            out2 = paddle.add(input=x, other=y, alpha=2)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x_np.reshape(1, 2),
                    'y': self.y_np.reshape(1, 2),
                },
                fetch_list=[out1, out2],
            )

            expected = self.x_np + self.y_np * 2
            for result in res:
                np.testing.assert_array_equal(result.flatten(), expected)
        paddle.disable_static()

    def test_param_alias_input_other(self):
        """test parameter alias input/other in dynamic graph"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)

        out1 = paddle.add(input=x, other=y, alpha=2)
        out2 = x.add(other=y, alpha=2)
        x_clone = x.clone()
        x_clone.add_(other=y, alpha=2)

        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out1.numpy(), expected)
        np.testing.assert_array_equal(out2.numpy(), expected)
        np.testing.assert_array_equal(x_clone.numpy(), expected)

    def test_scalar_addition(self):
        """test scalar addition"""
        x = paddle.to_tensor(self.x_np)

        out1 = paddle.add(x, self.scalar)
        expected1 = self.x_np + self.scalar
        np.testing.assert_array_equal(out1.numpy(), expected1)

        out2 = x.add(self.scalar)
        np.testing.assert_array_equal(out2.numpy(), expected1)

        out3 = paddle.add(x, self.scalar, alpha=2)
        expected3 = self.x_np + self.scalar * 2
        np.testing.assert_array_equal(out3.numpy(), expected3)

    def test_scalar_addition_inplace(self):
        """test inplace scalar addition"""
        x = paddle.to_tensor(self.x_np)
        x_clone = x.clone()

        x_clone.add_(self.scalar)
        expected = self.x_np + self.scalar
        np.testing.assert_array_equal(x_clone.numpy(), expected)

        x_clone2 = x.clone()
        x_clone2.add_(self.scalar, alpha=2)
        expected2 = self.x_np + self.scalar * 2
        np.testing.assert_array_equal(x_clone2.numpy(), expected2)

    def test_different_dtype_scalar(self):
        """test different dtype scalar addition"""
        x = paddle.to_tensor(self.x_np)

        out1 = x.add(2)
        expected1 = self.x_np + 2
        np.testing.assert_array_equal(out1.numpy(), expected1)

        out2 = x.add(2.5)
        expected2 = self.x_np + 2.5
        np.testing.assert_array_equal(out2.numpy(), expected2)

    def test_scalar_addition_static_graph(self):
        """test static graph scalar addition"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            out1 = paddle.add(x, self.scalar)
            out2 = paddle.add(x, self.scalar, alpha=2)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.x_np.reshape(1, 2)},
                fetch_list=[out1, out2],
            )

            expected1 = self.x_np + self.scalar
            expected2 = self.x_np + self.scalar * 2
            np.testing.assert_array_equal(res[0].flatten(), expected1)
            np.testing.assert_array_equal(res[1].flatten(), expected2)
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
