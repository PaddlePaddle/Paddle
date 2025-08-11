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


class TestRepeatBase(unittest.TestCase):

    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = 3
        self.expected = np.repeat(self.x.numpy(), self.repeats)

    def test_repeat_dygraph(self):
        paddle.disable_static()
        result = paddle.repeat(self.x, self.repeats)
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_repeat_static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.to_tensor(
                self.x.numpy() if hasattr(self.x, 'numpy') else self.x
            )
            result = paddle.repeat(x, self.repeats)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result_np,) = exe.run(fetch_list=[result])
            np.testing.assert_array_equal(result_np, self.expected)


class TestRepeat1DList(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = [2, 1, 3]
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeat2DSingleValue(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([[1, 2], [3, 4]])
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats, axis=0)


class TestRepeat2DList(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([[1, 2], [3, 4]])
        self.repeats = [2, 3]
        self.expected = np.repeat(self.x.numpy(), self.repeats, axis=1)


class TestRepeat3DSingleValue(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats, axis=0)


class TestRepeat3DList(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.repeats = [2, 1, 3]
        self.expected = np.repeat(self.x.numpy(), self.repeats, axis=2)


class TestRepeatVariableArgs(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([[1, 2], [3, 4]])
        self.repeats = (2, 3)
        self.expected = np.repeat(self.x.numpy(), self.repeats, axis=1)


class TestRepeatTensorRepeats(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([[1, 2], [3, 4]])
        self.repeats = paddle.to_tensor([2, 3])
        self.expected = np.repeat(self.x.numpy(), self.repeats.numpy(), axis=1)


class TestRepeatEmptyTensor(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([])
        self.repeats = 3
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatZeroRepeats(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = 0
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatZeroRepeatsList(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = [0, 1, 0]
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatFloat32(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1.5, 2.5, 3.5], dtype='float32')
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatFloat64(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1.5, 2.5, 3.5], dtype='float64')
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatInt32(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3], dtype='int32')
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatInt64(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3], dtype='int64')
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatBool(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([True, False, True])
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatComplex(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype='complex64')
        self.repeats = 2
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatSingleElement(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([42])
        self.repeats = 5
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatLargeRepeats(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2])
        self.repeats = 1000
        self.expected = np.repeat(self.x.numpy(), self.repeats)


class TestRepeatAPIEdgeCases(unittest.TestCase):
    def test_repeat_negative_repeats(self):
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            paddle.repeat(x, -1)

    def test_repeat_mismatched_length(self):
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            paddle.repeat(x, [1, 2])

    def test_repeat_no_repeats(self):
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.repeat(x)


class TestRepeatAPIGradient(unittest.TestCase):

    def test_repeat_gradient(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], stop_gradient=False)
        result = paddle.repeat(x, 2)
        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.repeat(np.ones_like(x.numpy()), 2, axis=0).reshape(
            x.shape
        )
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_repeat_gradient_complex(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], stop_gradient=False)
        result = paddle.repeat(x, [2, 3])
        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.array([[2.0, 3.0], [2.0, 3.0]])
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)


class TestRepeatAPIPerformance(unittest.TestCase):

    def test_repeat_large_tensor(self):
        x = paddle.randn([100, 100])
        result = paddle.repeat(x, 2)

        expected_shape = [200, 100]
        self.assertEqual(result.shape, expected_shape)

        self.assertGreater(paddle.sum(paddle.abs(result)).item(), 0)


if __name__ == "__main__":
    unittest.main()
