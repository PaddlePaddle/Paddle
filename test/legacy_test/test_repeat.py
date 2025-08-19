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
from utils import dygraph_guard, static_guard

import paddle


class TestRepeatBase(unittest.TestCase):

    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = 3
        self.expected = np.tile(self.x.numpy(), self.repeats)

    def test_dygraph(self):
        with dygraph_guard():
            result = self.x.repeat(self.repeats)
        np.testing.assert_array_equal(result.numpy(), self.expected)

    def test_static(self):
        with (
            static_guard(),
            paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ),
        ):
            x = paddle.to_tensor(self.x.numpy())
            result = x.repeat(self.repeats)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result_np,) = exe.run(fetch_list=[result])
        np.testing.assert_array_equal(result_np, self.expected)


class TestRepeat1DList(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = [2, 1, 3]
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatEmptyTensor(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([])
        self.repeats = 3
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatZeroRepeats(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = 0
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatZeroRepeatsList(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3])
        self.repeats = [0, 1, 0]
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatFloat32(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1.5, 2.5, 3.5], dtype='float32')
        self.repeats = 2
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatFloat64(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1.5, 2.5, 3.5], dtype='float64')
        self.repeats = 2
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatInt32(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3], dtype='int32')
        self.repeats = 2
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatInt64(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3], dtype='int64')
        self.repeats = 2
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatBool(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([True, False, True])
        self.repeats = 2
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatComplex(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype='complex64')
        self.repeats = 2
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatSingleElement(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([42])
        self.repeats = 5
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatLargeRepeats(TestRepeatBase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2])
        self.repeats = 1000
        self.expected = np.tile(self.x.numpy(), self.repeats)


class TestRepeatAPIEdgeCases(unittest.TestCase):
    def test_repeat_negative_repeats(self):
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            x.repeat(-1)

    def test_repeat_no_repeats(self):
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            x.repeat()


class TestRepeatVariableArgs(unittest.TestCase):
    def test_1d_variable_args(self):
        x = paddle.to_tensor([1, 2, 3])
        result = x.repeat(3)
        expected = np.tile(x.numpy(), 3)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_2d_variable_args(self):
        x = paddle.to_tensor([[1, 2], [3, 4]])
        result = x.repeat(2, 3)
        expected = np.tile(x.numpy(), (2, 3))
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_3d_variable_args(self):
        x = paddle.to_tensor([[[1, 2], [3, 4]]])
        result = x.repeat(2, 1, 3)
        expected = np.tile(x.numpy(), (2, 1, 3))
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
