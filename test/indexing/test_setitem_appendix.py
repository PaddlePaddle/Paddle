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


class TestSetitemDygraphBasicIndex(unittest.TestCase):
    def accuracy_check(self, numpy_array, paddle_t):
        np.testing.assert_allclose(numpy_array, paddle_t.numpy())

    def test_scalar(self):
        x = np.arange(27).reshape(3, 3, 3)
        y = paddle.to_tensor(x)
        # case1:
        x[0] = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        y[0] = paddle.to_tensor([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        self.accuracy_check(x, y)
        # case2: with broadcasting
        x[-1] = np.array(
            [24, 25, 26]
        )  # [[24, 25, 26], [21, 22, 23], [18, 19, 20]]
        y[-1] = paddle.to_tensor([24, 25, 26])
        self.accuracy_check(x, y)
        # case3:
        x[1, -2] = 100
        y[1, -2] = 100
        self.accuracy_check(x, y)
        # case4:
        x[0, -2, 1] = 1
        y[0, -2, 1] = 1
        self.accuracy_check(x, y)

    def test_slice(self):
        x = np.arange(10)
        y = paddle.to_tensor(x)
        # case 1:
        x[1:7:2] = np.array([10, 30, 50])
        y[1:7:2] = paddle.to_tensor([10, 30, 50])
        self.accuracy_check(x, y)
        # case 2:
        x[-3:9] = np.array([10, 10])
        y[-3:9] = paddle.to_tensor([10, 10])
        self.accuracy_check(x, y)
        x[:11:10] = np.array([100])
        y[:11:10] = paddle.to_tensor([100])
        self.accuracy_check(x, y)
        # case 4:
        x[5:] = np.array([50, 60, 70, 80, 90])
        y[5:] = paddle.to_tensor([50, 60, 70, 80, 90])
        self.accuracy_check(x, y)
        # case 5:
        x[5::2] = np.array([50, 70, 90])
        y[5::2] = paddle.to_tensor([50, 70, 90])
        self.accuracy_check(x, y)
        # case 6:
        x[:] = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        y[:] = paddle.to_tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.accuracy_check(x, y)
        # case 7:
        x[1:2] = np.array([10])
        y[1:2] = paddle.to_tensor([10])
        self.accuracy_check(x, y)
        # case 8:
        x[::-1] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y[::-1] = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.accuracy_check(x, y)

        x = np.arange(36).reshape(3, 6, 2)
        y = paddle.to_tensor(x)
        # case 9:
        x[2, 1:5:3] = np.array([[3], [6]])
        y[2, 1:5:3] = paddle.to_tensor([[3], [6]])
        self.accuracy_check(x, y)
        # case 10:
        x[1, 2, :] = 80
        y[1, 2, :] = 80
        self.accuracy_check(x, y)

    def test_none(self):
        x = np.arange(9).reshape(3, 3)
        y = paddle.to_tensor(x)
        # case 1:
        x[:, None] = -1
        y[:, None] = -1
        self.accuracy_check(x, y)

    def test_ellipsis(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        x[..., 0] = 10
        y[..., 0] = 10
        self.accuracy_check(x, y)

    def test_tuple(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        x[(0, 1)] = 1
        self.accuracy_check(x, y)
        # case 2:
        x[(0,)] = np.array([10, 10, 10, 10, 10])
        y[(0,)] = paddle.to_tensor([10, 10, 10, 10, 10])
        self.accuracy_check(x, y)
        # case 3:
        x[(slice(None, 1), slice(None, 3))] = np.array(
            [[0, 10, 20]]
        )  # x[0:1,0:3]
        y[(slice(None, 1), slice(None, 3))] = paddle.to_tensor([[0, 10, 20]])
        self.accuracy_check(x, y)
        # case 4:
        x[()] = -1
        y[()] = -1
        self.accuracy_check(x, y)


class TestSetitemDygraphAdvancedIndex(unittest.TestCase):
    def accuracy_check(self, numpy_array, paddle_t):
        np.testing.assert_allclose(numpy_array, paddle_t.numpy())

    def test_bool(self):
        x = np.array([0, 1, -1, -2, 2, 0, 5, 0, -3, 2])
        y = paddle.to_tensor(x)
        # case1:
        x[x < 0] = 0
        y[y < 0] = 0
        self.accuracy_check(x, y)
        # case2:
        x[x != 0] = 100
        y[y != 0] = 100
        self.accuracy_check(x, y)
        # case4:
        x[(x > 0) & (x < 2)] = -1
        y[(y > 0) & (y < 2)] = -1
        self.accuracy_check(x, y)

        x = np.arange(9).reshape(3, 3)
        y = paddle.to_tensor(x)
        # case 1:
        x[True] = np.array([[[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]])
        y[True] = paddle.to_tensor([[[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]])
        self.accuracy_check(x, y)
        # case 2:
        x[[True, False, True]] = np.array([[0, 10, 20], [60, 70, 80]])
        y[[True, False, True]] = paddle.to_tensor([[0, 10, 20], [60, 70, 80]])
        self.accuracy_check(x, y)
        # case 3:
        x[[True, False, True], [True, False, True]] = np.array([100])
        y[[True, False, True], [True, False, True]] = paddle.to_tensor([100])
        self.accuracy_check(x, y)

    def test_list(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        x[[1]] = np.array([[50, 60, 70, 80, 90]])
        y[[1]] = paddle.to_tensor([[50, 60, 70, 80, 90]])
        self.accuracy_check(x, y)
        # case 3:
        x[[0, 1], [3, 2]] = np.array([30, 70])
        y[[0, 1], [3, 2]] = paddle.to_tensor([30, 70])
        self.accuracy_check(x, y)
        # case 4:
        x[[0, 1, 0], [3, 2, 4]] = np.array([30, 70, 40])
        y[[0, 1, 0], [3, 2, 4]] = paddle.to_tensor([30, 70, 40])
        self.accuracy_check(x, y)

    def test_tensor(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        x[np.array([1])] = 0
        y[paddle.to_tensor([1])] = 0
        self.accuracy_check(x, y)
        # case 2:
        x[np.ones([], dtype=np.int64)] = 1
        y[paddle.to_tensor(1)] = 1
        self.accuracy_check(x, y)
        # case 3:
        x[np.array([0, 1]), np.array([3, 2])] = np.array([30, 70])
        y[paddle.to_tensor([0, 1]), paddle.to_tensor([3, 2])] = (
            paddle.to_tensor([30, 70])
        )
        self.accuracy_check(x, y)


class TestSetitemDygraphCombinedIndex(unittest.TestCase):
    def accuracy_check(self, numpy_array, paddle_t):
        np.testing.assert_allclose(numpy_array, paddle_t.numpy())

    def test_combined(self):
        x = np.arange(48).reshape(2, 4, 3, 2)
        y = paddle.to_tensor(x)
        # case 1:
        x[:, 3, [0, 2]] = np.array([[[1, 2], [3, 4]]])
        y[:, 3, [0, 2]] = paddle.to_tensor([[[1, 2], [3, 4]]])
        self.accuracy_check(x, y)
        # case 2:
        x[:, 3, [0, 2], [1]] = 100
        y[:, 3, [0, 2], [1]] = 100
        self.accuracy_check(x, y)
        # case 3:
        x[1, [1, 2], :, np.array([0, 1])] = np.array(
            [[0, -2, -4], [-7, -9, -11]]
        )
        y[1, [1, 2], :, paddle.to_tensor([0, 1])] = paddle.to_tensor(
            [[0, -2, -4], [-7, -9, -11]]
        )
        self.accuracy_check(x, y)
        # case 4:
        x[:, [0, 2, 3]][:, 1:3, 1] = np.array([[10, 20], [30, 40]])
        y[:, [0, 2, 3]][:, 1:3, 1] = paddle.to_tensor([[10, 20], [30, 40]])
        self.accuracy_check(x, y)
        # case 5:
        x[:, [0], :, 0] = 100
        y[:, [0], :, 0] = 100
        self.accuracy_check(x, y)
        x[:, [0], :, [0]] = -100
        y[:, [0], :, [0]] = -100
        self.accuracy_check(x, y)
        # case 6:
        x[[True, False], :, -1] = np.array([-4, -5])
        y[[True, False], :, -1] = paddle.to_tensor([-4, -5])
        self.accuracy_check(x, y)


class Test0DTensorIndexing(unittest.TestCase):
    def accuracy_check(self, paddle_t, numpy_array):
        np.testing.assert_allclose(paddle_t.numpy(), numpy_array)

    def test_indexing(self):
        x = paddle.to_tensor(42)
        # case 5:
        x = paddle.to_tensor(99)
        self.accuracy_check(x, 99)


class TestOSizeTensorIndexing(unittest.TestCase):
    def accuracy_check(self, paddle_t, numpy_array):
        np.testing.assert_allclose(paddle_t, numpy_array)

    def test_indexing(self):
        x = paddle.empty([0, 3])
        # case 1.5(set)
        x[:] = 2  # no error, no effect
        self.accuracy_check(x.shape, [0, 3])


class TestSetItemErrorCase(unittest.TestCase):
    def test_scalar(self):
        x = np.arange(27).reshape(3, 3, 3)
        y = paddle.to_tensor(x)
        # case6:
        with self.assertRaises(ValueError):
            x[::-1] = paddle.to_tensor(
                [0, 1, 2, 3]
            )  # ValueError: (InvalidArgument)


if __name__ == '__main__':
    unittest.main()
