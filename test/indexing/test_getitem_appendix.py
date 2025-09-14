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


class TestGetitemDygraphBasicIndex(unittest.TestCase):
    def accuracy_check(self, numpy_array, paddle_t):
        np.testing.assert_allclose(numpy_array, paddle_t.numpy())

    def test_scalar(self):
        x = np.arange(27).reshape(3, 3, 3)
        y = paddle.to_tensor(x)
        # case1:
        # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.accuracy_check(x[0], y[0])
        # case2:
        # [[18, 19, 20], [21, 22, 23], [24, 25, 26]]
        self.accuracy_check(x[-1], y[-1])
        # case3:
        # [12, 13, 14]
        self.accuracy_check(x[1, -2], y[1, -2])
        # case4:
        # 4
        self.accuracy_check(x[0, -2, 1], y[0, -2, 1])

    def test_slice(self):
        x = np.arange(10)
        y = paddle.to_tensor(x)
        # case 1:
        # [1, 3, 5]
        self.accuracy_check(x[1:7:2], y[1:7:2])
        # case 2:
        # [7, 8]
        self.accuracy_check(x[-3:9], y[-3:9])
        # Automatically adjust to effective range: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.accuracy_check(x[1:11], y[1:11])
        # [0]
        self.accuracy_check(x[:11:10], y[:11:10])
        self.accuracy_check(x[11:13], y[11:13])
        self.accuracy_check(x[10:21:10], y[10:21:10])
        self.accuracy_check(x[0:0], y[0:0])
        # case 3:
        # torch does not support negative step
        self.accuracy_check(x[3:-3:-1], y[3:-3:-1])  # []
        self.accuracy_check(x[-3:3:1], y[-3:3:1])  # []
        self.accuracy_check(x[-3:3:-1], y[-3:3:-1])  # [7, 6, 5, 4]
        # case 4:
        # [5, 6, 7, 8, 9]
        self.accuracy_check(x[0:0], y[0:0])
        # [5, 7, 9]
        self.accuracy_check(x[5::2], y[5::2])
        # case 5:
        # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.accuracy_check(x[:-1], y[:-1])
        # case 6:
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.accuracy_check(x[:], y[:])
        # case 7:
        # [1]
        self.accuracy_check(x[1:2], y[1:2])

        x = np.arange(36).reshape(3, 6, 2)
        y = paddle.to_tensor(x)
        # case 9:
        # [[13],[16]]
        self.accuracy_check(x[2, 1:5:3], y[2, 1:5:3])
        # case 10:
        # [8]
        self.accuracy_check(x[1, 2, :], y[1, 2, :])

    def test_none(self):
        x = np.arange(27).reshape(3, 3, 3)
        y = paddle.to_tensor(x)
        # case 1:
        # x.shape = [3,1,3,3]
        self.accuracy_check(x[:, None, :, :], y[:, None, :, :])
        # case 2:
        self.accuracy_check(x[:, None], y[:, None])

    def test_ellipsis(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        self.accuracy_check(x[...], y[...])
        # case 2:
        self.accuracy_check(x[..., 0], y[..., 0])

    def test_tuple(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        # 1
        self.accuracy_check(x[(0, 1)], y[(0, 1)])
        # case 2:
        # [0, 1, 2, 3, 4]
        self.accuracy_check(x[(0,)], y[(0,)])
        # case 3:
        self.accuracy_check(
            x[(slice(None, 1), slice(None, 3))],
            y[(slice(None, 1), slice(None, 3))],
        )
        # case 4:
        # [[0, 1, 2, 3, 4],[5, 6, 7, 8, 9]]
        self.accuracy_check(x[()], y[()])


class TestGetitemDygraphAdvancedIndex(unittest.TestCase):
    def accuracy_check(self, numpy_array, paddle_t):
        np.testing.assert_allclose(numpy_array, paddle_t.numpy())

    def test_bool(self):
        x = np.array([0, 1, -1, -2, 2, 0, 5, 0, -3, 2])
        y = paddle.to_tensor(x)
        # case1:
        #  [-1., -3.]
        self.accuracy_check(x[x < 0], y[y < 0])
        # case2:
        # [[ 1.],[-1.],[nan],[ 2.],[nan],[-3.],[ 2.]]
        self.accuracy_check(x[x != 0], y[y != 0])
        # case3:
        # [1.]
        self.accuracy_check(x[(x > 0) & (x < 2)], y[(y > 0) & (y < 2)])

        x = np.arange(9).reshape(3, 3)
        y = paddle.to_tensor(x)
        # case 1:
        # [[[0, 1, 2],[3, 4, 5],[6, 7, 8]]]
        self.accuracy_check(x[True], y[True])
        # case 2:
        # [[0, 1, 2],[6, 7, 8]]
        self.accuracy_check(x[[True, False, True]], y[[True, False, True]])
        # case 3:
        # [0, 8]
        self.accuracy_check(
            x[[True, False, True], [True, False, True]],
            y[[True, False, True], [True, False, True]],
        )

    def test_list(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        # [[5, 6, 7, 8, 9],[5, 6, 7, 8, 9]]
        self.accuracy_check(x[[1, 1]], y[[1, 1]])
        # case 2:
        # [[5, 6, 7, 8, 9],[5, 6, 7, 8, 9],[0, 1, 2, 3, 4]]
        self.accuracy_check(x[[1, 1, 0]], y[[1, 1, 0]])
        # case 3:
        # 7
        self.accuracy_check(x[[0, 1], [3, 2]], y[[0, 1], [3, 2]])
        # case 4:
        # [3, 7, 4]
        self.accuracy_check(x[[0, 1, 0], [3, 2, 4]], y[[0, 1, 0], [3, 2, 4]])

    def test_tensor(self):
        x = np.arange(10).reshape(2, 5)
        y = paddle.to_tensor(x)
        # case 1:
        # [[5, 6, 7, 8, 9],[5, 6, 7, 8, 9]]
        self.accuracy_check(x[np.array([1, 1])], y[paddle.to_tensor([1, 1])])
        # case 2:
        # [[5, 6, 7, 8, 9],[5, 6, 7, 8, 9],[0, 1, 2, 3, 4]]
        self.accuracy_check(
            x[np.array([1, 1, 0])], y[paddle.to_tensor([1, 1, 0])]
        )
        # case 3:
        # [3, 7]
        self.accuracy_check(
            x[np.array([0, 1]), np.array([3, 2])],
            y[paddle.to_tensor([0, 1]), paddle.to_tensor([3, 2])],
        )
        # case 4:
        # [3, 7, 4]
        self.accuracy_check(
            x[np.array([0, 1, 0]), np.array([3, 2, 4])],
            y[paddle.to_tensor([0, 1, 0]), paddle.to_tensor([3, 2, 4])],
        )
        # case 5:
        # [5, 6, 7, 8, 9]
        self.accuracy_check(
            x[np.ones([], dtype=np.int64)], y[paddle.to_tensor(1)]
        )


class TestGetitemDygraphCombinedIndex(unittest.TestCase):
    def accuracy_check(self, numpy_array, paddle_t):
        np.testing.assert_allclose(numpy_array, paddle_t.numpy())

    def test_combined(self):
        x = np.arange(48).reshape(2, 4, 3, 2)
        y = paddle.to_tensor(x)
        # case 1:
        # [[[18, 19],[22, 23]], [[42, 43],[46, 47]]]
        self.accuracy_check(x[:, 3, [0, 2]], y[:, 3, [0, 2]])
        # case 2:
        # [[19, 23],[43, 47]]
        self.accuracy_check(x[:, 3, [0, 2], [1]], y[:, 3, [0, 2], [1]])
        # case 3:
        # [[30, 32, 34],[37, 39, 41]]
        self.accuracy_check(
            x[1, [1, 2], :, np.array([0, 1])],
            y[1, [1, 2], :, paddle.to_tensor([0, 1])],
        )
        # case 4:
        # [[[14, 15],[20, 21]],[[38, 39],[44, 45]]]
        self.accuracy_check(
            x[:, [0, 2, 3]][:, 1:3, 1], y[:, [0, 2, 3]][:, 1:3, 1]
        )
        # case 5:
        x_array = [[[0, 2, 4]], [[24, 26, 28]]]  # x.shape=[2,1,3]
        self.accuracy_check(x_array, y[:, [0], :, 0])
        # x.shape=[1,2,3] [[[0 , 2 , 4 ],[24, 26, 28]]]
        self.accuracy_check(x[:, [0], :, [0]], y[:, [0], :, [0]])
        # case 6:
        # [[[4 , 5 ],[10, 11],[16, 17],[22, 23]]]
        self.accuracy_check(x[[True, False], :, -1], y[[True, False], :, -1])
        # case 7:
        # [[0, 3, 4, 5], [24, 26, 28, 29]]
        index_np = np.array([[True, False], [False, True], [True, True]])
        index_paddle = paddle.to_tensor(index_np)
        self.accuracy_check(x[:, 0, index_np], y[:, 0, index_paddle])
        # case 8:
        # [[[[0, 1]], [[2, 3]], [[24, 25]], [[26, 27]]]]
        index_np = np.array([[0], [1]])
        index_paddle = paddle.to_tensor(index_np)
        self.accuracy_check(x[:, 0, index_np], y[:, 0, index_paddle])


class Test0DTensorIndexing(unittest.TestCase):
    def accuracy_check(self, paddle_t, numpy_array):
        np.testing.assert_allclose(paddle_t.numpy(), numpy_array)

    def test_indexing(self):
        x = paddle.to_tensor(42)
        # y = torch.tensor(42)
        # case 1:
        # 42
        self.accuracy_check(x[...], 42)
        # case 3:
        self.accuracy_check(x[None, ...], [42])  # [42]
        # case 4:
        self.accuracy_check(x[paddle.to_tensor(True)], [42])
        self.accuracy_check(x[True], [42])


class TestOSizeTensorIndexing(unittest.TestCase):
    def accuracy_check(self, paddle_t, numpy_array):
        np.testing.assert_allclose(paddle_t, numpy_array)

    def test_indexing(self):
        x = paddle.empty([0, 3])
        # y = torch.empty([0, 3])
        # case 1:
        self.accuracy_check(x[:].shape, [0, 3])
        # case 2:
        self.accuracy_check(x[0:2, 1:].shape, [0, 2])
        # case 3:
        self.accuracy_check(x[...].shape, [0, 3])
        # case 4:
        self.accuracy_check(x[[]].shape, [0, 3])
        self.accuracy_check(x[[], []].shape, [0])
        # case 5:
        empty_index_p = paddle.to_tensor([], dtype='int64')
        # empty_index_t = torch.tensor([], dtype=torch.int64)
        self.accuracy_check(x[empty_index_p].shape, [0, 3])
        # case 6:
        self.accuracy_check(x[:, None].shape, [0, 1, 3])
        # case 7:
        mask_p = x > 1
        self.accuracy_check(mask_p.shape, [0, 3])
        self.accuracy_check(x[True].shape, [1, 0, 3])


class TestGetItemErrorCase(unittest.TestCase):
    def test_scalar(self):
        # case5:
        x = np.arange(27).reshape(3, 3, 3)
        paddle_t = paddle.to_tensor(x)
        with self.assertRaises(IndexError):
            res = x[3, 0]  # IndexError: (OutOfRange)
        # case6:
        with self.assertRaises(IndexError):
            res = x[0, 0, 0, 0]  # IndexError: (OutOfRange)

    def test_tuple(self):
        x = np.arange(10).reshape(2, 5)
        x = paddle.to_tensor(x)
        # case 5:
        with self.assertRaises(IndexError):
            res = x[(2, 4)]  # IndexError: (OutOfRange)

    def test_list(self):
        x = np.arange(10).reshape(2, 5)
        x = paddle.to_tensor(x)
        # case 5:
        with self.assertRaises(ValueError):
            res = x[
                [0, 1], [3, 2], [1, 1]
            ]  # ValueError: (InvalidArgument) Too many indices

    def test_bool(self):
        x = np.arange(9).reshape(3, 3)
        x = paddle.to_tensor(x)
        # case 4:
        with self.assertRaises(IndexError):
            res = x[
                [True, False, True], [True, False, True], [False, True, True]
            ]  # IndexError: (OutOfRange)
        # case 5:
        with self.assertRaises(IndexError):
            res = x[[True, False]]  # IndexError: (OutOfRange)
        # case 6:
        with self.assertRaises(IndexError):
            res = x[[True, False, True], [False]]  # IndexError: (OutOfRange)

    def test_tensor(self):
        x = np.arange(10).reshape(2, 5)
        x = paddle.to_tensor(x)
        # case 5:
        with self.assertRaises(ValueError):
            res = x[
                paddle.to_tensor([0, 1]),
                paddle.to_tensor([3, 2]),
                paddle.to_tensor([1, 1]),
            ]  # ValueError: (InvalidArgument) Too many indices

    def test_0D(self):
        x = paddle.to_tensor(42)
        # case 2:
        with self.assertRaises(ValueError):
            res = x[:]  # ValueError: (InvalidArgument) Too many indices
        # case 6:
        with self.assertRaises(IndexError):
            res = x[0]  # IndexError: (OutOfRange)


if __name__ == '__main__':
    unittest.main()
