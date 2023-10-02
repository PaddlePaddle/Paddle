# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base.variable_index import _getitem_static


class TestGetitemInDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_combined_index_1(self):
        # int tensor + slice (without decreasing axes)
        np_data = np.random.randn(3, 4, 5, 6)
        np_res = np_data[[0, 1], :, [1, 2]]
        x = paddle.to_tensor(np_data)
        y = x[[0, 1], :, [1, 2]]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_2(self):
        # int tensor + slice (with decreasing axes)
        np_data = np.random.randn(3, 4, 5, 6)
        x = paddle.to_tensor(np_data)

        np_res = np_data[:, 1, [1, 2], 0]
        y = x[:, 1, [1, 2], 0]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_3(self):
        # multiple int tensors, with one int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[[1, 0], :, [1, 4], 1:5:2, 4]

        x = paddle.to_tensor(np_data)
        y = x[[1, 0], :, [1, 4], 1:5:2, 4]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_4(self):
        # multiple not adjacent int tensors, with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[:, [1, 0], 0:4:2, [2, 3], 4]
        x = paddle.to_tensor(np_data)
        y = x[:, [1, 0], 0:4:2, [2, 3], 4]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_5(self):
        # multiple adjacent int tensors, with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[::2, [1, 0], [2, 3], 0:4:2]
        x = paddle.to_tensor(np_data)
        y = x[::2, [1, 0], [2, 3], 0:4:2]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_6(self):
        # multiple adjacent and not adjacent int tensors, with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[::2, [1, 0], [2, 3], 0:4:2, [4, 6]]
        x = paddle.to_tensor(np_data)
        y = x[::2, [1, 0], [2, 3], 0:4:2, [4, 6]]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_7(self):
        # multiple adjacent and not adjacent int tensors (rank > 1d), with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[::2, [[1, 0]], [[2, 3]], 0:4:2, [[4, 6]]]
        x = paddle.to_tensor(np_data)
        y = x[::2, [[1, 0]], [[2, 3]], 0:4:2, [[4, 6]]]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_8(self):
        # multiple adjacent and not adjacent int tensors (rank > 1d), with int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[
            [[1, 0], [0, 1]], [[2, 3], [1, 0]], 0:4:2, [[3, 5], [4, 2]]
        ]
        x = paddle.to_tensor(np_data)
        y = x[[[1, 0], [0, 1]], [[2, 3], [1, 0]], 0:4:2, [[3, 5], [4, 2]]]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_9(self):
        # multiple int tensors, with broadcast.
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[[[1, 0]], [1, 0], 0:4:2, [[3, 5], [4, 2]]]
        x = paddle.to_tensor(np_data)
        y = x[[[1, 0]], [1, 0], 0:4:2, [[3, 5], [4, 2]]]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_10(self):
        # only one bool tensor with basic-index
        np_data = np.random.randn(3, 4, 5, 6)
        np_res = np_data[:, [True, False, True, False], 4]

        x = paddle.to_tensor(np_data)
        y = x[:, [True, False, True, False], 4]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_combined_index_11(self):
        # only one bool tensor with all False
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[:, [False, False, False, False], 4]

        x = paddle.to_tensor(np_data)
        y = x[:, [False, False, False, False], 4]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_index_has_range(self):
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[:, range(3), 4]

        x = paddle.to_tensor(np_data)
        y = x[:, range(3), 4]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_indexing_with_bool_list1(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[[True, False, True], [False, False, False, True]]

        x = paddle.to_tensor(np_data)
        y = x[[True, False, True], [False, False, False, True]]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_indexing_with_bool_list2(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ]

        x = paddle.to_tensor(np_data)
        y = x[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ]

        np.testing.assert_allclose(y.numpy(), np_res)

    def test_indexing_is_multi_dim_list(self):
        # indexing is multi-dim int list, should be treat as one index, like numpy>=1.23
        np_data = np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        np_res = np_data[np.array([[2, 3, 4], [1, 2, 5]])]

        x = paddle.to_tensor(np_data)
        y = x[[[2, 3, 4], [1, 2, 5]]]
        y_index_tensor = x[paddle.to_tensor([[2, 3, 4], [1, 2, 5]])]

        np.testing.assert_allclose(y.numpy(), np_res)
        np.testing.assert_allclose(y.numpy(), y_index_tensor.numpy())


class TestGetitemInStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    def test_combined_index_1(self):
        # int tensor + slice (without decreasing axes)
        np_data = np.random.randn(3, 4, 5, 6)
        np_res = np_data[[0, 1], :, [1, 2]]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(x, ([0, 1], slice(None, None, None), [1, 2]))
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_2(self):
        # int tensor + slice (with decreasing axes)
        np_data = np.random.randn(3, 4, 5, 6)
        np_res = np_data[:, 1, [1, 2], 0]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(x, (slice(None, None, None), 1, [1, 2], 0))
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_3(self):
        # multiple int tensors, with one int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[[1, 0], :, [1, 4], 1:5:2, 4]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x, ([1, 0], slice(None, None, None), [1, 4], slice(1, 5, 2), 4)
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_4(self):
        # multiple not adjacent int tensors, with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[:, [1, 0], 0:4:2, [2, 3], 4]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x, (slice(None, None, None), [1, 0], slice(0, 4, 2), [2, 3], 4)
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_5(self):
        # multiple adjacent int tensors, with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[::2, [1, 0], [2, 3], 0:4:2]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x, (slice(None, None, 2), [1, 0], [2, 3], slice(0, 4, 2))
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_6(self):
        # multiple adjacent and not adjacent int tensors, with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[::2, [1, 0], [2, 3], 0:4:2, [4, 6]]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x,
                (slice(None, None, 2), [1, 0], [2, 3], slice(0, 4, 2), [4, 6]),
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_7(self):
        # multiple adjacent and not adjacent int tensors (rank > 1d), with no int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[::2, [[1, 0]], [[2, 3]], 0:4:2, [[4, 6]]]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x,
                (
                    slice(None, None, 2),
                    [[1, 0]],
                    [[2, 3]],
                    slice(0, 4, 2),
                    [[4, 6]],
                ),
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_8(self):
        # multiple adjacent and not adjacent int tensors (rank > 1d), with int tensor at first axis
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[
            [[1, 0], [0, 1]], [[2, 3], [1, 0]], 0:4:2, [[3, 5], [4, 2]]
        ]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x,
                (
                    [[1, 0], [0, 1]],
                    [[2, 3], [1, 0]],
                    slice(0, 4, 2),
                    [[3, 5], [4, 2]],
                ),
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_9(self):
        # multiple int tensors, with broadcast.
        np_data = np.random.randn(3, 4, 5, 6, 7)
        np_res = np_data[[[1, 0]], [1, 0], 0:4:2, [[3, 5], [4, 2]]]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x, ([[1, 0]], [1, 0], slice(0, 4, 2), [[3, 5], [4, 2]])
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_10(self):
        # only one bool tensor with basic-index
        np_data = np.random.randn(3, 4, 5, 6)
        np_res = np_data[:, [True, False, True, False], 4]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x, (slice(None, None, None), [True, False, True, False], 4)
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_combined_index_11(self):
        # only one bool tensor with all False
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[:, [False, False, False, False], 4]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x, (slice(None, None, None), [False, False, False, False], 4)
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_index_has_range(self):
        # only one bool tensor with all False
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[:, range(3), 4]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(x, (slice(None, None, None), range(3), 4))
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_indexing_with_bool_list1(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[[True, False, True], [False, False, False, True]]

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x, ([True, False, True], [False, False, False, True])
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_indexing_with_bool_list2(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_res = np_data[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(
                x,
                (
                    [True, False, True],
                    [False, False, True, False],
                    [True, False, False, True, False],
                ),
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_res)

    def test_indexing_is_multi_dim_list(self):
        # indexing is multi-dim int list, should be treat as one index, like numpy>=1.23
        np_data = np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        np_res = np_data[np.array([[2, 3, 4], [1, 2, 5]])]
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.to_tensor(np_data)
            y = _getitem_static(x, ([[2, 3, 4], [1, 2, 5]]))
            y_index_tensor = _getitem_static(
                x, paddle.to_tensor([[2, 3, 4], [1, 2, 5]])
            )

            res = self.exe.run(fetch_list=[y.name, y_index_tensor.name])

        np.testing.assert_allclose(res[0], np_res)
        np.testing.assert_allclose(res[1], np_res)


class TestGetitemBasicIndexOutputView(unittest.TestCase):
    def setUp(self):
        # Stride now only supports in dygraph mode
        paddle.disable_static()

    def test_index_is_int(self):
        np_data = np.ones((5, 5, 5), dtype='float32')
        np_tmp = np_data[3, 2]
        np_tmp[2] = 20

        x = paddle.ones((5, 5, 5), dtype='float32')
        x_tmp = x[3, 2]
        x_tmp[2] = 20

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_index_is_0dTensor(self):
        np_data = np.ones((5, 5, 5), dtype='float32')
        np_tmp = np_data[3, 2]
        np_tmp[2] = 20

        x = paddle.ones((5, 5, 5), dtype='float32')
        x_tmp = x[paddle.to_tensor(3), paddle.to_tensor(2)]
        x_tmp[2] = 20

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_index_is_slice(self):
        np_data = np.ones((5, 5, 5), dtype='float32')
        np_tmp = np_data[::2, :, 0:4]
        np_tmp[2] = 20

        x = paddle.ones((5, 5, 5), dtype='float32')
        x_tmp = x[::2, :, 0:4]
        x_tmp[2] = 20

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_index_is_None(self):
        np_data = np.ones((5, 5, 5), dtype='float32')
        np_tmp = np_data[None]
        np_tmp[:, 2] = 20

        x = paddle.ones((5, 5, 5), dtype='float32')
        x_tmp = x[None]
        x_tmp[:, 2] = 20

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_index_is_ellipsis(self):
        np_data = np.ones((5, 5, 5), dtype='float32')
        np_tmp = np_data[...]
        np_tmp[2] = 20

        x = paddle.ones((5, 5, 5), dtype='float32')
        x_tmp = x[...]
        x_tmp[2] = 20

        np.testing.assert_allclose(x.numpy(), np_data)


class TestGetItemErrorCase(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_bool_shape_error1(self):
        x = paddle.randn((4, 3, 2))
        with self.assertRaises(IndexError):
            y = _getitem_static(x, ([True, False]))

    def test_bool_shape_error2(self):
        x = paddle.randn((4, 3, 2))
        with self.assertRaises(IndexError):
            y = _getitem_static(x, (1, paddle.to_tensor([True, False]), [0, 1]))
