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
from paddle.base.variable_index import _setitem_static


class TestSetitemInDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_combined_index_1(self):
        np_data = np.zeros((3, 4, 5, 6), dtype='float32')
        x = paddle.to_tensor(np_data)

        np_data[[0, 1], :, [1, 2]] = 10.0
        x[[0, 1], :, [1, 2]] = 10.0

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_index_2(self):
        np_data = np.ones((3, 4, 5, 6), dtype='float32')
        x = paddle.to_tensor(np_data)

        np_data[:, 1, [1, 2], 0] = 10.0
        x[:, 1, [1, 2], 0] = 10.0

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_combined_index_3(self):
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        x = paddle.to_tensor(np_data)

        np_data[:, [True, False, True, False], [1, 4]] = 10
        x[:, [True, False, True, False], [1, 4]] = 10

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_index_has_range(self):
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        x = paddle.to_tensor(np_data)

        np_data[:, range(3), [1, 2, 4]] = 10
        x[:, range(3), [1, 2, 4]] = 10

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_src_value_with_different_dtype_1(self):
        # basic-indexing, with set_value op
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_value = np.zeros((6,), dtype='float32')
        x = paddle.to_tensor(np_data)
        v = paddle.to_tensor(np_value)

        np_data[0, 2, 3] = np_value
        x[0, 2, 3] = v

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_src_value_with_different_dtype_2(self):
        # combined-indexing, with index_put op
        np_data = np.ones((3, 4, 5, 6), dtype='float32')
        np_value = np.zeros((6,), dtype='int64')

        x = paddle.to_tensor(np_data)
        v = paddle.to_tensor(np_value)

        np_data[:, [1, 0], 3] = np_value
        x[:, [1, 0], 3] = v

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_indexing_with_bool_list1(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_data[[True, False, True], [False, False, False, True]] = 7

        x = paddle.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        x[[True, False, True], [False, False, False, True]] = 7

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_indexing_with_bool_list2(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_data[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ] = 8

        x = paddle.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        x[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ] = 8

        np.testing.assert_allclose(x.numpy(), np_data)

    def test_indexing_is_multi_dim_list(self):
        # indexing is multi-dim int list, should be treat as one index, like numpy>=1.23
        np_data = np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        np_data[np.array([[2, 3, 4], [1, 2, 5]])] = 100

        x = paddle.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        x[[[2, 3, 4], [1, 2, 5]]] = 100

        np.testing.assert_allclose(x.numpy(), np_data)


class TestSetitemInStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    def test_combined_index_1(self):
        # int tensor + slice (without decreasing axes)
        np_data = np.zeros((3, 4, 5, 6), dtype='float32')
        np_data[[0, 1], :, [1, 2]] = 10.0
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.zeros((3, 4, 5, 6), dtype='float32')
            y = _setitem_static(
                x, ([0, 1], slice(None, None, None), [1, 2]), 10.0
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_combined_index_2(self):
        # int tensor + slice (with decreasing axes)
        np_data = np.ones((3, 4, 5, 6), dtype='float32')
        np_data[:, 1, [1, 2], 0] = 10.0
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='float32')
            y = _setitem_static(
                x, (slice(None, None, None), 1, [1, 2], 0), 10.0
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_combined_index_3(self):
        # int tensor + bool tensor + slice (without decreasing axes)
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[:, [True, False, True, False], [1, 4]] = 10
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                (slice(None, None, None), [True, False, True, False], [1, 4]),
                10,
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_combined_index_4(self):
        # int tensor (with ranks > 1) + bool tensor + slice (with decreasing axes)
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[[0, 0], [True, False, True, False], [[0, 2], [1, 4]], 4] = 16
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                ([0, 0], [True, False, True, False], [[0, 2], [1, 4]], 4),
                16,
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_combined_index_5(self):
        # int tensor + slice + Ellipsis
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[..., [1, 4, 3], ::2] = 5
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                (..., [1, 4, 3], slice(None, None, 2)),
                5,
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_index_has_range(self):
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_data[:, range(3), [1, 2, 4]] = 10
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            y = _setitem_static(
                x,
                (slice(None, None), range(3), [1, 2, 4]),
                10,
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_src_value_with_different_dtype_1(self):
        # basic-indexing, with set_value op
        np_data = np.ones((3, 4, 5, 6), dtype='int32')
        np_value = np.zeros((6,), dtype='float32')
        np_data[0, 2, 3] = np_value

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='int32')
            v = paddle.zeros((6,), dtype='float32')
            y = _setitem_static(
                x,
                (0, 2, 3),
                v,
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_src_value_with_different_dtype_2(self):
        # combined-indexing, with index_put op
        np_data = np.ones((3, 4, 5, 6), dtype='float32')
        np_value = np.zeros((6,), dtype='int64')
        np_data[:, [1, 0], 3] = np_value

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.ones((3, 4, 5, 6), dtype='float32')
            v = paddle.zeros((6,), dtype='int64')
            y = _setitem_static(
                x,
                (slice(None, None), [1, 0], 3),
                v,
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_indexing_with_bool_list1(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_data[[True, False, True], [False, False, False, True]] = 7

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
            y = _setitem_static(
                x, ([True, False, True], [False, False, False, True]), 7
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_indexing_with_bool_list2(self):
        # test bool-list indexing when axes num less than x.rank
        np_data = np.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
        np_data[
            [True, False, True],
            [False, False, True, False],
            [True, False, False, True, False],
        ] = 8
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((3, 4, 5, 6))
            y = _setitem_static(
                x,
                (
                    [True, False, True],
                    [False, False, True, False],
                    [True, False, False, True, False],
                ),
                8,
            )
            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)

    def test_indexing_is_multi_dim_list(self):
        # indexing is multi-dim int list, should be treat as one index, like numpy>=1.23
        np_data = np.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
        np_data[np.array([[2, 3, 4], [1, 2, 5]])] = 10
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.arange(3 * 4 * 5 * 6).reshape((6, 5, 4, 3))
            y = _setitem_static(x, [[[2, 3, 4], [1, 2, 5]]], 10)

            res = self.exe.run(fetch_list=[y.name])

        np.testing.assert_allclose(res[0], np_data)
