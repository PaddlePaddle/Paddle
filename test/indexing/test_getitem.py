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
from paddle.fluid.variable_index import _getitem_static

# class TestGetitemInDygraph(unittest.TestCase):
#     def setUp(self):
#         paddle.disable_static()

#     def test_combined_index_1(self):
#         np_data = np.zeros((3, 4, 5, 6), dtype='float32')
#         x = paddle.to_tensor(np_data)

#         np_res = np_data[[0, 1], :, [1, 2]]
#         y = x[[0, 1], :, [1, 2]]

#         np.testing.assert_allclose(y.numpy(), np_res)

#     def test_combined_index_2(self):
#         np_data = np.ones((3, 4, 5, 6), dtype='float32')
#         x = paddle.to_tensor(np_data)

#         np_res = np_data[:, 1, [1, 2], 0]
#         y = x[:, 1, [1, 2], 0]

#         np.testing.assert_allclose(y.numpy(), np_res)

#     def test_combined_index_3(self):
#         np_data = np.ones((3, 4, 5, 6), dtype='int32')
#         x = paddle.to_tensor(np_data)

#         np_res = np_data[:, [True, False, True, False], [1, 4]]
#         y = x[:, [True, False, True, False], [1, 4]]

#         np.testing.assert_allclose(y.numpy(), np_res)


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

    # def test_combined_index_7(self):
    #     # int tensor + bool tensor + slice (without decreasing axes)
    #     np_data = np.random.randn(3, 4, 5, 6)
    #     np_res = np_data[:, [True, False, True, False], [1, 4]]
    #     with paddle.static.program_guard(
    #         paddle.static.Program(), paddle.static.Program()
    #     ):
    #         x = paddle.to_tensor(np_data)
    #         y = _getitem_static(
    #             x,
    #             (slice(None, None, None), [True, False, True, False], [1, 4])
    #         )
    #         res = self.exe.run(fetch_list=[y.name])

    #     np.testing.assert_allclose(res[0], np_res)

    # def test_combined_index_4(self):
    #     # int tensor (with ranks > 1) + bool tensor + slice (with decreasing axes)
    #     np_data = np.arange(3*4*5*6).reshape((3, 4, 5, 6))
    #     np_res = np_data[[0, 0], [True, False, True, False], [[0, 2], [1, 4]], 4]
    #     with paddle.static.program_guard(
    #         paddle.static.Program(), paddle.static.Program()
    #     ):
    #         x = paddle.to_tensor(np_data)
    #         y = _getitem_static(
    #             x,
    #             ([0, 0], [True, False, True, False], [[0, 2], [1, 4]], 4)
    #         )
    #         res = self.exe.run(fetch_list=[y.name])

    #     np.testing.assert_allclose(res[0], np_res)

    # def test_combined_index_5(self):
    #     # int tensor + slice + Ellipsis
    #     np_data = np.arange(3*4*5*6).reshape((3, 4, 5, 6))
    #     np_res = np_data[..., [1, 4, 3], ::2]
    #     with paddle.static.program_guard(
    #         paddle.static.Program(), paddle.static.Program()
    #     ):
    #         x = paddle.to_tensor(np_data)
    #         y = _getitem_static(
    #             x,
    #             (..., [1, 4, 3], slice(None, None, 2)),
    #         )
    #         res = self.exe.run(fetch_list=[y.name])

    #     np.testing.assert_allclose(res[0], np_res)
