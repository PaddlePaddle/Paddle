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
from paddle.nn import ConstantPad1d, ConstantPad2d, ConstantPad3d


class TestConstantPad1d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_ncl_format(self):
        # NCL: (1, 2, 3)
        data_np = np.array([[[1, 2, 3], [4, 5, 6]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 2]  # pad_left, pad_right
        val = 0.5
        my_pad = ConstantPad1d(padding=pad, value=val, data_format="NCL")
        result = my_pad(data)
        assert (
            my_pad.__repr__()
            == "ConstantPad1D(padding=[1, 2], mode=constant, value=0.5, data_format=NCL)"
        )
        expected_np = np.array(
            [[[0.5, 1.0, 2.0, 3.0, 0.5, 0.5], [0.5, 4.0, 5.0, 6.0, 0.5, 0.5]]],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 2, 6])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_nlc_format(self):
        # NLC: (1, 3, 2)
        data_np = np.array([[[1, 4], [2, 5], [3, 6]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 2]  # pad_left, pad_right (applies to L dim)
        val = -1.0
        my_pad = ConstantPad1d(padding=pad, value=val, data_format="NLC")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [-1.0, -1.0],
                    [1.0, 4.0],
                    [2.0, 5.0],
                    [3.0, 6.0],
                    [-1.0, -1.0],
                    [-1.0, -1.0],
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 6, 2])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_int_padding(self):
        # NCL: (1, 2, 3)
        data_np = np.array([[[1, 2, 3], [4, 5, 6]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = 2  # (2, 2)
        val = 0.0
        my_pad = ConstantPad1d(padding=pad, value=val, data_format="NCL")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 2, 7])
        np.testing.assert_allclose(result.numpy(), expected_np)


class TestConstantPad2d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_nchw_format(self):
        # NCHW: (1, 1, 2, 3)
        data_np = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 0, 1, 2]  # (pad_left, pad_right, pad_top, pad_bottom)
        val = 0.5
        my_pad = ConstantPad2d(padding=pad, value=val, data_format="NCHW")
        result = my_pad(data)
        assert (
            my_pad.__repr__()
            == "ConstantPad2D(padding=[1, 0, 1, 2], mode=constant, value=0.5, data_format=NCHW)"
        )

        expected_np = np.array(
            [
                [
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 1.0, 2.0, 3.0],
                        [0.5, 4.0, 5.0, 6.0],
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ]
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 1, 5, 4])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_nhwc_format(self):
        # NHWC: (1, 2, 3, 1)
        data_np = np.array(
            [[[[1], [2], [3]], [[4], [5], [6]]]], dtype="float32"
        )
        data = paddle.to_tensor(data_np)

        pad = [1, 0, 1, 2]  # (pad_left, pad_right, pad_top, pad_bottom)
        val = 9.9
        my_pad = ConstantPad2d(padding=pad, value=val, data_format="NHWC")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [[9.9], [9.9], [9.9], [9.9]],
                    [[9.9], [1.0], [2.0], [3.0]],
                    [[9.9], [4.0], [5.0], [6.0]],
                    [[9.9], [9.9], [9.9], [9.9]],
                    [[9.9], [9.9], [9.9], [9.9]],
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 5, 4, 1])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_int_padding(self):
        # NCHW: (1, 1, 2, 2)
        data_np = np.array([[[[1, 2], [3, 4]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = 1  # (1, 1, 1, 1)
        val = 0.0
        my_pad = ConstantPad2d(padding=pad, value=val, data_format="NCHW")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 2.0, 0.0],
                        [0.0, 3.0, 4.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 1, 4, 4])
        np.testing.assert_allclose(result.numpy(), expected_np)


class TestConstantPad3d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_ncdhw_format(self):
        # NCDHW: (1, 1, 1, 2, 3)
        data_np = np.array([[[[[1, 2, 3], [4, 5, 6]]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad = [1, 0, 1, 2, 0, 0]
        val = -1.0
        my_pad = ConstantPad3d(padding=pad, value=val, data_format="NCDHW")
        assert (
            my_pad.__repr__()
            == "ConstantPad3D(padding=[1, 0, 1, 2, 0, 0], mode=constant, value=-1.0, data_format=NCDHW)"
        )
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [
                        [
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, 1.0, 2.0, 3.0],
                            [-1.0, 4.0, 5.0, 6.0],
                            [-1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0],
                        ]
                    ]
                ]
            ],
            dtype="float32",
        )

        # Shape: (1, 1, 1+0+0, 2+1+2, 3+1+0) = (1, 1, 1, 5, 4)
        self.assertEqual(result.shape, [1, 1, 1, 5, 4])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_ndhwc_format(self):
        # NDHWC: (1, 1, 2, 3, 1)
        data_np = np.array(
            [[[[[1], [2], [3]], [[4], [5], [6]]]]], dtype="float32"
        )
        data = paddle.to_tensor(data_np)

        # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad = [1, 0, 1, 2, 0, 0]
        val = 0.1
        my_pad = ConstantPad3d(padding=pad, value=val, data_format="NDHWC")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [
                        [[0.1], [0.1], [0.1], [0.1]],
                        [[0.1], [1.0], [2.0], [3.0]],
                        [[0.1], [4.0], [5.0], [6.0]],
                        [[0.1], [0.1], [0.1], [0.1]],
                        [[0.1], [0.1], [0.1], [0.1]],
                    ]
                ]
            ],
            dtype="float32",
        )

        # Shape: (1, 1+0+0, 2+1+2, 3+1+0, 1) = (1, 1, 5, 4, 1)
        self.assertEqual(result.shape, [1, 1, 5, 4, 1])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_int_padding(self):
        # NCDHW: (1, 1, 1, 2, 2)
        data_np = np.array([[[[[1, 2], [3, 4]]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = 1  # (1, 1, 1, 1, 1, 1)
        val = 0.0
        my_pad = ConstantPad3d(padding=pad, value=val, data_format="NCDHW")
        result = my_pad(data)

        # Shape: (1, 1, 1+1+1, 2+1+1, 2+1+1) = (1, 1, 3, 4, 4)
        self.assertEqual(result.shape, [1, 1, 3, 4, 4])

        expected_d0 = np.zeros((4, 4), dtype="float32")
        expected_d2 = np.zeros((4, 4), dtype="float32")
        expected_d1 = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype="float32",
        )

        expected_np = np.array(
            [[[expected_d0, expected_d1, expected_d2]]], dtype="float32"
        )

        np.testing.assert_allclose(result.numpy(), expected_np)


if __name__ == "__main__":
    unittest.main()
