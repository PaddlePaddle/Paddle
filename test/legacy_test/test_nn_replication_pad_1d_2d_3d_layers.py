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
from paddle.nn import ReplicationPad1d, ReplicationPad2d, ReplicationPad3d


class TestReplicationPad1d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_ncl_format(self):
        # NCL: (1, 2, 3)
        data_np = np.array([[[1, 2, 3], [4, 5, 6]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 2]  # pad_left, pad_right
        my_pad = ReplicationPad1d(padding=pad, data_format="NCL")
        assert (
            my_pad.__repr__()
            == "ReplicationPad1D(padding=[1, 2], mode=replicate, value=0.0, data_format=NCL)"
        )
        result = my_pad(data)

        expected_np = np.array(
            [[[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], [4.0, 4.0, 5.0, 6.0, 6.0, 6.0]]],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 2, 6])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_nlc_format(self):
        # NLC: (1, 3, 2)
        data_np = np.array([[[1, 4], [2, 5], [3, 6]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 2]  # pad_left, pad_right (applies to L dim)
        my_pad = ReplicationPad1d(padding=pad, data_format="NLC")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [1.0, 4.0],  # Replicated left
                    [1.0, 4.0],
                    [2.0, 5.0],
                    [3.0, 6.0],
                    [3.0, 6.0],  # Replicated right
                    [3.0, 6.0],  # Replicated right
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
        my_pad = ReplicationPad1d(padding=pad, data_format="NCL")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 2, 7])
        np.testing.assert_allclose(result.numpy(), expected_np)


class TestReplicationPad2d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_nchw_format(self):
        # NCHW: (1, 1, 2, 3)
        data_np = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 0, 1, 2]  # (pad_left, pad_right, pad_top, pad_bottom)
        my_pad = ReplicationPad2d(padding=pad, data_format="NCHW")
        assert (
            my_pad.__repr__()
            == "ReplicationPad2D(padding=[1, 0, 1, 2], mode=replicate, value=0.0, data_format=NCHW)"
        )
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [
                        [1.0, 1.0, 2.0, 3.0],  # Replicated top
                        [1.0, 1.0, 2.0, 3.0],
                        [4.0, 4.0, 5.0, 6.0],
                        [4.0, 4.0, 5.0, 6.0],  # Replicated bottom
                        [4.0, 4.0, 5.0, 6.0],  # Replicated bottom
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
        my_pad = ReplicationPad2d(padding=pad, data_format="NHWC")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [[1.0], [1.0], [2.0], [3.0]],  # Replicated top
                    [[1.0], [1.0], [2.0], [3.0]],
                    [[4.0], [4.0], [5.0], [6.0]],
                    [[4.0], [4.0], [5.0], [6.0]],  # Replicated bottom
                    [[4.0], [4.0], [5.0], [6.0]],  # Replicated bottom
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
        my_pad = ReplicationPad2d(padding=pad, data_format="NCHW")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [
                        [1.0, 1.0, 2.0, 2.0],  # Replicated top
                        [1.0, 1.0, 2.0, 2.0],
                        [3.0, 3.0, 4.0, 4.0],
                        [3.0, 3.0, 4.0, 4.0],  # Replicated bottom
                    ]
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 1, 4, 4])
        np.testing.assert_allclose(result.numpy(), expected_np)


class TestReplicationPad3d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_ncdhw_format(self):
        # NCDHW: (1, 1, 1, 2, 3)
        data_np = np.array([[[[[1, 2, 3], [4, 5, 6]]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad = [1, 0, 1, 2, 0, 0]  # Pad W, H. D is unchanged.
        my_pad = ReplicationPad3d(padding=pad, data_format="NCDHW")
        assert (
            my_pad.__repr__()
            == "ReplicationPad3D(padding=[1, 0, 1, 2, 0, 0], mode=replicate, value=0.0, data_format=NCDHW)"
        )
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [
                        [
                            [1.0, 1.0, 2.0, 3.0],  # Replicated top
                            [1.0, 1.0, 2.0, 3.0],
                            [4.0, 4.0, 5.0, 6.0],
                            [4.0, 4.0, 5.0, 6.0],  # Replicated bottom
                            [4.0, 4.0, 5.0, 6.0],  # Replicated bottom
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
        pad = [1, 0, 1, 2, 0, 0]  # Pad W, H. D is unchanged.
        my_pad = ReplicationPad3d(padding=pad, data_format="NDHWC")
        result = my_pad(data)

        expected_np = np.array(
            [
                [
                    [
                        [[1.0], [1.0], [2.0], [3.0]],  # Replicated top
                        [[1.0], [1.0], [2.0], [3.0]],
                        [[4.0], [4.0], [5.0], [6.0]],
                        [[4.0], [4.0], [5.0], [6.0]],  # Replicated bottom
                        [[4.0], [4.0], [5.0], [6.0]],  # Replicated bottom
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
        my_pad = ReplicationPad3d(padding=pad, data_format="NCDHW")
        result = my_pad(data)

        # Shape: (1, 1, 1+1+1, 2+1+1, 2+1+1) = (1, 1, 3, 4, 4)
        self.assertEqual(result.shape, [1, 1, 3, 4, 4])

        # A HxW plane padded
        expected_plane = np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ],
            dtype="float32",
        )

        # Replicate the single D plane
        expected_np = np.array(
            [[[expected_plane, expected_plane, expected_plane]]],
            dtype="float32",
        )

        np.testing.assert_allclose(result.numpy(), expected_np)


if __name__ == "__main__":
    unittest.main()
