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
from paddle.nn import CircularPad1d, CircularPad2d, CircularPad3d


@unittest.skipIf(
    paddle.is_compiled_with_xpu(),
    "XPU does not support circular padding mode.",
)
class TestCircularPad1d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_ncl_format(self):
        # NCL: (1, 2, 3)
        data_np = np.array([[[1, 2, 3], [4, 5, 6]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 2]  # pad_left, pad_right
        my_pad = CircularPad1d(padding=pad, data_format="NCL")
        assert (
            my_pad.__repr__()
            == "CircularPad1D(padding=[1, 2], mode=circular, value=0.0, data_format=NCL)"
        )
        result = my_pad(data)

        # Expected:
        # [1, 2, 3] -> pad_left(1) [3], pad_right(2) [1, 2] -> [3, 1, 2, 3, 1, 2]
        # [4, 5, 6] -> pad_left(1) [6], pad_right(2) [4, 5] -> [6, 4, 5, 6, 4, 5]
        expected_np = np.array(
            [[[3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [6.0, 4.0, 5.0, 6.0, 4.0, 5.0]]],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 2, 6])
        np.testing.assert_allclose(result.numpy(), expected_np)

    def test_nlc_format(self):
        # NLC: (1, 3, 2)
        data_np = np.array([[[1, 4], [2, 5], [3, 6]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 2]  # pad_left, pad_right (applies to L dim)
        my_pad = CircularPad1d(padding=pad, data_format="NLC")
        result = my_pad(data)

        # Expected: L-dim slices are [1,4], [2,5], [3,6]
        # pad_left(1) [3,6]
        # pad_right(2) [1,4], [2,5]
        # Result slices: [3,6], [1,4], [2,5], [3,6], [1,4], [2,5]
        expected_np = np.array(
            [
                [
                    [3.0, 6.0],  # Circ-padded left
                    [1.0, 4.0],
                    [2.0, 5.0],
                    [3.0, 6.0],
                    [1.0, 4.0],  # Circ-padded right
                    [2.0, 5.0],  # Circ-padded right
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
        my_pad = CircularPad1d(padding=pad, data_format="NCL")
        result = my_pad(data)

        # Expected:
        # [1, 2, 3] -> pad_left(2) [2, 3], pad_right(2) [1, 2] -> [2, 3, 1, 2, 3, 1, 2]
        # [4, 5, 6] -> pad_left(2) [5, 6], pad_right(2) [4, 5] -> [5, 6, 4, 5, 6, 4, 5]
        expected_np = np.array(
            [
                [
                    [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
                    [5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0],
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 2, 7])
        np.testing.assert_allclose(result.numpy(), expected_np)


@unittest.skipIf(
    paddle.is_compiled_with_xpu(),
    "XPU does not support circular padding mode.",
)
class TestCircularPad2d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_nchw_format(self):
        # NCHW: (1, 1, 2, 3)
        data_np = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        pad = [1, 0, 1, 2]  # (pad_left, pad_right, pad_top, pad_bottom)
        my_pad = CircularPad2d(padding=pad, data_format="NCHW")
        assert (
            my_pad.__repr__()
            == "CircularPad2D(padding=[1, 0, 1, 2], mode=circular, value=0.0, data_format=NCHW)"
        )
        result = my_pad(data)

        # Original:
        # [[1, 2, 3],  (Row A)
        #  [4, 5, 6]]  (Row B)
        #
        # Step 1: Pad L/R (pad_left=1, pad_right=0)
        # [1, 2, 3] -> [3, 1, 2, 3] (Row A')
        # [4, 5, 6] -> [6, 4, 5, 6] (Row B')
        #
        # Step 2: Pad T/B (pad_top=1, pad_bottom=2) on [Row A', Row B']
        # Pad top 1 (from bottom): [Row B']
        # Pad bottom 2 (from top): [Row A', Row B']
        #
        # Result: [Row B', Row A', Row B', Row A', Row B']
        expected_np = np.array(
            [
                [
                    [
                        [6.0, 4.0, 5.0, 6.0],  # Row B' (Circ-padded top)
                        [3.0, 1.0, 2.0, 3.0],  # Row A'
                        [6.0, 4.0, 5.0, 6.0],  # Row B'
                        [3.0, 1.0, 2.0, 3.0],  # Row A' (Circ-padded bottom)
                        [6.0, 4.0, 5.0, 6.0],  # Row B' (Circ-padded bottom)
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
        my_pad = CircularPad2d(padding=pad, data_format="NHWC")
        result = my_pad(data)

        # Original H-dim slices:
        # [[1], [2], [3]]  (Row A)
        # [[4], [5], [6]]  (Row B)
        #
        # Step 1: Pad L/R (pad_left=1, pad_right=0) on W-dim
        # [[1], [2], [3]] -> [[3], [1], [2], [3]] (Row A')
        # [[4], [5], [6]] -> [[6], [4], [5], [6]] (Row B')
        #
        # Step 2: Pad T/B (pad_top=1, pad_bottom=2) on H-dim [Row A', Row B']
        # Result: [Row B', Row A', Row B', Row A', Row B']
        expected_np = np.array(
            [
                [
                    [[6.0], [4.0], [5.0], [6.0]],  # Row B' (Circ-padded top)
                    [[3.0], [1.0], [2.0], [3.0]],  # Row A'
                    [[6.0], [4.0], [5.0], [6.0]],  # Row B'
                    [[3.0], [1.0], [2.0], [3.0]],  # Row A' (Circ-padded bottom)
                    [[6.0], [4.0], [5.0], [6.0]],  # Row B' (Circ-padded bottom)
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
        my_pad = CircularPad2d(padding=pad, data_format="NCHW")
        result = my_pad(data)

        # Original:
        # [[1, 2],  (Row A)
        #  [3, 4]]  (Row B)
        #
        # Step 1: Pad L/R (pad_left=1, pad_right=1)
        # [1, 2] -> [2, 1, 2, 1] (Row A')
        # [3, 4] -> [4, 3, 4, 3] (Row B')
        #
        # Step 2: Pad T/B (pad_top=1, pad_bottom=1) on [Row A', Row B']
        # Pad top 1 (from bottom): [Row B']
        # Pad bottom 1 (from top): [Row A']
        #
        # Result: [Row B', Row A', Row B', Row A']
        expected_np = np.array(
            [
                [
                    [
                        [4.0, 3.0, 4.0, 3.0],  # Row B' (Circ-padded top)
                        [2.0, 1.0, 2.0, 1.0],  # Row A'
                        [4.0, 3.0, 4.0, 3.0],  # Row B'
                        [2.0, 1.0, 2.0, 1.0],  # Row A' (Circ-padded bottom)
                    ]
                ]
            ],
            dtype="float32",
        )

        self.assertEqual(result.shape, [1, 1, 4, 4])
        np.testing.assert_allclose(result.numpy(), expected_np)


@unittest.skipIf(
    paddle.is_compiled_with_xpu(),
    "XPU does not support circular padding mode.",
)
class TestCircularPad3d(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_ncdhw_format(self):
        # NCDHW: (1, 1, 1, 2, 3)
        data_np = np.array([[[[[1, 2, 3], [4, 5, 6]]]]], dtype="float32")
        data = paddle.to_tensor(data_np)

        # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad = [1, 0, 1, 2, 0, 0]  # Pad W, H. D is unchanged.
        my_pad = CircularPad3d(padding=pad, data_format="NCDHW")
        assert (
            my_pad.__repr__()
            == "CircularPad3D(padding=[1, 0, 1, 2, 0, 0], mode=circular, value=0.0, data_format=NCDHW)"
        )
        result = my_pad(data)

        # Since D padding is (0, 0), this is just the 2D padding from
        # TestCircularPad2d.test_nchw_format applied to the single D-plane.
        expected_np = np.array(
            [
                [
                    [
                        [
                            [6.0, 4.0, 5.0, 6.0],
                            [3.0, 1.0, 2.0, 3.0],
                            [6.0, 4.0, 5.0, 6.0],
                            [3.0, 1.0, 2.0, 3.0],
                            [6.0, 4.0, 5.0, 6.0],
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
        my_pad = CircularPad3d(padding=pad, data_format="NDHWC")
        result = my_pad(data)

        # Since D padding is (0, 0), this is just the 2D padding from
        # TestCircularPad2d.test_nhwc_format applied to the single D-plane.
        expected_np = np.array(
            [
                [
                    [
                        [[6.0], [4.0], [5.0], [6.0]],
                        [[3.0], [1.0], [2.0], [3.0]],
                        [[6.0], [4.0], [5.0], [6.0]],
                        [[3.0], [1.0], [2.0], [3.0]],
                        [[6.0], [4.0], [5.0], [6.0]],
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
        my_pad = CircularPad3d(padding=pad, data_format="NCDHW")
        result = my_pad(data)

        # Shape: (1, 1, 1+1+1, 2+1+1, 2+1+1) = (1, 1, 3, 4, 4)
        self.assertEqual(result.shape, [1, 1, 3, 4, 4])

        # This is the 2D padded plane from TestCircularPad2d.test_int_padding
        expected_plane = np.array(
            [
                [4.0, 3.0, 4.0, 3.0],
                [2.0, 1.0, 2.0, 1.0],
                [4.0, 3.0, 4.0, 3.0],
                [2.0, 1.0, 2.0, 1.0],
            ],
            dtype="float32",
        )

        # Pad D (front=1, back=1)
        # Since there is only one D-plane, it wraps to pad both
        # front and back.
        # Result: [expected_plane, expected_plane, expected_plane]
        expected_np = np.array(
            [[[expected_plane, expected_plane, expected_plane]]],
            dtype="float32",
        )

        np.testing.assert_allclose(result.numpy(), expected_np)


if __name__ == "__main__":
    unittest.main()
