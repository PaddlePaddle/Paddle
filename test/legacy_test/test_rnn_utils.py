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
from paddle.nn.utils.rnn import pad_sequence, unpad_sequence


class TestPadSequence(unittest.TestCase):
    """Tests for paddle.nn.utils.pad_sequence."""

    def test_basic_batch_first_false(self):
        """Test basic padding with batch_first=False (default)."""
        a = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # [3, 2]
        b = paddle.to_tensor([[7.0, 8.0]])  # [1, 2]
        result = pad_sequence([a, b])
        # Output shape: T x B x * = [3, 2, 2]
        self.assertEqual(result.shape, [3, 2, 2])
        # First sequence should be unchanged
        np.testing.assert_allclose(result[:, 0, :].numpy(), a.numpy())
        # Second sequence: first row is original, rest are padding (0)
        np.testing.assert_allclose(result[0, 1, :].numpy(), [7.0, 8.0])
        np.testing.assert_allclose(result[1, 1, :].numpy(), [0.0, 0.0])
        np.testing.assert_allclose(result[2, 1, :].numpy(), [0.0, 0.0])

    def test_basic_batch_first_true(self):
        """Test basic padding with batch_first=True."""
        a = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # [3, 2]
        b = paddle.to_tensor([[7.0, 8.0]])  # [1, 2]
        result = pad_sequence([a, b], batch_first=True)
        # Output shape: B x T x * = [2, 3, 2]
        self.assertEqual(result.shape, [2, 3, 2])
        np.testing.assert_allclose(result[0].numpy(), a.numpy())
        np.testing.assert_allclose(result[1, 0, :].numpy(), [7.0, 8.0])
        np.testing.assert_allclose(result[1, 1, :].numpy(), [0.0, 0.0])

    def test_custom_padding_value(self):
        """Test padding with a non-zero padding value."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0])
        result = pad_sequence([a, b], batch_first=True, padding_value=-1.0)
        self.assertEqual(result.shape, [2, 3])
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[1].numpy(), [4.0, -1.0, -1.0])

    def test_padding_side_left(self):
        """Test left-side padding."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0])
        result = pad_sequence([a, b], batch_first=True, padding_side='left')
        self.assertEqual(result.shape, [2, 3])
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[1].numpy(), [0.0, 0.0, 4.0])

    def test_padding_side_left_with_value(self):
        """Test left-side padding with custom value."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0])
        result = pad_sequence(
            [a, b],
            batch_first=True,
            padding_value=-1.0,
            padding_side='left',
        )
        np.testing.assert_allclose(result[1].numpy(), [-1.0, -1.0, 4.0])

    def test_single_sequence(self):
        """Test with a single sequence (no padding needed)."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        result = pad_sequence([a], batch_first=True)
        self.assertEqual(result.shape, [1, 3])
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0, 3.0])

    def test_equal_length_sequences(self):
        """Test with sequences of equal length (no padding needed)."""
        a = paddle.to_tensor([1.0, 2.0])
        b = paddle.to_tensor([3.0, 4.0])
        result = pad_sequence([a, b], batch_first=True)
        self.assertEqual(result.shape, [2, 2])
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0])
        np.testing.assert_allclose(result[1].numpy(), [3.0, 4.0])

    def test_multidimensional_sequences(self):
        """Test with multi-dimensional trailing dimensions."""
        a = paddle.ones([5, 3, 4])
        b = paddle.ones([3, 3, 4]) * 2
        result = pad_sequence([a, b], batch_first=True)
        self.assertEqual(result.shape, [2, 5, 3, 4])
        # First sequence: all ones
        np.testing.assert_allclose(result[0].numpy(), np.ones([5, 3, 4]))
        # Second sequence: first 3 rows are 2s, last 2 rows are 0s
        np.testing.assert_allclose(
            result[1, :3].numpy(), np.full([3, 3, 4], 2.0)
        )
        np.testing.assert_allclose(result[1, 3:].numpy(), np.zeros([2, 3, 4]))

    def test_0d_trailing_dims(self):
        """Test with 1D tensors (no trailing dimensions)."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0, 5.0])
        c = paddle.to_tensor([6.0])
        result = pad_sequence([a, b, c], batch_first=True)
        self.assertEqual(result.shape, [3, 3])
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[1].numpy(), [4.0, 5.0, 0.0])
        np.testing.assert_allclose(result[2].numpy(), [6.0, 0.0, 0.0])

    def test_error_not_list(self):
        """Test TypeError when input is not a list."""
        with self.assertRaises(TypeError):
            pad_sequence(paddle.to_tensor([1.0, 2.0]))

    def test_error_invalid_padding_side(self):
        """Test ValueError for invalid padding_side."""
        a = paddle.to_tensor([1.0])
        with self.assertRaises(ValueError):
            pad_sequence([a], padding_side='center')

    def test_integer_dtype(self):
        """Test with integer dtype tensors."""
        a = paddle.to_tensor([1, 2, 3])
        b = paddle.to_tensor([4])
        result = pad_sequence([a, b], batch_first=True, padding_value=0)
        self.assertEqual(result.shape, [2, 3])
        np.testing.assert_array_equal(result[0].numpy(), [1, 2, 3])
        np.testing.assert_array_equal(result[1].numpy(), [4, 0, 0])


class TestUnpadSequence(unittest.TestCase):
    """Tests for paddle.nn.utils.unpad_sequence."""

    def test_basic_batch_first_false(self):
        """Test basic unpadding with batch_first=False (default)."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0])
        padded = pad_sequence([a, b])  # T x B = [3, 2]
        lengths = paddle.to_tensor([3, 1])
        result = unpad_sequence(padded, lengths)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].numpy(), a.numpy())
        np.testing.assert_allclose(result[1].numpy(), b.numpy())

    def test_basic_batch_first_true(self):
        """Test basic unpadding with batch_first=True."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0])
        padded = pad_sequence([a, b], batch_first=True)  # B x T = [2, 3]
        lengths = paddle.to_tensor([3, 1])
        result = unpad_sequence(padded, lengths, batch_first=True)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].numpy(), a.numpy())
        np.testing.assert_allclose(result[1].numpy(), b.numpy())

    def test_roundtrip(self):
        """Test pad then unpad recovers original sequences."""
        sequences = [
            paddle.randn([10, 5]),
            paddle.randn([7, 5]),
            paddle.randn([3, 5]),
        ]
        padded = pad_sequence(sequences, batch_first=True)
        lengths = paddle.to_tensor([s.shape[0] for s in sequences])
        result = unpad_sequence(padded, lengths, batch_first=True)
        self.assertEqual(len(result), 3)
        for orig, recovered in zip(sequences, result):
            np.testing.assert_allclose(
                recovered.numpy(), orig.numpy(), rtol=1e-6
            )

    def test_roundtrip_batch_first_false(self):
        """Test round-trip with batch_first=False."""
        sequences = [
            paddle.randn([8, 4]),
            paddle.randn([5, 4]),
            paddle.randn([2, 4]),
        ]
        padded = pad_sequence(sequences)  # T x B x D
        lengths = paddle.to_tensor([s.shape[0] for s in sequences])
        result = unpad_sequence(padded, lengths)
        self.assertEqual(len(result), 3)
        for orig, recovered in zip(sequences, result):
            np.testing.assert_allclose(
                recovered.numpy(), orig.numpy(), rtol=1e-6
            )

    def test_multidimensional(self):
        """Test unpadding with multi-dimensional trailing dims."""
        sequences = [
            paddle.randn([6, 3, 2]),
            paddle.randn([4, 3, 2]),
        ]
        padded = pad_sequence(sequences, batch_first=True)
        lengths = paddle.to_tensor([6, 4])
        result = unpad_sequence(padded, lengths, batch_first=True)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [6, 3, 2])
        self.assertEqual(result[1].shape, [4, 3, 2])
        for orig, recovered in zip(sequences, result):
            np.testing.assert_allclose(
                recovered.numpy(), orig.numpy(), rtol=1e-6
            )

    def test_equal_length(self):
        """Test unpadding when all sequences have equal length."""
        a = paddle.to_tensor([1.0, 2.0])
        b = paddle.to_tensor([3.0, 4.0])
        padded = pad_sequence([a, b], batch_first=True)
        lengths = paddle.to_tensor([2, 2])
        result = unpad_sequence(padded, lengths, batch_first=True)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].numpy(), a.numpy())
        np.testing.assert_allclose(result[1].numpy(), b.numpy())

    def test_single_sequence(self):
        """Test unpadding a single sequence."""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        padded = pad_sequence([a], batch_first=True)
        lengths = paddle.to_tensor([3])
        result = unpad_sequence(padded, lengths, batch_first=True)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].numpy(), a.numpy())


class TestPadUnpadIntegration(unittest.TestCase):
    """Integration tests combining pad_sequence and unpad_sequence."""

    def test_left_pad_unpad_roundtrip(self):
        """Test round-trip with left padding."""
        sequences = [
            paddle.to_tensor([1.0, 2.0, 3.0]),
            paddle.to_tensor([4.0]),
        ]
        padded = pad_sequence(sequences, batch_first=True, padding_side='left')
        # With left padding, unpad needs adjusted logic - the data is at the end
        # unpad_sequence always slices from the beginning, so it works with
        # right-padded data. For left-padded data, we need to slice from the end.
        # This test verifies the padded values are correct.
        np.testing.assert_allclose(padded[0].numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(padded[1].numpy(), [0.0, 0.0, 4.0])

    def test_various_dtypes(self):
        """Test pad_sequence preserves dtype."""
        for dtype in [
            paddle.float32,
            paddle.float64,
            paddle.int32,
            paddle.int64,
        ]:
            a = paddle.ones([3], dtype=dtype)
            b = paddle.ones([1], dtype=dtype)
            result = pad_sequence([a, b], batch_first=True)
            self.assertEqual(result.dtype, dtype)


if __name__ == '__main__':
    unittest.main()
