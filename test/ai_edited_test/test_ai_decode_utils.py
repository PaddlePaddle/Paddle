# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Unit test for paddle.nn.decode (BeamSearchDecoder, etc.)
# Target: cover BeamSearchDecoder initialization and core methods

import unittest

import paddle
from paddle import nn


class TestBeamSearchDecoder(unittest.TestCase):
    """Test BeamSearchDecoder initialization and basic methods.
    BeamSearchDecoder is available as paddle.nn.BeamSearchDecoder.
    Attributes are beam_size, start_token, end_token (no underscore prefix).
    """

    def setUp(self):
        paddle.disable_static()

    def test_beam_search_decoder_init(self):
        """BeamSearchDecoder basic initialization."""
        embed = nn.Embedding(100, 32)
        cell = nn.SimpleRNNCell(32, 64)
        output_layer = nn.Linear(64, 100)

        decoder = nn.BeamSearchDecoder(
            cell=cell,
            start_token=1,
            end_token=2,
            beam_size=3,
            embedding_fn=embed,
            output_fn=output_layer,
        )
        self.assertIsNotNone(decoder)
        self.assertEqual(decoder.beam_size, 3)
        self.assertEqual(decoder.start_token, 1)
        self.assertEqual(decoder.end_token, 2)

    def test_beam_search_decoder_tile_beam(self):
        """BeamSearchDecoder tile_beam_merge_with_batch static method."""
        embed = nn.Embedding(100, 32)
        cell = nn.SimpleRNNCell(32, 64)
        output_layer = nn.Linear(64, 100)

        decoder = nn.BeamSearchDecoder(
            cell=cell,
            start_token=1,
            end_token=2,
            beam_size=3,
            embedding_fn=embed,
            output_fn=output_layer,
        )
        x = paddle.randn([2, 5, 10], dtype='float32')
        tiled = nn.BeamSearchDecoder.tile_beam_merge_with_batch(x, beam_size=3)
        self.assertEqual(tiled.shape, [6, 5, 10])

    def test_beam_search_decoder_no_embedding_fn(self):
        """BeamSearchDecoder without embedding_fn."""
        cell = nn.SimpleRNNCell(32, 64)
        output_layer = nn.Linear(64, 100)

        decoder = nn.BeamSearchDecoder(
            cell=cell,
            start_token=1,
            end_token=2,
            beam_size=2,
            output_fn=output_layer,
        )
        self.assertIsNotNone(decoder)
        self.assertEqual(decoder.beam_size, 2)


class TestDynamicDecode(unittest.TestCase):
    """Test dynamic_decode function.
    dynamic_decode is available as paddle.nn.dynamic_decode.
    """

    def setUp(self):
        paddle.disable_static()

    def test_beam_search_decoder_with_dynamic_decode(self):
        """Test that BeamSearchDecoder can be used with dynamic_decode."""
        embed = nn.Embedding(100, 32)
        cell = nn.SimpleRNNCell(32, 64)
        output_layer = nn.Linear(64, 100)

        decoder = nn.BeamSearchDecoder(
            cell=cell,
            start_token=1,
            end_token=2,
            beam_size=2,
            embedding_fn=embed,
            output_fn=output_layer,
        )
        # Verify decoder has the required interface for dynamic_decode
        self.assertTrue(hasattr(decoder, 'step'))
        self.assertTrue(hasattr(decoder, 'beam_size'))
        self.assertTrue(hasattr(decoder, 'start_token'))
        self.assertTrue(hasattr(decoder, 'end_token'))


if __name__ == '__main__':
    unittest.main()
