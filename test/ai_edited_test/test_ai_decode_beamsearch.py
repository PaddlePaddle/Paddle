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

# [AUTO-GENERATED] Unit test for paddle.nn.decode
# 自动生成的单测，覆盖 paddle.nn.decode 模块中未覆盖的代码
# Target: paddle/nn/decode.py

"""
测试模块：paddle.nn.decode
Test Module: paddle.nn.decode

本测试覆盖以下功能：
This test covers the following functions:
1. ArrayWrapper - 数组包装器 / Array wrapper append and indexing
2. Decoder base class - 解码器基类 / Decoder base class properties
3. BeamSearchDecoder - 束搜索解码器 / BeamSearchDecoder initialization, tile_beam_merge, step
4. BeamSearchDecoder helper methods - 辅助方法 / _split_batch_beams, _merge_batch_beams, _expand_to_beam_size
5. dynamic_decode - 动态解码 / dynamic_decode with max_step_num, return_length
"""

import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.nn import BeamSearchDecoder, dynamic_decode


class TestArrayWrapper(unittest.TestCase):
    """测试ArrayWrapper数组包装器
    Test ArrayWrapper"""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_array_wrapper_init(self):
        """测试初始化 / Test initialization"""
        from paddle.nn.decode import ArrayWrapper

        wrapper = ArrayWrapper(42)
        self.assertEqual(wrapper[0], 42)

    def test_array_wrapper_append(self):
        """测试追加 / Test append"""
        from paddle.nn.decode import ArrayWrapper

        wrapper = ArrayWrapper(1)
        result = wrapper.append(2)
        self.assertIs(result, wrapper)
        self.assertEqual(wrapper[0], 1)
        self.assertEqual(wrapper[1], 2)

    def test_array_wrapper_getitem(self):
        """测试索引 / Test getitem"""
        from paddle.nn.decode import ArrayWrapper

        wrapper = ArrayWrapper(10)
        wrapper.append(20)
        wrapper.append(30)
        self.assertEqual(wrapper[0], 10)
        self.assertEqual(wrapper[1], 20)
        self.assertEqual(wrapper[2], 30)


class TestDecoderBase(unittest.TestCase):
    """测试Decoder基类
    Test Decoder base class"""

    def test_tracks_own_finished(self):
        """测试tracks_own_finished默认值 / Test tracks_own_finished default"""
        from paddle.nn.decode import Decoder

        decoder = Decoder()
        self.assertFalse(decoder.tracks_own_finished)

    def test_not_implemented_initialize(self):
        """测试initialize未实现 / Test initialize raises NotImplementedError"""
        from paddle.nn.decode import Decoder

        decoder = Decoder()
        with self.assertRaises(NotImplementedError):
            decoder.initialize(None)

    def test_not_implemented_step(self):
        """测试step未实现 / Test step raises NotImplementedError"""
        from paddle.nn.decode import Decoder

        decoder = Decoder()
        with self.assertRaises(NotImplementedError):
            decoder.step(None, None, None)

    def test_not_implemented_finalize(self):
        """测试finalize未实现 / Test finalize raises NotImplementedError"""
        from paddle.nn.decode import Decoder

        decoder = Decoder()
        with self.assertRaises(NotImplementedError):
            decoder.finalize(None, None, None)


class TestBeamSearchDecoderHelperMethods(unittest.TestCase):
    """测试BeamSearchDecoder辅助方法
    Test BeamSearchDecoder helper methods"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_tile_beam_merge_with_batch(self):
        """测试tile_beam_merge_with_batch / Test tile_beam_merge_with_batch"""
        x = paddle.randn([2, 4])
        result = BeamSearchDecoder.tile_beam_merge_with_batch(x, beam_size=3)
        self.assertEqual(result.shape, [6, 4])

    def test_tile_beam_merge_3d(self):
        """测试tile_beam_merge_with_batch 3D / Test with 3D tensor"""
        x = paddle.randn([2, 4, 8])
        result = BeamSearchDecoder.tile_beam_merge_with_batch(x, beam_size=3)
        self.assertEqual(result.shape, [6, 4, 8])

    def test_split_batch_beams(self):
        """测试_split_batch_beams / Test _split_batch_beams"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=3
        )
        x = paddle.randn([6, 8])  # batch_size * beam_size = 2*3
        result = decoder._split_batch_beams(x)
        self.assertEqual(result.shape, [2, 3, 8])

    def test_merge_batch_beams(self):
        """测试_merge_batch_beams / Test _merge_batch_beams"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=3
        )
        x = paddle.randn([2, 3, 8])  # batch_size, beam_size, feature
        result = decoder._merge_batch_beams(x)
        self.assertEqual(result.shape, [6, 8])

    def test_expand_to_beam_size(self):
        """测试_expand_to_beam_size / Test _expand_to_beam_size"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=3
        )
        x = paddle.randn([2, 8])
        result = decoder._expand_to_beam_size(x)
        self.assertEqual(result.shape, [2, 3, 8])


class TestBeamSearchDecoderInit(unittest.TestCase):
    """测试BeamSearchDecoder初始化
    Test BeamSearchDecoder initialization"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_tracks_own_finished(self):
        """测试tracks_own_finished属性 / Test tracks_own_finished property"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=4
        )
        self.assertTrue(decoder.tracks_own_finished)

    def test_initialize_basic(self):
        """测试基本初始化 / Test basic initialize"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=4
        )
        initial_states = cell.get_initial_states(paddle.randn([2, 6, 8]))
        init_inputs, init_states, init_finished = decoder.initialize(
            initial_states
        )
        self.assertEqual(init_finished.shape, [2, 4])
        self.assertTrue(paddle.all(~init_finished))

    def test_initialize_with_embedding(self):
        """测试带embedding_fn的初始化 / Test initialize with embedding_fn"""
        embedder = nn.Embedding(10, 8)
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=4, embedding_fn=embedder
        )
        initial_states = cell.get_initial_states(paddle.randn([2, 6, 8]))
        init_inputs, init_states, init_finished = decoder.initialize(
            initial_states
        )
        # With embedding_fn, init_inputs is the embedded result (batch_size * beam_size, embed_dim)
        self.assertEqual(init_inputs.shape[-1], 8)

    def test_output_wrapper(self):
        """测试OutputWrapper命名元组 / Test OutputWrapper namedtuple"""
        from paddle.nn.decode import BeamSearchDecoder

        scores = paddle.randn([2, 4])
        predicted_ids = paddle.randint(0, 10, [2, 4])
        parent_ids = paddle.randint(0, 4, [2, 4])
        wrapper = BeamSearchDecoder.OutputWrapper(
            scores, predicted_ids, parent_ids
        )
        self.assertEqual(wrapper.scores.shape, [2, 4])
        self.assertEqual(wrapper.predicted_ids.shape, [2, 4])
        self.assertEqual(wrapper.parent_ids.shape, [2, 4])

    def test_state_wrapper(self):
        """测试StateWrapper命名元组 / Test StateWrapper namedtuple"""
        from paddle.nn.decode import BeamSearchDecoder

        cell_states = paddle.randn([8, 8])
        log_probs = paddle.randn([2, 4])
        finished = paddle.zeros([2, 4], dtype='bool')
        lengths = paddle.zeros([2, 4], dtype='int64')
        wrapper = BeamSearchDecoder.StateWrapper(
            cell_states, log_probs, finished, lengths
        )
        self.assertEqual(wrapper.log_probs.shape, [2, 4])
        self.assertEqual(wrapper.finished.shape, [2, 4])
        self.assertEqual(wrapper.lengths.shape, [2, 4])


class TestDynamicDecodeBasic(unittest.TestCase):
    """测试dynamic_decode基本功能
    Test dynamic_decode basic functionality"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_dynamic_decode_basic(self):
        """测试基本动态解码 / Test basic dynamic_decode"""
        embedder = nn.Embedding(20, 8)
        output_layer = nn.Linear(8, 20)
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell,
            start_token=0,
            end_token=1,
            beam_size=2,
            embedding_fn=embedder,
            output_fn=output_layer,
        )
        encoder_output = paddle.randn([2, 6, 8])
        initial_states = cell.get_initial_states(encoder_output)
        outputs, final_states = dynamic_decode(
            decoder=decoder, inits=initial_states, max_step_num=5
        )
        # outputs should have shape [batch_size, seq_len, beam_size]
        self.assertEqual(len(outputs.shape), 3)

    def test_dynamic_decode_return_length(self):
        """测试return_length / Test dynamic_decode with return_length=True"""
        embedder = nn.Embedding(20, 8)
        output_layer = nn.Linear(8, 20)
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell,
            start_token=0,
            end_token=1,
            beam_size=2,
            embedding_fn=embedder,
            output_fn=output_layer,
        )
        encoder_output = paddle.randn([2, 6, 8])
        initial_states = cell.get_initial_states(encoder_output)
        result = dynamic_decode(
            decoder=decoder,
            inits=initial_states,
            max_step_num=5,
            return_length=True,
        )
        self.assertEqual(len(result), 3)

    def test_dynamic_decode_output_time_major(self):
        """测试output_time_major / Test dynamic_decode with output_time_major=True"""
        embedder = nn.Embedding(20, 8)
        output_layer = nn.Linear(8, 20)
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell,
            start_token=0,
            end_token=1,
            beam_size=2,
            embedding_fn=embedder,
            output_fn=output_layer,
        )
        encoder_output = paddle.randn([2, 6, 8])
        initial_states = cell.get_initial_states(encoder_output)
        outputs, _ = dynamic_decode(
            decoder=decoder,
            inits=initial_states,
            max_step_num=5,
            output_time_major=True,
        )
        # time_major: shape [seq_len, batch_size, beam_size]
        self.assertEqual(outputs.shape[1], 2)  # batch_size


class TestMaskProbs(unittest.TestCase):
    """测试_mask_probs
    Test _mask_probs"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_mask_probs_unfinished(self):
        """测试未完成的概率掩码 / Test mask_probs with unfinished beams"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=2
        )
        vocab_size = 10
        probs = paddle.randn([2, 2, vocab_size])
        finished = paddle.zeros([2, 2], dtype='bool')
        # Manually set up decoder state
        decoder.vocab_size = vocab_size
        noend_array = [-1e9] * vocab_size
        noend_array[1] = 0
        decoder.noend_mask_tensor = paddle.assign(
            np.array(noend_array, "float32")
        )
        result = decoder._mask_probs(probs, finished)
        self.assertEqual(result.shape, [2, 2, vocab_size])

    def test_mask_probs_finished(self):
        """测试已完成的概率掩码 / Test mask_probs with finished beams"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=2
        )
        vocab_size = 10
        probs = paddle.randn([2, 2, vocab_size])
        finished = paddle.ones([2, 2], dtype='bool')
        decoder.vocab_size = vocab_size
        noend_array = [-1e9] * vocab_size
        noend_array[1] = 0
        decoder.noend_mask_tensor = paddle.assign(
            np.array(noend_array, "float32")
        )
        result = decoder._mask_probs(probs, finished)
        self.assertEqual(result.shape, [2, 2, vocab_size])


class TestGatherMethod(unittest.TestCase):
    """测试_gather方法
    Test _gather method"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_gather_basic(self):
        """测试基本gather / Test basic gather"""
        cell = nn.GRUCell(input_size=8, hidden_size=8)
        decoder = BeamSearchDecoder(
            cell, start_token=0, end_token=1, beam_size=2
        )
        x = paddle.randn([2, 2, 8])  # batch_size, beam_size, feature
        indices = paddle.to_tensor([[0, 1], [1, 0]], dtype='int64')
        batch_size = paddle.to_tensor([2], dtype='int64')
        result = decoder._gather(x, indices, batch_size)
        self.assertEqual(result.shape, [2, 2, 8])


if __name__ == '__main__':
    unittest.main()
