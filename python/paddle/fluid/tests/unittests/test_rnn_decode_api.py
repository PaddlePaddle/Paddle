# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core

from paddle.fluid.executor import Executor
from paddle.fluid import framework

from paddle.fluid.layers.rnn import LSTMCell, GRUCell, RNNCell, BeamSearchDecoder, dynamic_decode
from paddle.fluid.layers import rnn as dynamic_rnn
from paddle.fluid import contrib
from paddle.fluid.contrib.layers import basic_lstm

import numpy as np


class EncoderCell(RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = []
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size))

    def call(self, step_input, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](step_input, states[i])
            step_input = layers.dropout(
                out, self.dropout_prob) if self.dropout_prob > 0 else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class DecoderCell(RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = []
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size))

    def attention(self, hidden, encoder_output, encoder_padding_mask):
        query = layers.fc(hidden,
                          size=encoder_output.shape[-1],
                          bias_attr=False)
        attn_scores = layers.matmul(
            layers.unsqueeze(query, [1]), encoder_output, transpose_y=True)
        if encoder_padding_mask is not None:
            attn_scores = layers.elementwise_add(attn_scores,
                                                 encoder_padding_mask)
        attn_scores = layers.softmax(attn_scores)
        attn_out = layers.squeeze(
            layers.matmul(attn_scores, encoder_output), [1])
        attn_out = layers.concat([attn_out, hidden], 1)
        attn_out = layers.fc(attn_out, size=self.hidden_size, bias_attr=False)
        return attn_out

    def call(self,
             step_input,
             states,
             encoder_output,
             encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = layers.concat([step_input, input_feed], 1)
        for i in range(self.num_layers):
            out, new_lstm_state = self.lstm_cells[i](step_input, lstm_states[i])
            step_input = layers.dropout(
                out, self.dropout_prob) if self.dropout_prob > 0 else out
            new_lstm_states.append(new_lstm_state)
        out = self.attention(step_input, encoder_output, encoder_padding_mask)
        return out, [new_lstm_states, out]


class TestDynamicDecode(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16
        self.seq_len = 4

    def test_run(self):
        start_token = 0
        end_token = 1
        src_vocab_size = 10
        trg_vocab_size = 10
        num_layers = 1
        hidden_size = self.hidden_size
        beam_size = 8
        max_length = self.seq_len

        src = layers.data(name="src", shape=[-1, 1], dtype='int64')
        src_len = layers.data(name="src_len", shape=[-1], dtype='int64')

        trg = layers.data(name="trg", shape=[-1, 1], dtype='int64')
        trg_len = layers.data(name="trg_len", shape=[-1], dtype='int64')

        src_embeder = lambda x: fluid.embedding(
            x,
            size=[src_vocab_size, hidden_size],
            param_attr=fluid.ParamAttr(name="src_embedding"))

        trg_embeder = lambda x: fluid.embedding(
            x,
            size=[trg_vocab_size, hidden_size],
            param_attr=fluid.ParamAttr(name="trg_embedding"))

        # use basic_lstm
        encoder_cell = EncoderCell(num_layers, hidden_size)
        encoder_output, encoder_final_state = dynamic_rnn(
            cell=encoder_cell,
            inputs=src_embeder(src),
            sequence_length=src_len,
            is_reverse=False)

        src_mask = layers.sequence_mask(
            src_len, maxlen=layers.shape(src)[1], dtype='float32')
        encoder_padding_mask = (src_mask - 1.0) * 1000000000
        encoder_padding_mask = layers.unsqueeze(encoder_padding_mask, [1])

        decoder_cell = DecoderCell(num_layers, hidden_size)
        decoder_initial_states = [
            encoder_final_state, decoder_cell.get_initial_states(
                batch_ref=encoder_output, shape=[hidden_size])
        ]

        decoder_output, _ = dynamic_rnn(
            cell=decoder_cell,
            inputs=trg_embeder(trg),
            initial_states=decoder_initial_states,
            sequence_length=None,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)

        output_layer = lambda x: layers.fc(x,
                                           size=trg_vocab_size,
                                           num_flatten_dims=len(x.shape) - 1,
                                           param_attr=fluid.ParamAttr(
                                               name="output_w"),
                                           bias_attr=False)

        # inference
        encoder_output = BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, beam_size)
        encoder_padding_mask = BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, beam_size)
        beam_search_decoder = BeamSearchDecoder(
            decoder_cell,
            start_token,
            end_token,
            beam_size,
            embedding_fn=trg_embeder,
            output_fn=output_layer)
        outputs, _ = dynamic_decode(
            beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=max_length,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        src_np = np.random.randint(
            0, src_vocab_size, (self.batch_size, max_length)).astype('int64')
        src_len_np = np.ones(self.batch_size, dtype='int64') * max_length
        trg_np = np.random.randint(
            0, trg_vocab_size, (self.batch_size, max_length)).astype('int64')
        trg_len_np = np.ones(self.batch_size, dtype='int64') * max_length

        out = exe.run(feed={
            'src': src_np,
            'src_len': src_len_np,
            'trg': trg_np,
            'trg_len': trg_len_np
        },
                      fetch_list=[outputs])

        self.assertTrue(out[0].shape[0] == self.batch_size)
        self.assertTrue(out[0].shape[1] <= max_length + 1)
        self.assertTrue(out[0].shape[2] == beam_size)


if __name__ == '__main__':
    unittest.main()
