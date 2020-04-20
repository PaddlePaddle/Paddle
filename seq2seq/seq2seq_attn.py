# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import UniformInitializer
from paddle.fluid.dygraph import Embedding, Linear, Layer
from paddle.fluid.layers import BeamSearchDecoder

from text import DynamicDecode, RNN, BasicLSTMCell, RNNCell
from model import Model, Loss
from seq2seq_base import Encoder


class AttentionLayer(Layer):
    def __init__(self, hidden_size, bias=False, init_scale=0.1):
        super(AttentionLayer, self).__init__()
        self.input_proj = Linear(
            hidden_size,
            hidden_size,
            param_attr=ParamAttr(initializer=UniformInitializer(
                low=-init_scale, high=init_scale)),
            bias_attr=bias)
        self.output_proj = Linear(
            hidden_size + hidden_size,
            hidden_size,
            param_attr=ParamAttr(initializer=UniformInitializer(
                low=-init_scale, high=init_scale)),
            bias_attr=bias)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        # query = self.input_proj(hidden)
        encoder_output = self.input_proj(encoder_output)
        attn_scores = layers.matmul(
            layers.unsqueeze(hidden, [1]), encoder_output, transpose_y=True)
        if encoder_padding_mask is not None:
            attn_scores = layers.elementwise_add(attn_scores,
                                                 encoder_padding_mask)
        attn_scores = layers.softmax(attn_scores)
        attn_out = layers.squeeze(
            layers.matmul(attn_scores, encoder_output), [1])
        attn_out = layers.concat([attn_out, hidden], 1)
        attn_out = self.output_proj(attn_out)
        return attn_out


class DecoderCell(RNNCell):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(DecoderCell, self).__init__()
        self.dropout_prob = dropout_prob
        # use add_sublayer to add multi-layers
        self.lstm_cells = []
        for i in range(num_layers):
            self.lstm_cells.append(
                self.add_sublayer(
                    "lstm_%d" % i,
                    BasicLSTMCell(
                        input_size=input_size + hidden_size
                        if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        param_attr=ParamAttr(initializer=UniformInitializer(
                            low=-init_scale, high=init_scale)))))
        self.attention_layer = AttentionLayer(hidden_size)

    def forward(self,
                step_input,
                states,
                encoder_output,
                encoder_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = layers.concat([step_input, input_feed], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
                dropout_implementation='upscale_in_train'
            ) if self.dropout_prob > 0 else out
            new_lstm_states.append(new_lstm_state)
        out = self.attention_layer(step_input, encoder_output,
                                   encoder_padding_mask)
        return out, [new_lstm_states, out]


class Decoder(Layer):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(Decoder, self).__init__()
        self.embedder = Embedding(
            size=[vocab_size, embed_dim],
            param_attr=ParamAttr(initializer=UniformInitializer(
                low=-init_scale, high=init_scale)))
        self.lstm_attention = RNN(DecoderCell(
            num_layers, embed_dim, hidden_size, dropout_prob, init_scale),
                                  is_reverse=False,
                                  time_major=False)
        self.output_layer = Linear(
            hidden_size,
            vocab_size,
            param_attr=ParamAttr(initializer=UniformInitializer(
                low=-init_scale, high=init_scale)),
            bias_attr=False)

    def forward(self, target, decoder_initial_states, encoder_output,
                encoder_padding_mask):
        inputs = self.embedder(target)
        decoder_output, _ = self.lstm_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        predict = self.output_layer(decoder_output)
        return predict


class AttentionModel(Model):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_size,
                               num_layers, dropout_prob, init_scale)
        self.decoder = Decoder(trg_vocab_size, embed_dim, hidden_size,
                               num_layers, dropout_prob, init_scale)

    def forward(self, src, src_length, trg):
        # encoder
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        # decoder initial states: use input_feed and the structure is
        # [[h,c] * num_layers, input_feed], consistent with DecoderCell.states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # attention mask to avoid paying attention on padddings
        src_mask = layers.sequence_mask(
            src_length,
            maxlen=layers.shape(src)[1],
            dtype=encoder_output.dtype)
        encoder_padding_mask = (src_mask - 1.0) * 1e9
        encoder_padding_mask = layers.unsqueeze(encoder_padding_mask, [1])

        # decoder with attentioon
        predict = self.decoder(trg, decoder_initial_states, encoder_output,
                               encoder_padding_mask)
        return predict


class AttentionInferModel(AttentionModel):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)  # py3
        self.bos_id = args.pop("bos_id")
        self.eos_id = args.pop("eos_id")
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        super(AttentionInferModel, self).__init__(**args)
        # dynamic decoder for inference
        decoder = BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)
        self.beam_search_decoder = DynamicDecode(
            decoder, max_step_num=max_out_len, is_test=True)

    def forward(self, src, src_length):
        # encoding
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        # decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # attention mask to avoid paying attention on padddings
        src_mask = layers.sequence_mask(
            src_length,
            maxlen=layers.shape(src)[1],
            dtype=encoder_output.dtype)
        encoder_padding_mask = (src_mask - 1.0) * 1e9
        encoder_padding_mask = layers.unsqueeze(encoder_padding_mask, [1])

        # Tile the batch dimension with beam_size
        encoder_output = BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_output, self.beam_size)
        encoder_padding_mask = BeamSearchDecoder.tile_beam_merge_with_batch(
            encoder_padding_mask, self.beam_size)

        # dynamic decoding with beam search
        rs, _ = self.beam_search_decoder(
            inits=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        return rs


class GreedyEmbeddingHelper(fluid.layers.GreedyEmbeddingHelper):
    def __init__(self, embedding_fn, start_tokens, end_token):
        if isinstance(start_tokens, int):
            self.need_convert_start_tokens = True
            self.start_token_value = start_tokens
        super(GreedyEmbeddingHelper, self).__init__(embedding_fn, start_tokens,
                                                    end_token)
        self.end_token = fluid.layers.create_global_var(
            shape=[1], dtype="int64", value=end_token, persistable=True)

    def initialize(self, batch_ref=None):
        if getattr(self, "need_convert_start_tokens", False):
            assert batch_ref is not None, (
                "Need to give batch_ref to get batch size "
                "to initialize the tensor for start tokens.")
            self.start_tokens = fluid.layers.fill_constant_batch_size_like(
                input=fluid.layers.utils.flatten(batch_ref)[0],
                shape=[-1],
                dtype="int64",
                value=self.start_token_value,
                input_dim_idx=0)
        return super(GreedyEmbeddingHelper, self).initialize()


class BasicDecoder(fluid.layers.BasicDecoder):
    def initialize(self, initial_cell_states):
        (initial_inputs,
         initial_finished) = self.helper.initialize(initial_cell_states)
        return initial_inputs, initial_cell_states, initial_finished


class AttentionGreedyInferModel(AttentionModel):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 bos_id=0,
                 eos_id=1,
                 beam_size=1,
                 max_out_len=256):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)  # py3
        args.pop("beam_size", None)
        self.bos_id = args.pop("bos_id")
        self.eos_id = args.pop("eos_id")
        self.max_out_len = args.pop("max_out_len")
        super(AttentionGreedyInferModel, self).__init__(**args)
        # dynamic decoder for inference
        decoder_helper = GreedyEmbeddingHelper(
            start_tokens=bos_id,
            end_token=eos_id,
            embedding_fn=self.decoder.embedder)
        decoder = BasicDecoder(
            cell=self.decoder.lstm_attention.cell,
            helper=decoder_helper,
            output_fn=self.decoder.output_layer)
        self.greedy_search_decoder = DynamicDecode(
            decoder, max_step_num=max_out_len, is_test=True)

    def forward(self, src, src_length):
        # encoding
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        # decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # attention mask to avoid paying attention on padddings
        src_mask = layers.sequence_mask(
            src_length,
            maxlen=layers.shape(src)[1],
            dtype=encoder_output.dtype)
        encoder_padding_mask = (src_mask - 1.0) * 1e9
        encoder_padding_mask = layers.unsqueeze(encoder_padding_mask, [1])

        # dynamic decoding with greedy search
        rs, _ = self.greedy_search_decoder(
            inits=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        return rs.sample_ids
