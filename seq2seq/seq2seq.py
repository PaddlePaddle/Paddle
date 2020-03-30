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
from rnn_api import DynamicDecode, RNN, BasicLSTMCell, RNNCell
from model import Model, Loss


class CrossEntropyCriterion(Loss):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, outputs, labels):
        (predict, mask), label = outputs, labels[0]

        cost = layers.softmax_with_cross_entropy(logits=predict,
                                                 label=label,
                                                 soft_label=False)
        masked_cost = layers.elementwise_mul(cost, mask, axis=0)
        batch_mean_cost = layers.reduce_mean(masked_cost, dim=[0])
        seq_cost = layers.reduce_sum(batch_mean_cost)
        return seq_cost


class EncoderCell(RNNCell):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(EncoderCell, self).__init__()
        self.dropout_prob = dropout_prob
        # use add_sublayer to add multi-layers
        self.lstm_cells = []
        for i in range(num_layers):
            self.lstm_cells.append(
                self.add_sublayer(
                    "lstm_%d" % i,
                    BasicLSTMCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        param_attr=ParamAttr(initializer=UniformInitializer(
                            low=-init_scale, high=init_scale)))))

    def forward(self, step_input, states):
        new_states = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_state = lstm_cell(step_input, states[i])
            step_input = layers.dropout(
                out, self.dropout_prob) if self.dropout_prob > 0 else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class Encoder(Layer):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(Encoder, self).__init__()
        self.embedder = Embedding(
            size=[vocab_size, embed_dim],
            param_attr=ParamAttr(initializer=UniformInitializer(
                low=-init_scale, high=init_scale)))
        self.stack_lstm = RNN(EncoderCell(num_layers, embed_dim, hidden_size,
                                          init_scale),
                              is_reverse=False,
                              time_major=False)

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence)
        encoder_output, encoder_state = self.stack_lstm(
            inputs, sequence_length=sequence_length)
        return encoder_output, encoder_state


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
        query = self.input_proj(hidden)
        attn_scores = layers.matmul(layers.unsqueeze(query, [1]),
                                    encoder_output,
                                    transpose_y=True)
        if encoder_padding_mask is not None:
            attn_scores = layers.elementwise_add(attn_scores,
                                                 encoder_padding_mask)
        attn_scores = layers.softmax(attn_scores)
        attn_out = layers.squeeze(layers.matmul(attn_scores, encoder_output),
                                  [1])
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
                    BasicLSTMCell(input_size=input_size +
                                  hidden_size if i == 0 else hidden_size,
                                  hidden_size=hidden_size)))
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
                out, self.dropout_prob) if self.dropout_prob > 0 else out
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
        self.lstm_attention = RNN(DecoderCell(num_layers, embed_dim,
                                              hidden_size, init_scale),
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
            decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
        predict = self.output_layer(decoder_output)
        return predict


class Seq2Seq(Model):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_size,
                               num_layers, dropout_prob, init_scale)
        self.decoder = Decoder(trg_vocab_size, embed_dim, hidden_size,
                               num_layers, dropout_prob, init_scale)

    def forward(self, src, src_length, trg, trg_length):
        # encoder
        encoder_output, encoder_final_state = self.encoder(src, src_length)

        # decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        # attention mask to avoid paying attention on padddings
        src_mask = layers.sequence_mask(src_length,
                                        maxlen=layers.shape(src)[1],
                                        dtype=encoder_output.dtype)
        encoder_padding_mask = (src_mask - 1.0) * 1e9
        encoder_padding_mask = layers.unsqueeze(encoder_padding_mask, [1])

        # decoder with attentioon
        predict = self.decoder(trg, decoder_initial_states, encoder_output,
                               encoder_padding_mask)

        # for target padding mask
        mask = layers.sequence_mask(trg_length,
                                    maxlen=layers.shape(trg)[1],
                                    dtype=predict.dtype)
        return predict, mask


class Seq2SeqInferModel(Seq2Seq):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.):
        pass
