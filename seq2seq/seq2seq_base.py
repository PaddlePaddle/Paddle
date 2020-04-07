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


class CrossEntropyCriterion(Loss):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, outputs, labels):
        (predict, mask), label = outputs, labels[0]

        cost = layers.softmax_with_cross_entropy(
            logits=predict, label=label, soft_label=False)
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
                out,
                self.dropout_prob,
                dropout_implementation='upscale_in_train'
            ) if self.dropout_prob > 0 else out
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
                                          dropout_prob, init_scale),
                              is_reverse=False,
                              time_major=False)

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence)
        encoder_output, encoder_state = self.stack_lstm(
            inputs, sequence_length=sequence_length)
        return encoder_output, encoder_state


DecoderCell = EncoderCell


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
        self.stack_lstm = RNN(DecoderCell(num_layers, embed_dim, hidden_size,
                                          dropout_prob, init_scale),
                              is_reverse=False,
                              time_major=False)
        self.output_layer = Linear(
            hidden_size,
            vocab_size,
            param_attr=ParamAttr(initializer=UniformInitializer(
                low=-init_scale, high=init_scale)),
            bias_attr=False)

    def forward(self, target, decoder_initial_states):
        inputs = self.embedder(target)
        decoder_output, _ = self.stack_lstm(
            inputs, initial_states=decoder_initial_states)
        predict = self.output_layer(decoder_output)
        return predict


class BaseModel(Model):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_size,
                               num_layers, dropout_prob, init_scale)
        self.decoder = Decoder(trg_vocab_size, embed_dim, hidden_size,
                               num_layers, dropout_prob, init_scale)

    def forward(self, src, src_length, trg, trg_length):
        # encoder
        encoder_output, encoder_final_states = self.encoder(src, src_length)

        # decoder
        predict = self.decoder(trg, encoder_final_states)

        # for target padding mask
        mask = layers.sequence_mask(
            trg_length, maxlen=layers.shape(trg)[1], dtype=predict.dtype)
        return predict, mask


class BaseInferModel(BaseModel):
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
        super(BaseInferModel, self).__init__(**args)
        # dynamic decoder for inference
        decoder = BeamSearchDecoder(
            self.decoder.stack_lstm.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)
        self.beam_search_decoder = DynamicDecode(
            decoder, max_step_num=max_out_len, is_test=True)

    def forward(self, src, src_length):
        # encoding
        encoder_output, encoder_final_states = self.encoder(src, src_length)
        # dynamic decoding with beam search
        rs, _ = self.beam_search_decoder(inits=encoder_final_states)
        return rs
