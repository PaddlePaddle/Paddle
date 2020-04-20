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
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers import BeamSearchDecoder

from hapi.text import RNNCell, RNN, DynamicDecode
from hapi.model import Model, Loss


class ConvBNPool(fluid.dygraph.Layer):
    def __init__(self,
                 in_ch,
                 out_ch,
                 act="relu",
                 is_test=False,
                 pool=True,
                 use_cudnn=True):
        super(ConvBNPool, self).__init__()
        self.pool = pool

        filter_size = 3
        std = (2.0 / (filter_size**2 * in_ch))**0.5
        param_0 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, std))

        std = (2.0 / (filter_size**2 * out_ch))**0.5
        param_1 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, std))

        self.conv0 = fluid.dygraph.Conv2D(
            in_ch,
            out_ch,
            3,
            padding=1,
            param_attr=param_0,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn0 = fluid.dygraph.BatchNorm(out_ch, act=act)
        self.conv1 = fluid.dygraph.Conv2D(
            out_ch,
            out_ch,
            filter_size=3,
            padding=1,
            param_attr=param_1,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn1 = fluid.dygraph.BatchNorm(out_ch, act=act)

        if self.pool:
            self.pool = fluid.dygraph.Pool2D(
                pool_size=2,
                pool_type='max',
                pool_stride=2,
                use_cudnn=use_cudnn,
                ceil_mode=True)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.bn0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.pool:
            out = self.pool(out)
        return out


class CNN(fluid.dygraph.Layer):
    def __init__(self, in_ch=1, is_test=False):
        super(CNN, self).__init__()
        self.conv_bn1 = ConvBNPool(in_ch, 16)
        self.conv_bn2 = ConvBNPool(16, 32)
        self.conv_bn3 = ConvBNPool(32, 64)
        self.conv_bn4 = ConvBNPool(64, 128, pool=False)

    def forward(self, inputs):
        conv = self.conv_bn1(inputs)
        conv = self.conv_bn2(conv)
        conv = self.conv_bn3(conv)
        conv = self.conv_bn4(conv)
        return conv


class GRUCell(RNNCell):
    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 origin_mode=False):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.fc_layer = fluid.dygraph.Linear(
            input_size,
            hidden_size * 3,
            param_attr=param_attr,
            bias_attr=False)

        self.gru_unit = fluid.dygraph.GRUUnit(
            hidden_size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)

    def forward(self, inputs, states):
        # step_outputs, new_states = cell(step_inputs, states)
        # for GRUCell, `step_outputs` and `new_states` both are hidden
        x = self.fc_layer(inputs)
        hidden, _, _ = self.gru_unit(x, states)
        return hidden, hidden

    @property
    def state_shape(self):
        return [self.hidden_size]


class Encoder(fluid.dygraph.Layer):
    def __init__(
            self,
            in_channel=1,
            rnn_hidden_size=200,
            decoder_size=128,
            is_test=False, ):
        super(Encoder, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size

        self.backbone = CNN(in_ch=in_channel, is_test=is_test)

        para_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, 0.02))
        bias_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)
        self.gru_fwd = RNN(cell=GRUCell(
            input_size=128 * 6,
            hidden_size=rnn_hidden_size,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu'),
                           is_reverse=False,
                           time_major=False)
        self.gru_bwd = RNN(cell=GRUCell(
            input_size=128 * 6,
            hidden_size=rnn_hidden_size,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu'),
                           is_reverse=True,
                           time_major=False)
        self.encoded_proj_fc = fluid.dygraph.Linear(
            rnn_hidden_size * 2, decoder_size, bias_attr=False)

    def forward(self, inputs):
        conv_features = self.backbone(inputs)
        conv_features = fluid.layers.transpose(
            conv_features, perm=[0, 3, 1, 2])

        n, w, c, h = conv_features.shape
        seq_feature = fluid.layers.reshape(conv_features, [0, -1, c * h])

        gru_fwd, _ = self.gru_fwd(seq_feature)
        gru_bwd, _ = self.gru_bwd(seq_feature)

        encoded_vector = fluid.layers.concat(input=[gru_fwd, gru_bwd], axis=2)
        encoded_proj = self.encoded_proj_fc(encoded_vector)
        return gru_bwd, encoded_vector, encoded_proj


class Attention(fluid.dygraph.Layer):
    """
    Neural Machine Translation by Jointly Learning to Align and Translate.
    https://arxiv.org/abs/1409.0473
    """

    def __init__(self, decoder_size):
        super(Attention, self).__init__()
        self.fc1 = fluid.dygraph.Linear(
            decoder_size, decoder_size, bias_attr=False)
        self.fc2 = fluid.dygraph.Linear(decoder_size, 1, bias_attr=False)

    def forward(self, encoder_vec, encoder_proj, decoder_state):
        # alignment model, single-layer multilayer perceptron
        decoder_state = self.fc1(decoder_state)
        decoder_state = fluid.layers.unsqueeze(decoder_state, [1])

        e = fluid.layers.elementwise_add(encoder_proj, decoder_state)
        e = fluid.layers.tanh(e)

        att_scores = self.fc2(e)
        att_scores = fluid.layers.squeeze(att_scores, [2])
        att_scores = fluid.layers.softmax(att_scores)

        context = fluid.layers.elementwise_mul(
            x=encoder_vec, y=att_scores, axis=0)
        context = fluid.layers.reduce_sum(context, dim=1)
        return context


class DecoderCell(RNNCell):
    def __init__(self, encoder_size=200, decoder_size=128):
        super(DecoderCell, self).__init__()
        self.attention = Attention(decoder_size)
        self.gru_cell = GRUCell(
            input_size=encoder_size * 2 + decoder_size,
            hidden_size=decoder_size)

    def forward(self, current_word, states, encoder_vec, encoder_proj):
        context = self.attention(encoder_vec, encoder_proj, states)
        decoder_inputs = fluid.layers.concat([current_word, context], axis=1)
        hidden, _ = self.gru_cell(decoder_inputs, states)
        return hidden, hidden


class Decoder(fluid.dygraph.Layer):
    def __init__(self, num_classes, emb_dim, encoder_size, decoder_size):
        super(Decoder, self).__init__()
        self.decoder_attention = RNN(DecoderCell(encoder_size, decoder_size))
        self.fc = fluid.dygraph.Linear(
            decoder_size, num_classes + 2, act='softmax')

    def forward(self, target, initial_states, encoder_vec, encoder_proj):
        out, _ = self.decoder_attention(
            target,
            initial_states=initial_states,
            encoder_vec=encoder_vec,
            encoder_proj=encoder_proj)
        pred = self.fc(out)
        return pred


class Seq2SeqAttModel(Model):
    def __init__(
            self,
            in_channle=1,
            encoder_size=200,
            decoder_size=128,
            emb_dim=128,
            num_classes=None, ):
        super(Seq2SeqAttModel, self).__init__()
        self.encoder = Encoder(in_channle, encoder_size, decoder_size)
        self.fc = fluid.dygraph.Linear(
            input_dim=encoder_size,
            output_dim=decoder_size,
            bias_attr=False,
            act='relu')
        self.embedding = fluid.dygraph.Embedding(
            [num_classes + 2, emb_dim], dtype='float32')
        self.decoder = Decoder(num_classes, emb_dim, encoder_size,
                               decoder_size)

    def forward(self, inputs, target):
        gru_backward, encoded_vector, encoded_proj = self.encoder(inputs)
        decoder_boot = self.fc(gru_backward[:, 0])
        trg_embedding = self.embedding(target)
        prediction = self.decoder(trg_embedding, decoder_boot, encoded_vector,
                                  encoded_proj)
        return prediction


class Seq2SeqAttInferModel(Seq2SeqAttModel):
    def __init__(
            self,
            in_channle=1,
            encoder_size=200,
            decoder_size=128,
            emb_dim=128,
            num_classes=None,
            beam_size=0,
            bos_id=0,
            eos_id=1,
            max_out_len=20, ):
        super(Seq2SeqAttInferModel, self).__init__(
            in_channle, encoder_size, decoder_size, emb_dim, num_classes)
        self.beam_size = beam_size
        # dynamic decoder for inference
        decoder = BeamSearchDecoder(
            self.decoder.decoder_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.embedding,
            output_fn=self.decoder.fc)
        self.infer_decoder = DynamicDecode(
            decoder, max_step_num=max_out_len, is_test=True)

    def forward(self, inputs, *args):
        gru_backward, encoded_vector, encoded_proj = self.encoder(inputs)
        decoder_boot = self.fc(gru_backward[:, 0])

        if self.beam_size:
            # Tile the batch dimension with beam_size
            encoded_vector = BeamSearchDecoder.tile_beam_merge_with_batch(
                encoded_vector, self.beam_size)
            encoded_proj = BeamSearchDecoder.tile_beam_merge_with_batch(
                encoded_proj, self.beam_size)
        # dynamic decoding with beam search
        rs, _ = self.infer_decoder(
            inits=decoder_boot,
            encoder_vec=encoded_vector,
            encoder_proj=encoded_proj)
        return rs


class WeightCrossEntropy(Loss):
    def __init__(self):
        super(WeightCrossEntropy, self).__init__(average=False)

    def forward(self, outputs, labels):
        predict, (label, mask) = outputs[0], labels
        loss = layers.cross_entropy(predict, label=label)
        loss = layers.elementwise_mul(loss, mask, axis=0)
        loss = layers.reduce_sum(loss)
        return loss
