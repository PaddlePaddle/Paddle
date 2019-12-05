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

import os
import contextlib
import unittest
import numpy as np
import six
import pickle

import paddle
import paddle.dataset.wmt16 as wmt16
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph import Embedding, LayerNorm, FC, to_variable, Layer, guard
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay

from test_dist_base import runtime_main, TestParallelDyGraphRunnerBase


class TrainTaskConfig(object):
    """
    TrainTaskConfig
    """
    # the epoch number to train.
    pass_num = 20
    # the number of sequences contained in a mini-batch.
    # deprecated, set batch_size in args.
    batch_size = 32
    # the hyper parameters for Adam optimizer.
    # This static learning_rate will be multiplied to the LearningRateScheduler
    # derived learning rate the to get the final learning rate.
    learning_rate = 2.0
    beta1 = 0.9
    beta2 = 0.997
    eps = 1e-9
    # the parameters for learning rate scheduling.
    warmup_steps = 8000
    # the weight used to mix up the ground-truth distribution and the fixed
    # uniform distribution in label smoothing when training.
    # Set this as zero if label smoothing is not wanted.
    label_smooth_eps = 0.1


class ModelHyperParams(object):
    """
    ModelHyperParams
    """
    # These following five vocabularies related configurations will be set
    # automatically according to the passed vocabulary path and special tokens.
    # size of source word dictionary.
    src_vocab_size = 10000
    # size of target word dictionay
    trg_vocab_size = 10000
    # index for <bos> token
    bos_idx = 0
    # index for <eos> token
    eos_idx = 1
    # index for <unk> token
    unk_idx = 2

    # max length of sequences deciding the size of position encoding table.
    max_length = 50
    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.
    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 2048
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    n_layer = 6
    # dropout rates of different modules.
    prepostprocess_dropout = 0.1
    attention_dropout = 0.1
    relu_dropout = 0.1
    # to process before each sub-layer
    preprocess_cmd = "n"  # layer normalization
    # to process after each sub-layer
    postprocess_cmd = "da"  # dropout + residual connection
    # the flag indicating whether to share embedding and softmax weights.
    # vocabularies in source and target should be same for weight sharing.
    weight_sharing = False


# The placeholder for batch_size in compile time. Must be -1 currently to be
# consistent with some ops' infer-shape output in compile time, such as the
# sequence_expand op used in beamsearch decoder.
batch_size = -1
# The placeholder for squence length in compile time.
seq_len = ModelHyperParams.max_length
# Here list the data shapes and data types of all inputs.
# The shapes here act as placeholder and are set to pass the infer-shape in
# compile time.
input_descs = {
    # The actual data shape of src_word is:
    # [batch_size, max_src_len_in_batch, 1]
    "src_word": [(batch_size, seq_len, 1), "int64", 2],
    # The actual data shape of src_pos is:
    # [batch_size, max_src_len_in_batch, 1]
    "src_pos": [(batch_size, seq_len, 1), "int64"],
    # This input is used to remove attention weights on paddings in the
    # encoder.
    # The actual data shape of src_slf_attn_bias is:
    # [batch_size, n_head, max_src_len_in_batch, max_src_len_in_batch]
    "src_slf_attn_bias":
    [(batch_size, ModelHyperParams.n_head, seq_len, seq_len), "float32"],
    # The actual data shape of trg_word is:
    # [batch_size, max_trg_len_in_batch, 1]
    "trg_word": [(batch_size, seq_len, 1), "int64",
                 2],  # lod_level is only used in fast decoder.
    # The actual data shape of trg_pos is:
    # [batch_size, max_trg_len_in_batch, 1]
    "trg_pos": [(batch_size, seq_len, 1), "int64"],
    # This input is used to remove attention weights on paddings and
    # subsequent words in the decoder.
    # The actual data shape of trg_slf_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_trg_len_in_batch]
    "trg_slf_attn_bias":
    [(batch_size, ModelHyperParams.n_head, seq_len, seq_len), "float32"],
    # This input is used to remove attention weights on paddings of the source
    # input in the encoder-decoder attention.
    # The actual data shape of trg_src_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_src_len_in_batch]
    "trg_src_attn_bias":
    [(batch_size, ModelHyperParams.n_head, seq_len, seq_len), "float32"],
    # This input is used in independent decoder program for inference.
    # The actual data shape of enc_output is:
    # [batch_size, max_src_len_in_batch, d_model]
    "enc_output": [(batch_size, seq_len, ModelHyperParams.d_model), "float32"],
    # The actual data shape of label_word is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_word": [(batch_size * seq_len, 1), "int64"],
    # This input is used to mask out the loss of paddding tokens.
    # The actual data shape of label_weight is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_weight": [(batch_size * seq_len, 1), "float32"],
    # This input is used in beam-search decoder.
    "init_score": [(batch_size, 1), "float32", 2],
    # This input is used in beam-search decoder for the first gather
    # (cell states updation)
    "init_idx": [(batch_size, ), "int32"],
}

# Names of word embedding table which might be reused for weight sharing.
word_emb_param_names = (
    "src_word_emb_table",
    "trg_word_emb_table", )
# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )
# separated inputs for different usages.
encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias", )
decoder_data_input_fields = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "enc_output", )
label_data_input_fields = (
    "lbl_word",
    "lbl_weight", )
# In fast decoder, trg_pos (only containing the current time step) is generated
# by ops and trg_slf_attn_bias is not needed.
fast_decoder_data_input_fields = (
    "trg_word",
    # "init_score",
    # "init_idx",
    "trg_src_attn_bias", )


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


class NoamDecay(LearningRateDecay):
    """
    learning rate scheduler
    """

    def __init__(self,
                 d_model,
                 warmup_steps,
                 static_lr=2.0,
                 begin=1,
                 step=1,
                 dtype='float32'):
        super(NoamDecay, self).__init__(begin, step, dtype)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.static_lr = static_lr

    def step(self):
        a = self.create_lr_var(self.step_num**-0.5)
        b = self.create_lr_var((self.warmup_steps**-1.5) * self.step_num)
        lr_value = (self.d_model**-0.5) * layers.elementwise_min(
            a, b) * self.static_lr
        return lr_value


class PrePostProcessLayer(Layer):
    """
    PrePostProcessLayer
    """

    def __init__(self, name_scope, process_cmd, shape_len=None):
        super(PrePostProcessLayer, self).__init__(name_scope)
        for cmd in process_cmd:
            if cmd == "n":
                self._layer_norm = LayerNorm(
                    name_scope=self.full_name(),
                    begin_norm_axis=shape_len - 1,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(1.)),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(0.)))

    def forward(self, prev_out, out, process_cmd, dropout_rate=0.):
        """
        forward
        :param prev_out:
        :param out:
        :param process_cmd:
        :param dropout_rate:
        :return:
        """
        for cmd in process_cmd:
            if cmd == "a":  # add residual connection
                out = out + prev_out if prev_out else out
            elif cmd == "n":  # add layer normalization
                out = self._layer_norm(out)
            elif cmd == "d":  # add dropout
                if dropout_rate:
                    out = layers.dropout(
                        out, dropout_prob=dropout_rate, is_test=False)
        return out


class PositionwiseFeedForwardLayer(Layer):
    """
    PositionwiseFeedForwardLayer
    """

    def __init__(self, name_scope, d_inner_hid, d_hid, dropout_rate):
        super(PositionwiseFeedForwardLayer, self).__init__(name_scope)
        self._i2h = FC(name_scope=self.full_name(),
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
        self._h2o = FC(name_scope=self.full_name(),
                       size=d_hid,
                       num_flatten_dims=2)
        self._dropout_rate = dropout_rate

    def forward(self, x):
        """
        forward
        :param x:
        :return:
        """
        hidden = self._i2h(x)
        if self._dropout_rate:
            hidden = layers.dropout(
                hidden, dropout_prob=self._dropout_rate, is_test=False)
        out = self._h2o(hidden)
        return out


class MultiHeadAttentionLayer(Layer):
    """
    MultiHeadAttentionLayer
    """

    def __init__(self,
                 name_scope,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 cache=None,
                 gather_idx=None,
                 static_kv=False):
        super(MultiHeadAttentionLayer, self).__init__(name_scope)
        self._n_head = n_head
        self._d_key = d_key
        self._d_value = d_value
        self._d_model = d_model
        self._dropout_rate = dropout_rate
        self._q_fc = FC(name_scope=self.full_name(),
                        size=d_key * n_head,
                        bias_attr=False,
                        num_flatten_dims=2)
        self._k_fc = FC(name_scope=self.full_name(),
                        size=d_key * n_head,
                        bias_attr=False,
                        num_flatten_dims=2)
        self._v_fc = FC(name_scope=self.full_name(),
                        size=d_value * n_head,
                        bias_attr=False,
                        num_flatten_dims=2)
        self._proj_fc = FC(name_scope=self.full_name(),
                           size=self._d_model,
                           bias_attr=False,
                           num_flatten_dims=2)

    def forward(self,
                queries,
                keys,
                values,
                attn_bias,
                cache=None,
                gather_idx=None):
        """
        forward
        :param queries:
        :param keys:
        :param values:
        :param attn_bias:
        :return:
        """
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self._q_fc(queries)
        k = self._k_fc(keys)
        v = self._v_fc(values)

        # split head
        reshaped_q = layers.reshape(
            x=q, shape=[0, 0, self._n_head, self._d_key], inplace=False)
        transpose_q = layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
        reshaped_k = layers.reshape(
            x=k, shape=[0, 0, self._n_head, self._d_key], inplace=False)
        transpose_k = layers.transpose(x=reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = layers.reshape(
            x=v, shape=[0, 0, self._n_head, self._d_value], inplace=False)
        transpose_v = layers.transpose(x=reshaped_v, perm=[0, 2, 1, 3])

        if cache is not None:
            cache_k, cache_v = cache["k"], cache["v"]
            transpose_k = layers.concat([cache_k, transpose_k], axis=2)
            transpose_v = layers.concat([cache_v, transpose_v], axis=2)
            cache["k"], cache["v"] = transpose_k, transpose_v

        # scale dot product attention
        product = layers.matmul(
            x=transpose_q,
            y=transpose_k,
            transpose_y=True,
            alpha=self._d_model**-0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if self._dropout_rate:
            weights_droped = layers.dropout(
                weights, dropout_prob=self._dropout_rate, is_test=False)
            out = layers.matmul(weights_droped, transpose_v)
        else:
            out = layers.matmul(weights, transpose_v)

        # combine heads
        if len(out.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        trans_x = layers.transpose(out, perm=[0, 2, 1, 3])
        final_out = layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=False)

        # fc to output
        proj_out = self._proj_fc(final_out)
        return proj_out


class EncoderSubLayer(Layer):
    """
    EncoderSubLayer
    """

    def __init__(self,
                 name_scope,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(EncoderSubLayer, self).__init__(name_scope)
        self._preprocess_cmd = preprocess_cmd
        self._postprocess_cmd = postprocess_cmd
        self._prepostprocess_dropout = prepostprocess_dropout

        self._preprocess_layer = PrePostProcessLayer(self.full_name(),
                                                     self._preprocess_cmd, 3)
        self._multihead_attention_layer = MultiHeadAttentionLayer(
            self.full_name(), d_key, d_value, d_model, n_head,
            attention_dropout)
        self._postprocess_layer = PrePostProcessLayer(
            self.full_name(), self._postprocess_cmd, None)
        self._preprocess_layer2 = PrePostProcessLayer(self.full_name(),
                                                      self._preprocess_cmd, 3)
        self._positionwise_feed_forward = PositionwiseFeedForwardLayer(
            self.full_name(), d_inner_hid, d_model, relu_dropout)
        self._postprocess_layer2 = PrePostProcessLayer(
            self.full_name(), self._postprocess_cmd, None)

    def forward(self, enc_input, attn_bias):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        pre_process_multihead = self._preprocess_layer(
            None, enc_input, self._preprocess_cmd, self._prepostprocess_dropout)
        attn_output = self._multihead_attention_layer(pre_process_multihead,
                                                      None, None, attn_bias)
        attn_output = self._postprocess_layer(enc_input, attn_output,
                                              self._postprocess_cmd,
                                              self._prepostprocess_dropout)
        pre_process2_output = self._preprocess_layer2(
            None, attn_output, self._preprocess_cmd,
            self._prepostprocess_dropout)
        ffd_output = self._positionwise_feed_forward(pre_process2_output)
        return self._postprocess_layer2(attn_output, ffd_output,
                                        self._postprocess_cmd,
                                        self._prepostprocess_dropout)


class EncoderLayer(Layer):
    """
    encoder
    """

    def __init__(self,
                 name_scope,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da"):

        super(EncoderLayer, self).__init__(name_scope)
        self._preprocess_cmd = preprocess_cmd
        self._encoder_sublayers = list()
        self._prepostprocess_dropout = prepostprocess_dropout
        self._n_layer = n_layer
        self._preprocess_layer = PrePostProcessLayer(self.full_name(),
                                                     self._preprocess_cmd, 3)
        for i in range(n_layer):
            self._encoder_sublayers.append(
                self.add_sublayer(
                    'esl_%d' % i,
                    EncoderSubLayer(
                        self.full_name(), n_head, d_key, d_value, d_model,
                        d_inner_hid, prepostprocess_dropout, attention_dropout,
                        relu_dropout, preprocess_cmd, postprocess_cmd)))

    def forward(self, enc_input, attn_bias):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        for i in range(self._n_layer):
            enc_output = self._encoder_sublayers[i](enc_input, attn_bias)
            enc_input = enc_output

        return self._preprocess_layer(None, enc_output, self._preprocess_cmd,
                                      self._prepostprocess_dropout)


class PrepareEncoderDecoderLayer(Layer):
    """
    PrepareEncoderDecoderLayer
    """

    def __init__(self,
                 name_scope,
                 src_vocab_size,
                 src_emb_dim,
                 src_max_len,
                 dropout_rate,
                 word_emb_param_name=None,
                 pos_enc_param_name=None):
        super(PrepareEncoderDecoderLayer, self).__init__(name_scope)
        self._src_max_len = src_max_len
        self._src_emb_dim = src_emb_dim
        self._src_vocab_size = src_vocab_size
        self._dropout_rate = dropout_rate
        self._input_emb = Embedding(
            name_scope=self.full_name(),
            size=[src_vocab_size, src_emb_dim],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                name=word_emb_param_name,
                initializer=fluid.initializer.Normal(0., src_emb_dim**-0.5)))

        pos_inp = position_encoding_init(src_max_len, src_emb_dim)
        self._pos_emb = Embedding(
            name_scope=self.full_name(),
            size=[self._src_max_len, src_emb_dim],
            param_attr=fluid.ParamAttr(
                name=pos_enc_param_name,
                initializer=fluid.initializer.NumpyArrayInitializer(pos_inp),
                trainable=False))

        # use in dygraph_mode to fit different length batch
        # self._pos_emb._w = to_variable(
        #     position_encoding_init(self._src_max_len, self._src_emb_dim))

    def forward(self, src_word, src_pos):
        """
        forward
        :param src_word:
        :param src_pos:
        :return:
        """
        # print("here")
        # print(self._input_emb._w._numpy().shape)
        src_word_emb = self._input_emb(src_word)

        src_word_emb = layers.scale(
            x=src_word_emb, scale=self._src_emb_dim**0.5)
        # # TODO change this to fit dynamic length input
        src_pos_emb = self._pos_emb(src_pos)
        src_pos_emb.stop_gradient = True
        enc_input = src_word_emb + src_pos_emb
        return layers.dropout(
            enc_input, dropout_prob=self._dropout_rate,
            is_test=False) if self._dropout_rate else enc_input


class WrapEncoderLayer(Layer):
    """
    encoderlayer
    """

    def __init__(self, name_cope, src_vocab_size, max_length, n_layer, n_head,
                 d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout,
                 attention_dropout, relu_dropout, preprocess_cmd,
                 postprocess_cmd, weight_sharing):
        """
        The wrapper assembles together all needed layers for the encoder.
        """
        super(WrapEncoderLayer, self).__init__(name_cope)

        self._prepare_encoder_layer = PrepareEncoderDecoderLayer(
            self.full_name(),
            src_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            word_emb_param_name=word_emb_param_names[0],
            pos_enc_param_name=pos_enc_param_names[0])
        self._encoder = EncoderLayer(
            self.full_name(), n_layer, n_head, d_key, d_value, d_model,
            d_inner_hid, prepostprocess_dropout, attention_dropout,
            relu_dropout, preprocess_cmd, postprocess_cmd)

    def forward(self, enc_inputs):
        """forward"""
        src_word, src_pos, src_slf_attn_bias = enc_inputs
        enc_input = self._prepare_encoder_layer(src_word, src_pos)
        enc_output = self._encoder(enc_input, src_slf_attn_bias)
        return enc_output


class DecoderSubLayer(Layer):
    """
    decoder
    """

    def __init__(self, name_scope, n_head, d_key, d_value, d_model, d_inner_hid,
                 prepostprocess_dropout, attention_dropout, relu_dropout,
                 preprocess_cmd, postprocess_cmd):
        super(DecoderSubLayer, self).__init__(name_scope)
        self._postprocess_cmd = postprocess_cmd
        self._preprocess_cmd = preprocess_cmd
        self._prepostprcess_dropout = prepostprocess_dropout
        self._pre_process_layer = PrePostProcessLayer(self.full_name(),
                                                      preprocess_cmd, 3)
        self._multihead_attention_layer = MultiHeadAttentionLayer(
            self.full_name(), d_key, d_value, d_model, n_head,
            attention_dropout)
        self._post_process_layer = PrePostProcessLayer(self.full_name(),
                                                       postprocess_cmd, None)
        self._pre_process_layer2 = PrePostProcessLayer(self.full_name(),
                                                       preprocess_cmd, 3)
        self._multihead_attention_layer2 = MultiHeadAttentionLayer(
            self.full_name(), d_key, d_value, d_model, n_head,
            attention_dropout)
        self._post_process_layer2 = PrePostProcessLayer(self.full_name(),
                                                        postprocess_cmd, None)
        self._pre_process_layer3 = PrePostProcessLayer(self.full_name(),
                                                       preprocess_cmd, 3)
        self._positionwise_feed_forward_layer = PositionwiseFeedForwardLayer(
            self.full_name(), d_inner_hid, d_model, relu_dropout)
        self._post_process_layer3 = PrePostProcessLayer(self.full_name(),
                                                        postprocess_cmd, None)

    def forward(self,
                dec_input,
                enc_output,
                slf_attn_bias,
                dec_enc_attn_bias,
                cache=None,
                gather_idx=None):
        """
        forward
        :param dec_input:
        :param enc_output:
        :param slf_attn_bias:
        :param dec_enc_attn_bias:
        :return:
        """
        pre_process_rlt = self._pre_process_layer(
            None, dec_input, self._preprocess_cmd, self._prepostprcess_dropout)
        slf_attn_output = self._multihead_attention_layer(
            pre_process_rlt, None, None, slf_attn_bias, cache, gather_idx)
        slf_attn_output_pp = self._post_process_layer(
            dec_input, slf_attn_output, self._postprocess_cmd,
            self._prepostprcess_dropout)
        pre_process_rlt2 = self._pre_process_layer2(None, slf_attn_output_pp,
                                                    self._preprocess_cmd,
                                                    self._prepostprcess_dropout)
        enc_attn_output_pp = self._multihead_attention_layer2(
            pre_process_rlt2, enc_output, enc_output, dec_enc_attn_bias)
        enc_attn_output = self._post_process_layer2(
            slf_attn_output_pp, enc_attn_output_pp, self._postprocess_cmd,
            self._prepostprcess_dropout)
        pre_process_rlt3 = self._pre_process_layer3(None, enc_attn_output,
                                                    self._preprocess_cmd,
                                                    self._prepostprcess_dropout)
        ffd_output = self._positionwise_feed_forward_layer(pre_process_rlt3)
        dec_output = self._post_process_layer3(enc_attn_output, ffd_output,
                                               self._postprocess_cmd,
                                               self._prepostprcess_dropout)
        return dec_output


class DecoderLayer(Layer):
    """
    decoder
    """

    def __init__(self, name_scope, n_layer, n_head, d_key, d_value, d_model,
                 d_inner_hid, prepostprocess_dropout, attention_dropout,
                 relu_dropout, preprocess_cmd, postprocess_cmd):
        super(DecoderLayer, self).__init__(name_scope)
        self._pre_process_layer = PrePostProcessLayer(self.full_name(),
                                                      preprocess_cmd, 3)
        self._decoder_sub_layers = list()
        self._n_layer = n_layer
        self._preprocess_cmd = preprocess_cmd
        self._prepostprocess_dropout = prepostprocess_dropout
        for i in range(n_layer):
            self._decoder_sub_layers.append(
                self.add_sublayer(
                    'dsl_%d' % i,
                    DecoderSubLayer(
                        self.full_name(), n_head, d_key, d_value, d_model,
                        d_inner_hid, prepostprocess_dropout, attention_dropout,
                        relu_dropout, preprocess_cmd, postprocess_cmd)))

    def forward(self,
                dec_input,
                enc_output,
                dec_slf_attn_bias,
                dec_enc_attn_bias,
                caches=None,
                gather_idx=None):
        """
        forward
        :param dec_input:
        :param enc_output:
        :param dec_slf_attn_bias:
        :param dec_enc_attn_bias:
        :return:
        """
        for i in range(self._n_layer):
            tmp_dec_output = self._decoder_sub_layers[i](
                dec_input, enc_output, dec_slf_attn_bias, dec_enc_attn_bias,
                None if caches is None else caches[i], gather_idx)
            dec_input = tmp_dec_output

        dec_output = self._pre_process_layer(None, tmp_dec_output,
                                             self._preprocess_cmd,
                                             self._prepostprocess_dropout)
        return dec_output


class WrapDecoderLayer(Layer):
    """
    decoder
    """

    def __init__(self,
                 name_scope,
                 trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 gather_idx=None):
        """
        The wrapper assembles together all needed layers for the encoder.
        """
        super(WrapDecoderLayer, self).__init__(name_scope)

        self._prepare_decoder_layer = PrepareEncoderDecoderLayer(
            self.full_name(),
            trg_vocab_size,
            d_model,
            max_length,
            prepostprocess_dropout,
            word_emb_param_name=word_emb_param_names[1],
            pos_enc_param_name=pos_enc_param_names[1])
        self._decoder_layer = DecoderLayer(
            self.full_name(), n_layer, n_head, d_key, d_value, d_model,
            d_inner_hid, prepostprocess_dropout, attention_dropout,
            relu_dropout, preprocess_cmd, postprocess_cmd)
        self._weight_sharing = weight_sharing
        if not weight_sharing:
            self._fc = FC(self.full_name(),
                          size=trg_vocab_size,
                          bias_attr=False)

    def forward(self, dec_inputs, enc_output, caches=None, gather_idx=None):
        """
        forward
        :param dec_inputs:
        :param enc_output:
        :return:
        """
        trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias = dec_inputs
        dec_input = self._prepare_decoder_layer(trg_word, trg_pos)
        dec_output = self._decoder_layer(dec_input, enc_output,
                                         trg_slf_attn_bias, trg_src_attn_bias,
                                         caches, gather_idx)

        dec_output_reshape = layers.reshape(
            dec_output, shape=[-1, dec_output.shape[-1]], inplace=False)

        if self._weight_sharing:
            predict = layers.matmul(
                x=dec_output_reshape,
                y=self._prepare_decoder_layer._input_emb._w,
                transpose_y=True)
        else:
            predict = self._fc(dec_output_reshape)

        if dec_inputs is None:
            # Return probs for independent decoder program.
            predict_out = layers.softmax(predict)
            return predict_out
        return predict


class TransFormer(Layer):
    """
    model
    """

    def __init__(self,
                 name_scope,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 label_smooth_eps=0.0):
        super(TransFormer, self).__init__(name_scope)
        self._label_smooth_eps = label_smooth_eps
        self._trg_vocab_size = trg_vocab_size
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
        self._wrap_encoder_layer = WrapEncoderLayer(
            self.full_name(), src_vocab_size, max_length, n_layer, n_head,
            d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)
        self._wrap_decoder_layer = WrapDecoderLayer(
            self.full_name(), trg_vocab_size, max_length, n_layer, n_head,
            d_key, d_value, d_model, d_inner_hid, prepostprocess_dropout,
            attention_dropout, relu_dropout, preprocess_cmd, postprocess_cmd,
            weight_sharing)

        if weight_sharing:
            self._wrap_decoder_layer._prepare_decoder_layer._input_emb._w = self._wrap_encoder_layer._prepare_encoder_layer._input_emb._w

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value

    def forward(self, enc_inputs, dec_inputs, label, weights):
        """
        forward
        :param enc_inputs:
        :param dec_inputs:
        :param label:
        :param weights:
        :return:
        """
        enc_output = self._wrap_encoder_layer(enc_inputs)
        predict = self._wrap_decoder_layer(dec_inputs, enc_output)
        if self._label_smooth_eps:
            label_out = layers.label_smooth(
                label=layers.one_hot(
                    input=label, depth=self._trg_vocab_size),
                epsilon=self._label_smooth_eps)

        cost = layers.softmax_with_cross_entropy(
            logits=predict,
            label=label_out,
            soft_label=True if self._label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = layers.reduce_sum(weighted_cost)
        token_num = layers.reduce_sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, predict, token_num

    def beam_search(self,
                    enc_inputs,
                    dec_inputs,
                    bos_id=0,
                    eos_id=1,
                    beam_size=4,
                    max_len=30,
                    alpha=0.6):
        """
        Beam search with the alive and finished two queues, both have a beam size
        capicity separately. It includes `grow_topk` `grow_alive` `grow_finish` as
        steps. 
        
        1. `grow_topk` selects the top `2*beam_size` candidates to avoid all getting
        EOS.

        2. `grow_alive` selects the top `beam_size` non-EOS candidates as the inputs
        of next decoding step.

        3. `grow_finish` compares the already finished candidates in the finished queue
        and newly added finished candidates from `grow_topk`, and selects the top
        `beam_size` finished candidates.
        """

        def expand_to_beam_size(tensor, beam_size):
            tensor = layers.reshape(tensor,
                                    [tensor.shape[0], 1] + tensor.shape[1:])
            tile_dims = [1] * len(tensor.shape)
            tile_dims[1] = beam_size
            return layers.expand(tensor, tile_dims)

        def merge_beam_dim(tensor):
            return layers.reshape(tensor, [-1] + tensor.shape[2:])

        # run encoder
        enc_output = self._wrap_encoder_layer(enc_inputs)

        # constant number
        inf = float(1. * 1e7)
        batch_size = enc_output.shape[0]

        ### initialize states of beam search ###
        ## init for the alive ##
        initial_ids, trg_src_attn_bias = dec_inputs  # (batch_size, 1)
        initial_log_probs = to_variable(
            np.array(
                [[0.] + [-inf] * (beam_size - 1)], dtype="float32"))
        alive_log_probs = layers.expand(initial_log_probs, [batch_size, 1])
        alive_seq = to_variable(
            np.tile(
                np.array(
                    [[[bos_id]]], dtype="int64"), (batch_size, beam_size, 1)))

        ## init for the finished ##
        finished_scores = to_variable(
            np.array(
                [[-inf] * beam_size], dtype="float32"))
        finished_scores = layers.expand(finished_scores, [batch_size, 1])
        finished_seq = to_variable(
            np.tile(
                np.array(
                    [[[bos_id]]], dtype="int64"), (batch_size, beam_size, 1)))
        finished_flags = layers.zeros_like(finished_scores)

        ### initialize inputs and states of transformer decoder ###
        ## init inputs for decoder, shaped `[batch_size*beam_size, ...]`
        trg_word = layers.reshape(alive_seq[:, :, -1],
                                  [batch_size * beam_size, 1, 1])
        trg_pos = layers.zeros_like(trg_word)
        trg_src_attn_bias = merge_beam_dim(
            expand_to_beam_size(trg_src_attn_bias, beam_size))
        enc_output = merge_beam_dim(expand_to_beam_size(enc_output, beam_size))
        ## init states (caches) for transformer, need to be updated according to selected beam
        caches = [{
            "k": layers.fill_constant(
                shape=[batch_size * beam_size, self.n_head, 0, self.d_key],
                dtype=enc_output.dtype,
                value=0),
            "v": layers.fill_constant(
                shape=[batch_size * beam_size, self.n_head, 0, self.d_value],
                dtype=enc_output.dtype,
                value=0),
        } for i in range(self.n_layer)]

        def update_states(caches, beam_idx, beam_size):
            for cache in caches:
                cache["k"] = gather_2d_by_gather(cache["k"], beam_idx,
                                                 beam_size, batch_size, False)
                cache["v"] = gather_2d_by_gather(cache["v"], beam_idx,
                                                 beam_size, batch_size, False)
            return caches

        def gather_2d_by_gather(tensor_nd,
                                beam_idx,
                                beam_size,
                                batch_size,
                                need_flat=True):
            batch_idx = layers.range(
                0, batch_size, 1, dtype="int64") * beam_size
            flat_tensor = merge_beam_dim(tensor_nd) if need_flat else tensor_nd
            idx = layers.reshape(
                layers.elementwise_add(beam_idx, batch_idx, 0), [-1])
            new_flat_tensor = layers.gather(flat_tensor, idx)
            new_tensor_nd = layers.reshape(
                new_flat_tensor,
                shape=[batch_size, beam_idx.shape[1]] +
                tensor_nd.shape[2:]) if need_flat else new_flat_tensor
            return new_tensor_nd

        def early_finish(alive_log_probs, finished_scores,
                         finished_in_finished):
            max_length_penalty = np.power(((5. + max_len) / 6.), alpha)
            # The best possible score of the most likely alive sequence
            lower_bound_alive_scores = alive_log_probs[:,
                                                       0] / max_length_penalty

            # Now to compute the lowest score of a finished sequence in finished
            # If the sequence isn't finished, we multiply it's score by 0. since
            # scores are all -ve, taking the min will give us the score of the lowest
            # finished item.
            lowest_score_of_fininshed_in_finished = layers.reduce_min(
                finished_scores * finished_in_finished, 1)
            # If none of the sequences have finished, then the min will be 0 and
            # we have to replace it by -ve INF if it is. The score of any seq in alive
            # will be much higher than -ve INF and the termination condition will not
            # be met.
            lowest_score_of_fininshed_in_finished += (
                1. - layers.reduce_max(finished_in_finished, 1)) * -inf
            bound_is_met = layers.reduce_all(
                layers.greater_than(lowest_score_of_fininshed_in_finished,
                                    lower_bound_alive_scores))

            return bound_is_met

        def grow_topk(i, logits, alive_seq, alive_log_probs, states):
            logits = layers.reshape(logits, [batch_size, beam_size, -1])
            candidate_log_probs = layers.log(layers.softmax(logits, axis=2))
            log_probs = layers.elementwise_add(candidate_log_probs,
                                               alive_log_probs, 0)

            length_penalty = np.power(5.0 + (i + 1.0) / 6.0, alpha)
            curr_scores = log_probs / length_penalty
            flat_curr_scores = layers.reshape(curr_scores, [batch_size, -1])

            topk_scores, topk_ids = layers.topk(
                flat_curr_scores, k=beam_size * 2)

            topk_log_probs = topk_scores * length_penalty

            topk_beam_index = topk_ids // self._trg_vocab_size
            topk_ids = topk_ids % self._trg_vocab_size

            # use gather as gather_nd, TODO: use gather_nd
            topk_seq = gather_2d_by_gather(alive_seq, topk_beam_index,
                                           beam_size, batch_size)
            topk_seq = layers.concat(
                [topk_seq, layers.reshape(topk_ids, topk_ids.shape + [1])],
                axis=2)
            states = update_states(states, topk_beam_index, beam_size)
            eos = layers.fill_constant(
                shape=topk_ids.shape, dtype="int64", value=eos_id)
            topk_finished = layers.cast(layers.equal(topk_ids, eos), "float32")

            #topk_seq: [batch_size, 2*beam_size, i+1]
            #topk_log_probs, topk_scores, topk_finished: [batch_size, 2*beam_size]
            return topk_seq, topk_log_probs, topk_scores, topk_finished, states

        def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                       states):
            curr_scores += curr_finished * -inf
            _, topk_indexes = layers.topk(curr_scores, k=beam_size)
            alive_seq = gather_2d_by_gather(curr_seq, topk_indexes,
                                            beam_size * 2, batch_size)
            alive_log_probs = gather_2d_by_gather(curr_log_probs, topk_indexes,
                                                  beam_size * 2, batch_size)
            states = update_states(states, topk_indexes, beam_size * 2)

            return alive_seq, alive_log_probs, states

        def grow_finished(finished_seq, finished_scores, finished_flags,
                          curr_seq, curr_scores, curr_finished):
            # finished scores
            finished_seq = layers.concat(
                [
                    finished_seq, layers.fill_constant(
                        shape=[batch_size, beam_size, 1],
                        dtype="int64",
                        value=eos_id)
                ],
                axis=2)
            # Set the scores of the unfinished seq in curr_seq to large negative
            # values
            curr_scores += (1. - curr_finished) * -inf
            # concatenating the sequences and scores along beam axis
            curr_finished_seq = layers.concat([finished_seq, curr_seq], axis=1)
            curr_finished_scores = layers.concat(
                [finished_scores, curr_scores], axis=1)
            curr_finished_flags = layers.concat(
                [finished_flags, curr_finished], axis=1)
            _, topk_indexes = layers.topk(curr_finished_scores, k=beam_size)
            finished_seq = gather_2d_by_gather(curr_finished_seq, topk_indexes,
                                               beam_size * 3, batch_size)
            finished_scores = gather_2d_by_gather(
                curr_finished_scores, topk_indexes, beam_size * 3, batch_size)
            finished_flags = gather_2d_by_gather(
                curr_finished_flags, topk_indexes, beam_size * 3, batch_size)
            return finished_seq, finished_scores, finished_flags

        for i in range(max_len):
            logits = self._wrap_decoder_layer(
                (trg_word, trg_pos, None, trg_src_attn_bias), enc_output,
                caches)
            topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
                i, logits, alive_seq, alive_log_probs, caches)
            alive_seq, alive_log_probs, states = grow_alive(
                topk_seq, topk_scores, topk_log_probs, topk_finished, states)
            finished_seq, finished_scores, finished_flags = grow_finished(
                finished_seq, finished_scores, finished_flags, topk_seq,
                topk_scores, topk_finished)
            trg_word = layers.reshape(alive_seq[:, :, -1],
                                      [batch_size * beam_size, 1, 1])
            trg_pos = layers.fill_constant(
                shape=trg_word.shape, dtype="int64", value=i)
            if early_finish(alive_log_probs, finished_scores,
                            finished_flags).numpy():
                break

        return finished_seq, finished_scores


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array([[1.] * len(inst) + [0.] * (max_len - len(inst))
                                for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            list(range(0, len(inst))) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_train_input(insts, src_pad_idx, trg_pad_idx, n_head):
    """
    inputs for training
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_pos = trg_pos.reshape(-1, trg_max_len, 1)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    data_inputs = [
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
    ]

    var_inputs = []
    for i, field in enumerate(encoder_data_input_fields +
                              decoder_data_input_fields[:-1] +
                              label_data_input_fields):
        var_inputs.append(to_variable(data_inputs[i], name=field))

    enc_inputs = var_inputs[0:len(encoder_data_input_fields)]
    dec_inputs = var_inputs[len(encoder_data_input_fields):len(
        encoder_data_input_fields) + len(decoder_data_input_fields[:-1])]
    label = var_inputs[-2]
    weights = var_inputs[-1]

    return enc_inputs, dec_inputs, label, weights


naive_optimize = True


class TestTransformer(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = TransFormer(
            'transformer', ModelHyperParams.src_vocab_size,
            ModelHyperParams.trg_vocab_size, ModelHyperParams.max_length + 1,
            ModelHyperParams.n_layer, ModelHyperParams.n_head,
            ModelHyperParams.d_key, ModelHyperParams.d_value,
            ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
            ModelHyperParams.prepostprocess_dropout,
            ModelHyperParams.attention_dropout, ModelHyperParams.relu_dropout,
            ModelHyperParams.preprocess_cmd, ModelHyperParams.postprocess_cmd,
            ModelHyperParams.weight_sharing, TrainTaskConfig.label_smooth_eps)
        train_reader = paddle.batch(
            wmt16.train(ModelHyperParams.src_vocab_size,
                        ModelHyperParams.trg_vocab_size),
            TrainTaskConfig.batch_size)
        if naive_optimize:
            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        else:
            optimizer = fluid.optimizer.Adam(
                learning_rate=NoamDecay(ModelHyperParams.d_model,
                                        TrainTaskConfig.warmup_steps,
                                        TrainTaskConfig.learning_rate),
                beta1=TrainTaskConfig.beta1,
                beta2=TrainTaskConfig.beta2,
                epsilon=TrainTaskConfig.eps)

        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        enc_inputs, dec_inputs, label, weights = prepare_train_input(
            batch, ModelHyperParams.eos_idx, ModelHyperParams.eos_idx,
            ModelHyperParams.n_head)

        dy_sum_cost, dy_avg_cost, dy_predict, dy_token_num = model(
            enc_inputs, dec_inputs, label, weights)

        return dy_avg_cost


if __name__ == "__main__":
    runtime_main(TestTransformer)
