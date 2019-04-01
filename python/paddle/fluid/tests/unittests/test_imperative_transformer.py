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
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, FC, to_variable, Layer, guard
from test_imperative_base import new_program_scope
from paddle.fluid import core
import numpy as np
import six
np.set_printoptions(suppress=True)


# Copy from models
class TrainTaskConfig(object):
    # support both CPU and GPU now.
    use_gpu = True
    # the epoch number to train.
    pass_num = 30
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
    # the directory for saving trained models.
    model_dir = "trained_models"
    # the directory for saving checkpoints.
    ckpt_dir = "trained_ckpts"
    # the directory for loading checkpoint.
    # If provided, continue training from the checkpoint.
    ckpt_path = None
    # the parameter to initialize the learning rate scheduler.
    # It should be provided if use checkpoints, since the checkpoint doesn't
    # include the training step counter currently.
    start_step = 0
    # the frequency to save trained models.
    save_freq = 10000


class InferTaskConfig(object):
    use_gpu = True
    # the number of examples in one run for sequence generation.
    batch_size = 10
    # the parameters for beam search.
    beam_size = 5
    max_out_len = 256
    # the number of decoded sentences to output.
    n_best = 1
    # the flags indicating whether to output the special tokens.
    output_bos = False
    output_eos = False
    output_unk = True
    # the directory for loading the trained model.
    model_path = "trained_models/pass_1.infer.model"


class ModelHyperParams(object):
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
    max_length = 4
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
    n_layer = 1
    # dropout rates of different modules.
    prepostprocess_dropout = 0.1
    attention_dropout = 0.1
    relu_dropout = 0.1
    # to process before each sub-layer
    preprocess_cmd = "n"  # layer normalization
    # to process after each sub-layer
    postprocess_cmd = "da"  # dropout + residual connection
    # random seed used in dropout for CE.
    dropout_seed = 1
    # the flag indicating whether to share embedding and softmax weights.
    # vocabularies in source and target should be same for weight sharing.
    weight_sharing = True


def merge_cfg_from_list(cfg_list, g_cfgs):
    """
    Set the above global configurations using the cfg_list.
    """
    assert len(cfg_list) % 2 == 0
    for key, value in zip(cfg_list[0::2], cfg_list[1::2]):
        for g_cfg in g_cfgs:
            if hasattr(g_cfg, key):
                try:
                    value = eval(value)
                except Exception:  # for file path
                    pass
                setattr(g_cfg, key, value)
                break


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


def create_data(is_static=False):
    if is_static:
        return [
            src_word_np, src_pos_np, src_slf_attn_bias_np, trg_word_np,
            trg_pos_np, trg_slf_attn_bias_np, trg_src_attn_bias_np, lbl_word_np,
            lbl_weight_np
        ]
    else:
        enc_inputs = [
            to_variable(src_word_np), to_variable(src_pos_np),
            to_variable(src_slf_attn_bias_np)
        ]
        dec_inputs = [
            to_variable(trg_word_np), to_variable(trg_pos_np),
            to_variable(trg_slf_attn_bias_np), to_variable(trg_src_attn_bias_np)
        ]
        label = to_variable(lbl_word_np)
        weight = to_variable(lbl_weight_np)
        return enc_inputs, dec_inputs, label, weight


def create_feed_dict_list(data, init=False):
    if init:
        data_input_names = encoder_data_input_fields + \
                           decoder_data_input_fields[:-1] + label_data_input_fields + pos_enc_param_names
    else:
        data_input_names = encoder_data_input_fields + \
                           decoder_data_input_fields[:-1] + label_data_input_fields
    feed_dict_list = dict()
    for i in range(len(data_input_names)):
        feed_dict_list[data_input_names[i]] = data[i]
    return feed_dict_list


def make_all_inputs(input_fields):
    """
    Define the input data layers for the transformer model.
    """
    inputs = []
    for input_field in input_fields:
        input_var = fluid.layers.data(
            name=input_field,
            shape=input_descs[input_field][0],
            dtype=input_descs[input_field][1],
            lod_level=input_descs[input_field][2]
            if len(input_descs[input_field]) == 3 else 0,
            append_batch_size=False)
        inputs.append(input_var)
    return inputs


# The placeholder for batch_size in compile time. Must be -1 currently to be
# consistent with some ops' infer-shape output in compile time, such as the
# sequence_expand op used in beamsearch decoder.
batch_size = 32
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
    "src_slf_attn_bias": [(batch_size, ModelHyperParams.n_head, seq_len,
                           seq_len), "float32"],
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
    "trg_slf_attn_bias": [(batch_size, ModelHyperParams.n_head, seq_len,
                           seq_len), "float32"],
    # This input is used to remove attention weights on paddings of the source
    # input in the encoder-decoder attention.
    # The actual data shape of trg_src_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_src_len_in_batch]
    "trg_src_attn_bias": [(batch_size, ModelHyperParams.n_head, seq_len,
                           seq_len), "float32"],
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
    "init_score",
    "init_idx",
    "trg_src_attn_bias", )
# if we use py_reader
use_py_reader = False

# if we run sync mode
sync = False

# how many batches we use
batch_num = 1

np.random.seed = 1
src_word_np = np.random.randint(
    1,
    ModelHyperParams.src_vocab_size - 1,
    size=(batch_size, seq_len, 1),
    dtype='int64')
src_pos_np = np.random.randint(
    1, seq_len, size=(batch_size, seq_len, 1), dtype='int64')
src_slf_attn_bias_np = np.random.randn(batch_size, ModelHyperParams.n_head,
                                       seq_len, seq_len).astype('float32')

trg_word_np = np.random.randint(
    1,
    ModelHyperParams.src_vocab_size - 1,
    size=(batch_size, seq_len, 1),
    dtype='int64')
trg_pos_np = np.random.randint(
    1, seq_len, size=(batch_size, seq_len, 1), dtype='int64')
trg_slf_attn_bias_np = np.random.randn(batch_size, ModelHyperParams.n_head,
                                       seq_len, seq_len).astype('float32')
trg_src_attn_bias_np = np.random.randn(batch_size, ModelHyperParams.n_head,
                                       seq_len, seq_len).astype('float32')

lbl_word_np = np.random.randint(
    1,
    ModelHyperParams.src_vocab_size - 1,
    size=(batch_size * seq_len, 1),
    dtype='int64')
lbl_weight_np = np.random.randn(batch_size * seq_len, 1).astype('float32')

# np.random.seed = 1
# src_word_np = np.arange(0, 10).reshape([batch_size, seq_len, 1]).astype('int64')
# src_pos_np = np.random.randint(
#     1, seq_len, size=(batch_size, seq_len, 1), dtype='int64')
# src_slf_attn_bias_np = np.random.randn(batch_size, ModelHyperParams.n_head,
#                                        seq_len, seq_len).astype('float32')
#
# trg_word_np =  np.arange(0, 10).reshape([batch_size, seq_len, 1]).astype('int64')
# trg_pos_np = np.random.randint(
#     1, seq_len, size=(batch_size, seq_len, 1), dtype='int64')
# trg_slf_attn_bias_np = np.random.randn(batch_size, ModelHyperParams.n_head,
#                                        seq_len, seq_len).astype('float32')
# trg_src_attn_bias_np = np.random.randn(batch_size, ModelHyperParams.n_head,
#                                        seq_len, seq_len).astype('float32')
#
# lbl_word_np =  np.arange(0, 10).reshape([batch_size * seq_len, 1]).astype('int64')
# lbl_weight_np = np.random.randn(batch_size * seq_len, 1).astype('float32')
#
pos_inp1 = position_encoding_init(ModelHyperParams.max_length,
                                  ModelHyperParams.d_model)
pos_inp2 = position_encoding_init(ModelHyperParams.max_length,
                                  ModelHyperParams.d_model)


class MultiHeadAttentionLayer(Layer):
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

    def _mm(self, input):
        input_shape = input.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:], 1)
        ] + [self._size]
        self.x = self.create_parameter(
            attr=None, shape=param_shape, dtype=self._dtype, is_bias=False)

    def forward(self, queries, keys, values, attn_bias):
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values

        #  q = queries
        #  k = keys
        #  v = values
        q = self._q_fc(queries)
        k = self._k_fc(keys)
        v = self._v_fc(values)

        # split head
        reshaped_q = fluid.layers.reshape(
            x=q, shape=[0, 0, self._n_head, self._d_key], inplace=False)
        transpose_q = fluid.layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
        reshaped_k = fluid.layers.reshape(
            x=k, shape=[0, 0, self._n_head, self._d_key], inplace=False)
        transpose_k = fluid.layers.transpose(x=reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = fluid.layers.reshape(
            x=v, shape=[0, 0, self._n_head, self._d_value], inplace=False)
        transpose_v = fluid.layers.transpose(x=reshaped_v, perm=[0, 2, 1, 3])

        #scale dot product attention
        product = fluid.layers.matmul(
            x=transpose_q,
            y=transpose_k,
            transpose_y=True,
            alpha=self._d_model**-0.5)
        if attn_bias:
            product += attn_bias
        weights = fluid.layers.softmax(product)
        if self._dropout_rate:
            weights_droped = fluid.layers.dropout(
                weights,
                dropout_prob=self._dropout_rate,
                seed=ModelHyperParams.dropout_seed,
                is_test=False)
            out = fluid.layers.matmul(weights_droped, transpose_v)
        else:
            out = fluid.layers.matmul(weights, transpose_v)

        # combine heads
        if len(out.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        trans_x = fluid.layers.transpose(out, perm=[0, 2, 1, 3])
        final_out = fluid.layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=False)

        # fc to output
        print(final_out.shape)
        proj_out = self._proj_fc(final_out)
        return proj_out


class PrePostProcessLayer(Layer):
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
        for cmd in process_cmd:
            if cmd == "a":  # add residual connection
                out = out + prev_out if prev_out else out
            elif cmd == "n":  # add layer normalization
                out = self._layer_norm(out)
            elif cmd == "d":  # add dropout
                if dropout_rate:
                    out = fluid.layers.dropout(
                        out,
                        dropout_prob=dropout_rate,
                        seed=ModelHyperParams.dropout_seed,
                        is_test=False)
        return out


class DecoderSubLayer(Layer):
    def __init__(self,
                 name_scope,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 attention_dropout,
                 cache=None,
                 preprocess_cmd="n",
                 gather_idx=None):
        super(DecoderSubLayer, self).__init__(name_scope)
        self._preprocess_layer = PrePostProcessLayer(self.full_name(),
                                                     preprocess_cmd, 3)
        self._multihead_attention_layer = MultiHeadAttentionLayer(
            self.full_name(),
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            cache=cache,
            gather_idx=gather_idx)

    def forward(self, input, slf_attn_bias):
        print(input.shape)
        print(slf_attn_bias.shape)
        y = self._preprocess_layer(None, input, "n", 0.1)
        slf_attn_output = self._multihead_attention_layer(y, None, None,
                                                          slf_attn_bias)
        return slf_attn_output, y


class TestDygraphTransformer(unittest.TestCase):
    def test_transformer_float32(self):
        seed = 90
        x1 = np.ones([32, 4, 512]).astype('float32')
        x2 = np.ones([32, 8, 4, 4]).astype('float32')
        with guard(place=fluid.CPUPlace()):
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            transformer = DecoderSubLayer(
                'transformer', ModelHyperParams.n_head, ModelHyperParams.d_key,
                ModelHyperParams.d_value, ModelHyperParams.d_model,
                ModelHyperParams.attention_dropout)
            optimizer = fluid.optimizer.SGD(learning_rate=0.003)
            dy_param_init = dict()
            dy_param_updated = dict()
            for i in range(batch_num):
                loss, y = transformer(to_variable(x1), to_variable(x2))
                loss = fluid.layers.reduce_sum(loss)
                print('dy los', loss.shape)
                if i == 0:
                    for param in transformer.parameters():
                        dy_param_init[param.name] = param._numpy()

                loss._backward()
                optimizer.minimize(loss)
                dy_key_value = y._gradient()
                transformer.clear_gradients()
                if i == batch_num - 1:
                    for param in transformer.parameters():
                        dy_param_updated[param.name] = param._numpy()

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            transformer = DecoderSubLayer(
                'transformer', ModelHyperParams.n_head, ModelHyperParams.d_key,
                ModelHyperParams.d_value, ModelHyperParams.d_model,
                ModelHyperParams.attention_dropout)
            exe = fluid.Executor(fluid.CPUPlace())
            optimizer = fluid.optimizer.SGD(learning_rate=0.003)

            data1 = fluid.layers.data(name='X', shape=[4, 512], dtype='float32')
            data2 = fluid.layers.data(
                name='Y', shape=[8, 4, 4], dtype='float32')
            loss, y = transformer(data1, data2)
            loss = fluid.layers.reduce_sum(loss)
            print('loss hspae', loss.shape)

            optimizer.minimize(loss)

            static_param_init = {}
            static_param_name_list = []
            static_param_updated = {}
            for param in transformer.parameters():
                static_param_name_list.append(param.name)
            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)

            for i in range(len(static_param_name_list)):
                static_param_init[static_param_name_list[i]] = out[i]

            print(fluid.default_main_program())
            for i in range(batch_num):
                feed_dict = {"X": x1, "Y": x2}
                fetch_list = [
                    "transformer/DecoderSubLayer_0/PrePostProcessLayer_0/LayerNorm_0.tmp_2@GRAD"
                ]
                fetch_list.extend(static_param_name_list)

                out = exe.run(fluid.default_main_program(),
                              feed=feed_dict,
                              fetch_list=fetch_list)
                if i == batch_num - 1:
                    static_key_value = out[0]
                    for k in range(1, len(out)):
                        static_param_updated[static_param_name_list[k -
                                                                    1]] = out[k]

        for key, value in six.iteritems(static_param_init):
            self.assertTrue(np.array_equal(value, dy_param_init[key]))
        for key, value in six.iteritems(static_param_updated):
            if not (value == dy_param_updated[key]).all():
                print(key)
        if not np.array_equal(dy_key_value, static_key_value):
            print("xxx", dy_key_value, static_key_value)
            print("yyy")
            print(dy_key_value - static_key_value)
            print(np.where(dy_key_value - static_key_value))


if __name__ == '__main__':
    unittest.main()
