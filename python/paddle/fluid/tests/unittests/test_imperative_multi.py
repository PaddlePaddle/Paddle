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
from paddle.fluid.imperative import Embedding, LayerNorm, FC, to_variable, Layer, guard
from test_imperative_base import new_program_scope
from paddle.fluid import core
import numpy as np
import six
import pdb
np.set_printoptions(suppress=True)


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
    n_head = 1
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
    dropout_seed = None
    # the flag indicating whether to share embedding and softmax weights.
    # vocabularies in source and target should be same for weight sharing.
    weight_sharing = True


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
n_head = 1
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
dropout_seed = None
# the flag indicating whether to share embedding and softmax weights.
# vocabularies in source and target should be same for weight sharing.
weight_sharing = True


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


# class PositionwiseFeedForwardLayer(Layer):
#     def __init__(self, name_scope, d_inner_hid, d_hid, dropout_rate):
#         super(PositionwiseFeedForwardLayer, self).__init__(name_scope)
#         self._i2h = FC(name_scope=self.full_name(),
#                        size=d_inner_hid,
#                        num_flatten_dims=2,
#                        act="relu")
#         self._h2o = FC(name_scope=self.full_name(),
#                        size=d_hid,
#                        num_flatten_dims=2)
#         self._dropout_rate = dropout_rate
#
#     def forward(self, x):
#         hidden = self._i2h(x)
#         if self._dropout_rate:
#             hidden = fluid.layers.dropout(
#                 hidden,
#                 dropout_prob=self._dropout_rate,
#                 seed=ModelHyperParams.dropout_seed,
#                 is_test=False)
#         out = self._h2o(hidden)
#         return out


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

    def forward(self, queries, keys, values, attn_bias):
        # compute q ,k ,v
        keys = queries if keys is None else keys
        values = keys if values is None else values

        test = queries

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

        # scale dot product attention
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
                weights, dropout_prob=self._dropout_rate, seed=1, is_test=False)
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
        proj_out = self._proj_fc(final_out)
        out = fluid.layers.reduce_sum(proj_out)
        return out, test


class DecoderSubLayer(Layer):
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
                 preprocess_cmd,
                 postprocess_cmd,
                 cache=None,
                 gather_idx=None):
        super(DecoderSubLayer, self).__init__(name_scope)
        self._postprocess_cmd = postprocess_cmd
        self._preprocess_cmd = preprocess_cmd
        self._prepostprcess_dropout = prepostprocess_dropout
        self._pre_process_layer = PrePostProcessLayer(self.full_name(),
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

    def forward(self, dec_input, enc_output, slf_attn_bias, dec_enc_attn_bias):
        if fluid.framework._in_imperative_mode():
            pre_process_rlt = to_variable(dec_input)
        else:
            pre_process_rlt = fluid.layers.assign(dec_input)
        # pre_process_rlt = self._pre_process_layer(
        #     None, dec_input, self._preprocess_cmd, self._prepostprcess_dropout)
        # if fluid.framework._in_imperative_mode():
        #     k = to_variable(pre_process_rlt)
        #     v = to_variable(pre_process_rlt)
        # else:
        #     k = fluid.layers.assign(pre_process_rlt)
        #     v = fluid.layers.assign(pre_process_rlt)

        slf_attn_output, test = self._multihead_attention_layer(
            pre_process_rlt, None, None, slf_attn_bias)
        return slf_attn_output, test


class TestMulti_Input(unittest.TestCase):
    def test(self):
        seed = 90
        np_inp1 = np.random.randn(32, 4, 512).astype('float32')

        bias_np1 = np.random.randn(32, 1, 4, 4).astype('float32')
        np_inp2 = np.random.randn(32, 4, 512).astype('float32')
        bias_np2 = np.random.randn(32, 1, 4, 4).astype('float32')
        batch_num = 1
        with fluid.imperative.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            var_inp1 = fluid.imperative.base.to_variable(np_inp1)

            bias_var1 = fluid.imperative.base.to_variable(bias_np1)
            var_inp2 = fluid.imperative.base.to_variable(np_inp2)
            bias_var2 = fluid.imperative.base.to_variable(bias_np2)
            multi = DecoderSubLayer(
                'multi',
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
                cache=None,
                gather_idx=None)
            optimizer = fluid.optimizer.SGD(learning_rate=0.003)
            for i in range(batch_num):
                dy_out, dq = multi(var_inp1, var_inp2, bias_var1, bias_var2)
                dy_out._backward()
                dq_grad = dq._gradient()
                dw = multi._multihead_attention_layer._q_fc._w

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            inp_var1 = fluid.layers.data(
                name="inp1", shape=[-1, 4, 512], append_batch_size=False)
            bias_var1 = fluid.layers.data(
                name="bias1", shape=[-1, 1, 4, 4], append_batch_size=False)
            inp_var2 = fluid.layers.data(
                name="inp2", shape=[-1, 4, 512], append_batch_size=False)
            bias_var2 = fluid.layers.data(
                name="bias2", shape=[-1, 1, 4, 4], append_batch_size=False)

            multi = DecoderSubLayer(
                'multi',
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
                cache=None,
                gather_idx=None)
            static_out, q = multi(inp_var1, inp_var2, bias_var1, bias_var2)

            optimizer = fluid.optimizer.SGD(learning_rate=0.003)
            # optimizer.minimize(static_out)
            fluid.backward.append_backward(static_out)
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(fluid.default_startup_program())
            # print(fluid.framework.default_main_program().block(0).ops)
            for i in range(batch_num):
                st_out, sq_grad, sw = exe.run(
                    feed={
                        inp_var1.name: np_inp1,
                        inp_var2.name: np_inp2,
                        bias_var1.name: bias_np1,
                        bias_var2.name: bias_np2
                    },
                    fetch_list=[
                        static_out.name, q.name + '@GRAD',
                        multi._multihead_attention_layer._q_fc._w.name + '@GRAD'
                    ])

        print(st_out)
        print("==============")
        print(dy_out._numpy())

        if not np.array_equal(sq_grad, dq_grad):
            # print("Static {}".format(sq_grad))
            # print("dy {}".format(dq_grad))
            print(sq_grad - dq_grad)
        # print("diff is {}".format(np.where(sq_grad != np.inf)))


if __name__ == '__main__':
    unittest.main()
