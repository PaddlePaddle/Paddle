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
from paddle.fluid import Embedding, LayerNorm, FC, Layer
from paddle.fluid.dygraph import to_variable, guard
from test_imperative_base import new_program_scope
from paddle.fluid import core
from test_imperative_transformer import TransFormer, TrainTaskConfig, ModelHyperParams
import numpy as np
import six
np.set_printoptions(suppress=True)


def create_data(is_static=False):
    if is_static:
        return [
            src_word_np, src_pos_np, src_slf_attn_bias_np, trg_word_np,
            trg_pos_np, trg_slf_attn_bias_np, trg_src_attn_bias_np, lbl_word_np,
            lbl_weight_np
        ]
    else:
        enc_inputs = [
            to_variable(
                src_word_np, name='src_word'), to_variable(
                    src_pos_np, name='src_pos'), to_variable(
                        src_slf_attn_bias_np, name='src_slf_attn_bias')
        ]
        dec_inputs = [
            to_variable(
                trg_word_np, name='trg_word'), to_variable(
                    trg_pos_np, name='trg_pos'), to_variable(
                        trg_slf_attn_bias_np, name='trg_slf_attn_bias'),
            to_variable(
                trg_src_attn_bias_np, name='trg_src_attn_bias')
        ]
        label = to_variable(lbl_word_np, name='lbl_word')
        weight = to_variable(lbl_weight_np, name='lbl_weight')
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
batch_num = 5

np.random.seed = 90
src_word_np = np.arange(1, TrainTaskConfig.batch_size * seq_len + 1).reshape(
    [TrainTaskConfig.batch_size, seq_len, 1]).astype('int64')
src_pos_np = np.random.randint(
    1, seq_len, size=(TrainTaskConfig.batch_size, seq_len, 1), dtype='int64')
src_slf_attn_bias_np = np.random.randn(TrainTaskConfig.batch_size,
                                       ModelHyperParams.n_head, seq_len,
                                       seq_len).astype('float32')

trg_word_np = np.arange(1, TrainTaskConfig.batch_size * seq_len + 1).reshape(
    [TrainTaskConfig.batch_size, seq_len, 1]).astype('int64')
trg_pos_np = np.random.randint(
    1, seq_len, size=(TrainTaskConfig.batch_size, seq_len, 1), dtype='int64')
trg_slf_attn_bias_np = np.random.randn(TrainTaskConfig.batch_size,
                                       ModelHyperParams.n_head, seq_len,
                                       seq_len).astype('float32')
trg_src_attn_bias_np = np.random.randn(TrainTaskConfig.batch_size,
                                       ModelHyperParams.n_head, seq_len,
                                       seq_len).astype('float32')

lbl_word_np = np.random.randint(
    1,
    ModelHyperParams.src_vocab_size - 1,
    size=(TrainTaskConfig.batch_size * seq_len, 1),
    dtype='int64')
lbl_weight_np = np.random.randn(TrainTaskConfig.batch_size * seq_len,
                                1).astype('float32')


class TestDygraphTransformerSortGradient(unittest.TestCase):
    def test_transformer_sort_gradient_float32(self):
        seed = 90

        with guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True
            transformer = TransFormer(
                'transformer',
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=use_py_reader,
                is_test=False)
            if sync:
                lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
                    ModelHyperParams.d_model, TrainTaskConfig.warmup_steps)
                with fluid.default_main_program()._lr_schedule_guard():
                    learning_rate = lr_decay * TrainTaskConfig.learning_rate
                optimizer = fluid.optimizer.Adam(
                    learning_rate=learning_rate,
                    beta1=TrainTaskConfig.beta1,
                    beta2=TrainTaskConfig.beta2,
                    epsilon=TrainTaskConfig.eps)
            else:
                optimizer = fluid.optimizer.SGD(learning_rate=0.003)
            dy_param_init = dict()
            dy_param_updated = dict()
            for i in range(batch_num):
                enc_inputs, dec_inputs, label, weights = create_data()
                dy_sum_cost, dy_avg_cost, dy_predict, dy_token_num = transformer(
                    enc_inputs, dec_inputs, label, weights)

                if i == 0:
                    for param in transformer.parameters():
                        dy_param_init[param.name] = param.numpy()

                dy_avg_cost.backward(backward_strategy)
                optimizer.minimize(dy_avg_cost)
                transformer.clear_gradients()

                if i == batch_num - 1:
                    for param in transformer.parameters():
                        dy_param_updated[param.name] = param.numpy()

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            transformer = TransFormer(
                'transformer',
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=use_py_reader,
                is_test=False)
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            optimizer = fluid.optimizer.SGD(learning_rate=0.003)

            data_input_names = encoder_data_input_fields + decoder_data_input_fields[:
                                                                                     -1] + label_data_input_fields
            all_inputs = make_all_inputs(data_input_names)
            enc_inputs_len = len(encoder_data_input_fields)
            dec_inputs_len = len(decoder_data_input_fields[:-1])
            enc_inputs = all_inputs[0:enc_inputs_len]
            dec_inputs = all_inputs[enc_inputs_len:enc_inputs_len +
                                    dec_inputs_len]
            label = all_inputs[-2]
            weights = all_inputs[-1]
            static_param_updated = dict()
            static_param_init = dict()
            static_param_name_list = list()
            static_sum_cost, static_avg_cost, static_predict, static_token_num = transformer(
                enc_inputs, dec_inputs, label, weights)
            optimizer.minimize(static_avg_cost)
            for param in transformer.parameters():
                static_param_name_list.append(param.name)
            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)
            for i in range(len(static_param_name_list)):
                static_param_init[static_param_name_list[i]] = out[i]
            static_sum_cost_value = None
            static_avg_cost_value = None
            static_predict_value = None
            static_token_num_value = None
            for i in range(batch_num):
                feed_dict = create_feed_dict_list(create_data(True))
                fetch_list = [
                    static_sum_cost, static_avg_cost, static_predict,
                    static_token_num
                ]

                fetch_list.extend(static_param_name_list)
                out = exe.run(fluid.default_main_program(),
                              feed=feed_dict,
                              fetch_list=fetch_list)
                static_sum_cost_value = out[0]
                static_avg_cost_value = out[1]
                static_predict_value = out[2]
                static_token_num_value = out[3]
                if i == batch_num - 1:
                    for k in range(4, len(out)):
                        static_param_updated[static_param_name_list[k -
                                                                    4]] = out[k]

        self.assertTrue(
            np.array_equal(static_avg_cost_value, dy_avg_cost.numpy()))
        self.assertTrue(
            np.array_equal(static_sum_cost_value, dy_sum_cost.numpy()))
        self.assertTrue(
            np.array_equal(static_predict_value, dy_predict.numpy()))
        self.assertTrue(
            np.array_equal(static_token_num_value, dy_token_num.numpy()))

        for key, value in six.iteritems(static_param_init):
            self.assertTrue(np.array_equal(value, dy_param_init[key]))
        for key, value in six.iteritems(static_param_updated):
            self.assertTrue(np.array_equal(value, dy_param_updated[key]))


if __name__ == '__main__':
    unittest.main()
