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

import unittest
import numpy

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.framework import program_guard, Program

from paddle.fluid.executor import Executor
from paddle.fluid import framework

from paddle.fluid.layers.rnn import LSTMCell, GRUCell, RNNCell
from paddle.fluid.layers import rnn as dynamic_rnn
from paddle.fluid import contrib
from paddle.fluid.contrib.layers import basic_lstm
import paddle.fluid.layers.utils as utils

import numpy as np


class TestLSTMCellError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            batch_size, input_size, hidden_size = 4, 16, 16
            inputs = fluid.data(name='inputs',
                                shape=[None, input_size],
                                dtype='float32')
            pre_hidden = fluid.data(name='pre_hidden',
                                    shape=[None, hidden_size],
                                    dtype='float32')
            pre_cell = fluid.data(name='pre_cell',
                                  shape=[None, hidden_size],
                                  dtype='float32')
            cell = LSTMCell(hidden_size)

            def test_input_Variable():
                np_input = np.random.random(
                    (batch_size, input_size)).astype("float32")
                cell(np_input, [pre_hidden, pre_cell])

            self.assertRaises(TypeError, test_input_Variable)

            def test_pre_hidden_Variable():
                np_pre_hidden = np.random.random(
                    (batch_size, hidden_size)).astype("float32")
                cell(inputs, [np_pre_hidden, pre_cell])

            self.assertRaises(TypeError, test_pre_hidden_Variable)

            def test_pre_cell_Variable():
                np_pre_cell = np.random.random(
                    (batch_size, input_size)).astype("float32")
                cell(inputs, [pre_hidden, np_pre_cell])

            self.assertRaises(TypeError, test_pre_cell_Variable)

            def test_input_type():
                error_inputs = fluid.data(name='error_inputs',
                                          shape=[None, input_size],
                                          dtype='int32')
                cell(error_inputs, [pre_hidden, pre_cell])

            self.assertRaises(TypeError, test_input_type)

            def test_pre_hidden_type():
                error_pre_hidden = fluid.data(name='error_pre_hidden',
                                              shape=[None, hidden_size],
                                              dtype='int32')
                cell(inputs, [error_pre_hidden, pre_cell])

            self.assertRaises(TypeError, test_pre_hidden_type)

            def test_pre_cell_type():
                error_pre_cell = fluid.data(name='error_pre_cell',
                                            shape=[None, hidden_size],
                                            dtype='int32')
                cell(inputs, [pre_hidden, error_pre_cell])

            self.assertRaises(TypeError, test_pre_cell_type)

            def test_dtype():
                # the input type must be Variable
                LSTMCell(hidden_size, dtype="int32")

            self.assertRaises(TypeError, test_dtype)


class TestLSTMCell(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16

    def test_run(self):
        inputs = fluid.data(name='inputs',
                            shape=[None, self.input_size],
                            dtype='float32')
        pre_hidden = fluid.data(name='pre_hidden',
                                shape=[None, self.hidden_size],
                                dtype='float32')
        pre_cell = fluid.data(name='pre_cell',
                              shape=[None, self.hidden_size],
                              dtype='float32')

        cell = LSTMCell(self.hidden_size)
        lstm_hidden_new, lstm_states_new = cell(inputs, [pre_hidden, pre_cell])

        lstm_unit = contrib.layers.rnn_impl.BasicLSTMUnit(
            "basicLSTM", self.hidden_size, None, None, None, None, 1.0,
            "float32")
        lstm_hidden, lstm_cell = lstm_unit(inputs, pre_hidden, pre_cell)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        inputs_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.input_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')
        pre_cell_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        param_names = [[
            "LSTMCell/BasicLSTMUnit_0.w_0", "basicLSTM/BasicLSTMUnit_0.w_0"
        ], ["LSTMCell/BasicLSTMUnit_0.b_0", "basicLSTM/BasicLSTMUnit_0.b_0"]]

        for names in param_names:
            param = np.array(fluid.global_scope().find_var(
                names[0]).get_tensor())
            param = np.random.uniform(-0.1, 0.1,
                                      size=param.shape).astype('float32')
            fluid.global_scope().find_var(names[0]).get_tensor().set(
                param, place)
            fluid.global_scope().find_var(names[1]).get_tensor().set(
                param, place)

        out = exe.run(feed={
            'inputs': inputs_np,
            'pre_hidden': pre_hidden_np,
            'pre_cell': pre_cell_np
        },
                      fetch_list=[lstm_hidden_new, lstm_hidden])

        np.testing.assert_allclose(out[0], out[1], rtol=0.0001, atol=0)


class TestGRUCellError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            batch_size, input_size, hidden_size = 4, 16, 16
            inputs = fluid.data(name='inputs',
                                shape=[None, input_size],
                                dtype='float32')
            pre_hidden = layers.data(name='pre_hidden',
                                     shape=[None, hidden_size],
                                     append_batch_size=False,
                                     dtype='float32')
            cell = GRUCell(hidden_size)

            def test_input_Variable():
                np_input = np.random.random(
                    (batch_size, input_size)).astype("float32")
                cell(np_input, pre_hidden)

            self.assertRaises(TypeError, test_input_Variable)

            def test_pre_hidden_Variable():
                np_pre_hidden = np.random.random(
                    (batch_size, hidden_size)).astype("float32")
                cell(inputs, np_pre_hidden)

            self.assertRaises(TypeError, test_pre_hidden_Variable)

            def test_input_type():
                error_inputs = fluid.data(name='error_inputs',
                                          shape=[None, input_size],
                                          dtype='int32')
                cell(error_inputs, pre_hidden)

            self.assertRaises(TypeError, test_input_type)

            def test_pre_hidden_type():
                error_pre_hidden = fluid.data(name='error_pre_hidden',
                                              shape=[None, hidden_size],
                                              dtype='int32')
                cell(inputs, error_pre_hidden)

            self.assertRaises(TypeError, test_pre_hidden_type)

            def test_dtype():
                # the input type must be Variable
                GRUCell(hidden_size, dtype="int32")

            self.assertRaises(TypeError, test_dtype)


class TestGRUCell(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16

    def test_run(self):
        inputs = fluid.data(name='inputs',
                            shape=[None, self.input_size],
                            dtype='float32')
        pre_hidden = layers.data(name='pre_hidden',
                                 shape=[None, self.hidden_size],
                                 append_batch_size=False,
                                 dtype='float32')

        cell = GRUCell(self.hidden_size)
        gru_hidden_new, _ = cell(inputs, pre_hidden)

        gru_unit = contrib.layers.rnn_impl.BasicGRUUnit("basicGRU",
                                                        self.hidden_size, None,
                                                        None, None, None,
                                                        "float32")
        gru_hidden = gru_unit(inputs, pre_hidden)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        inputs_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.input_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        param_names = [
            ["GRUCell/BasicGRUUnit_0.w_0", "basicGRU/BasicGRUUnit_0.w_0"],
            ["GRUCell/BasicGRUUnit_0.w_1", "basicGRU/BasicGRUUnit_0.w_1"],
            ["GRUCell/BasicGRUUnit_0.b_0", "basicGRU/BasicGRUUnit_0.b_0"],
            ["GRUCell/BasicGRUUnit_0.b_1", "basicGRU/BasicGRUUnit_0.b_1"]
        ]

        for names in param_names:
            param = np.array(fluid.global_scope().find_var(
                names[0]).get_tensor())
            param = np.random.uniform(-0.1, 0.1,
                                      size=param.shape).astype('float32')
            fluid.global_scope().find_var(names[0]).get_tensor().set(
                param, place)
            fluid.global_scope().find_var(names[1]).get_tensor().set(
                param, place)

        out = exe.run(feed={
            'inputs': inputs_np,
            'pre_hidden': pre_hidden_np
        },
                      fetch_list=[gru_hidden_new, gru_hidden])

        np.testing.assert_allclose(out[0], out[1], rtol=0.0001, atol=0)


class TestRnnError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            batch_size = 4
            input_size = 16
            hidden_size = 16
            seq_len = 4
            inputs = fluid.data(name='inputs',
                                shape=[None, input_size],
                                dtype='float32')
            pre_hidden = layers.data(name='pre_hidden',
                                     shape=[None, hidden_size],
                                     append_batch_size=False,
                                     dtype='float32')
            inputs_basic_lstm = fluid.data(name='inputs_basic_lstm',
                                           shape=[None, None, input_size],
                                           dtype='float32')
            sequence_length = fluid.data(name="sequence_length",
                                         shape=[None],
                                         dtype='int64')

            inputs_dynamic_rnn = layers.transpose(inputs_basic_lstm,
                                                  perm=[1, 0, 2])
            cell = LSTMCell(hidden_size, name="LSTMCell_for_rnn")
            np_inputs_dynamic_rnn = np.random.random(
                (seq_len, batch_size, input_size)).astype("float32")

            def test_input_Variable():
                dynamic_rnn(cell=cell,
                            inputs=np_inputs_dynamic_rnn,
                            sequence_length=sequence_length,
                            is_reverse=False)

            self.assertRaises(TypeError, test_input_Variable)

            def test_input_list():
                dynamic_rnn(cell=cell,
                            inputs=[np_inputs_dynamic_rnn],
                            sequence_length=sequence_length,
                            is_reverse=False)

            self.assertRaises(TypeError, test_input_list)

            def test_initial_states_type():
                cell = GRUCell(hidden_size, name="GRUCell_for_rnn")
                error_initial_states = np.random.random(
                    (batch_size, hidden_size)).astype("float32")
                dynamic_rnn(cell=cell,
                            inputs=inputs_dynamic_rnn,
                            initial_states=error_initial_states,
                            sequence_length=sequence_length,
                            is_reverse=False)

            self.assertRaises(TypeError, test_initial_states_type)

            def test_initial_states_list():
                error_initial_states = [
                    np.random.random(
                        (batch_size, hidden_size)).astype("float32"),
                    np.random.random(
                        (batch_size, hidden_size)).astype("float32")
                ]
                dynamic_rnn(cell=cell,
                            inputs=inputs_dynamic_rnn,
                            initial_states=error_initial_states,
                            sequence_length=sequence_length,
                            is_reverse=False)

            self.assertRaises(TypeError, test_initial_states_type)

            def test_sequence_length_type():
                np_sequence_length = np.random.random(
                    (batch_size)).astype("float32")
                dynamic_rnn(cell=cell,
                            inputs=inputs_dynamic_rnn,
                            sequence_length=np_sequence_length,
                            is_reverse=False)

            self.assertRaises(TypeError, test_sequence_length_type)


class TestRnn(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16
        self.seq_len = 4

    def test_run(self):
        inputs_basic_lstm = fluid.data(name='inputs_basic_lstm',
                                       shape=[None, None, self.input_size],
                                       dtype='float32')
        sequence_length = fluid.data(name="sequence_length",
                                     shape=[None],
                                     dtype='int64')

        inputs_dynamic_rnn = layers.transpose(inputs_basic_lstm, perm=[1, 0, 2])
        cell = LSTMCell(self.hidden_size, name="LSTMCell_for_rnn")
        output, final_state = dynamic_rnn(cell=cell,
                                          inputs=inputs_dynamic_rnn,
                                          sequence_length=sequence_length,
                                          is_reverse=False)
        output_new = layers.transpose(output, perm=[1, 0, 2])

        rnn_out, last_hidden, last_cell = basic_lstm(inputs_basic_lstm, None, None, self.hidden_size, num_layers=1, \
                batch_first = False, bidirectional=False, sequence_length=sequence_length, forget_bias = 1.0)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        inputs_basic_lstm_np = np.random.uniform(
            -0.1, 0.1,
            (self.seq_len, self.batch_size, self.input_size)).astype('float32')
        sequence_length_np = np.ones(self.batch_size,
                                     dtype='int64') * self.seq_len

        inputs_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.input_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')
        pre_cell_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        param_names = [[
            "LSTMCell_for_rnn/BasicLSTMUnit_0.w_0",
            "basic_lstm_layers_0/BasicLSTMUnit_0.w_0"
        ],
                       [
                           "LSTMCell_for_rnn/BasicLSTMUnit_0.b_0",
                           "basic_lstm_layers_0/BasicLSTMUnit_0.b_0"
                       ]]

        for names in param_names:
            param = np.array(fluid.global_scope().find_var(
                names[0]).get_tensor())
            param = np.random.uniform(-0.1, 0.1,
                                      size=param.shape).astype('float32')
            fluid.global_scope().find_var(names[0]).get_tensor().set(
                param, place)
            fluid.global_scope().find_var(names[1]).get_tensor().set(
                param, place)

        out = exe.run(feed={
            'inputs_basic_lstm': inputs_basic_lstm_np,
            'sequence_length': sequence_length_np,
            'inputs': inputs_np,
            'pre_hidden': pre_hidden_np,
            'pre_cell': pre_cell_np
        },
                      fetch_list=[output_new, rnn_out])

        np.testing.assert_allclose(out[0], out[1], rtol=0.0001)


class TestRnnUtil(unittest.TestCase):
    """
    Test cases for rnn apis' utility methods for coverage.
    """

    def test_case(self):
        inputs = {"key1": 1, "key2": 2}
        func = lambda x: x + 1
        outputs = utils.map_structure(func, inputs)
        utils.assert_same_structure(inputs, outputs)
        try:
            inputs["key3"] = 3
            utils.assert_same_structure(inputs, outputs)
        except ValueError as identifier:
            pass


class EncoderCell(RNNCell):
    """Encoder Cell"""

    def __init__(
        self,
        num_layers,
        hidden_size,
        dropout_prob=0.,
        init_scale=0.1,
    ):
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
                out,
                self.dropout_prob,
            ) if self.dropout_prob else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class DecoderCell(RNNCell):
    """Decoder Cell"""

    def __init__(self, num_layers, hidden_size, dropout_prob=0.):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = []
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size))

    def call(self, step_input, states):
        new_lstm_states = []
        for i in range(self.num_layers):
            out, new_lstm_state = self.lstm_cells[i](step_input, states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
            ) if self.dropout_prob else out
            new_lstm_states.append(new_lstm_state)
        return step_input, new_lstm_states


def def_seq2seq_model(num_layers, hidden_size, dropout_prob, src_vocab_size,
                      trg_vocab_size):
    "vanilla seq2seq model"
    # data
    source = fluid.data(name="src", shape=[None, None], dtype="int64")
    source_length = fluid.data(name="src_sequence_length",
                               shape=[None],
                               dtype="int64")
    target = fluid.data(name="trg", shape=[None, None], dtype="int64")
    target_length = fluid.data(name="trg_sequence_length",
                               shape=[None],
                               dtype="int64")
    label = fluid.data(name="label", shape=[None, None, 1], dtype="int64")

    # embedding
    src_emb = fluid.embedding(source, (src_vocab_size, hidden_size))
    tar_emb = fluid.embedding(target, (src_vocab_size, hidden_size))

    # encoder
    enc_cell = EncoderCell(num_layers, hidden_size, dropout_prob)
    enc_output, enc_final_state = dynamic_rnn(cell=enc_cell,
                                              inputs=src_emb,
                                              sequence_length=source_length)

    # decoder
    dec_cell = DecoderCell(num_layers, hidden_size, dropout_prob)
    dec_output, dec_final_state = dynamic_rnn(cell=dec_cell,
                                              inputs=tar_emb,
                                              initial_states=enc_final_state)
    logits = layers.fc(dec_output,
                       size=trg_vocab_size,
                       num_flatten_dims=len(dec_output.shape) - 1,
                       bias_attr=False)

    # loss
    loss = layers.softmax_with_cross_entropy(logits=logits,
                                             label=label,
                                             soft_label=False)
    loss = layers.unsqueeze(loss, axes=[2])
    max_tar_seq_len = layers.shape(target)[1]
    tar_mask = layers.sequence_mask(target_length,
                                    maxlen=max_tar_seq_len,
                                    dtype="float32")
    loss = loss * tar_mask
    loss = layers.reduce_mean(loss, dim=[0])
    loss = layers.reduce_sum(loss)

    # optimizer
    optimizer = fluid.optimizer.Adam(0.001)
    optimizer.minimize(loss)
    return loss


class TestSeq2SeqModel(unittest.TestCase):
    """
    Test cases to confirm seq2seq api training correctly.
    """

    def setUp(self):
        np.random.seed(123)
        self.model_hparams = {
            "num_layers": 2,
            "hidden_size": 128,
            "dropout_prob": 0.1,
            "src_vocab_size": 100,
            "trg_vocab_size": 100
        }

        self.iter_num = iter_num = 2
        self.batch_size = batch_size = 4
        src_seq_len = 10
        trg_seq_len = 12
        self.data = {
            "src":
            np.random.randint(
                2, self.model_hparams["src_vocab_size"],
                (iter_num * batch_size, src_seq_len)).astype("int64"),
            "src_sequence_length":
            np.random.randint(1, src_seq_len,
                              (iter_num * batch_size, )).astype("int64"),
            "trg":
            np.random.randint(
                2, self.model_hparams["src_vocab_size"],
                (iter_num * batch_size, trg_seq_len)).astype("int64"),
            "trg_sequence_length":
            np.random.randint(1, trg_seq_len,
                              (iter_num * batch_size, )).astype("int64"),
            "label":
            np.random.randint(
                2, self.model_hparams["src_vocab_size"],
                (iter_num * batch_size, trg_seq_len, 1)).astype("int64"),
        }

        place = core.CUDAPlace(
            0) if core.is_compiled_with_cuda() else core.CPUPlace()
        self.exe = Executor(place)

    def test_seq2seq_model(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            cost = def_seq2seq_model(**self.model_hparams)
            self.exe.run(startup_program)
            for iter_idx in range(self.iter_num):
                cost_val = self.exe.run(feed={
                    "src":
                    self.data["src"][iter_idx * self.batch_size:(iter_idx + 1) *
                                     self.batch_size, :],
                    "src_sequence_length":
                    self.data["src_sequence_length"][iter_idx *
                                                     self.batch_size:(iter_idx +
                                                                      1) *
                                                     self.batch_size],
                    "trg":
                    self.data["trg"][iter_idx * self.batch_size:(iter_idx + 1) *
                                     self.batch_size, :],
                    "trg_sequence_length":
                    self.data["trg_sequence_length"][iter_idx *
                                                     self.batch_size:(iter_idx +
                                                                      1) *
                                                     self.batch_size],
                    "label":
                    self.data["label"][iter_idx *
                                       self.batch_size:(iter_idx + 1) *
                                       self.batch_size]
                },
                                        fetch_list=[cost])[0]
                print("iter_idx: %d, cost: %f" % (iter_idx, cost_val))


if __name__ == '__main__':
    unittest.main()
