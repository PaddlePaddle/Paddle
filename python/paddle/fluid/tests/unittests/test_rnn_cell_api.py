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
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.layers.utils as utils
from paddle.fluid import framework
from paddle.fluid.contrib.layers import basic_lstm
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.layers import rnn as dynamic_rnn


class TestRnnError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            batch_size = 4
            input_size = 16
            hidden_size = 16
            seq_len = 4
            inputs = fluid.data(
                name='inputs', shape=[None, input_size], dtype='float32'
            )
            pre_hidden = layers.data(
                name='pre_hidden',
                shape=[None, hidden_size],
                append_batch_size=False,
                dtype='float32',
            )
            inputs_basic_lstm = fluid.data(
                name='inputs_basic_lstm',
                shape=[None, None, input_size],
                dtype='float32',
            )
            sequence_length = fluid.data(
                name="sequence_length", shape=[None], dtype='int64'
            )

            inputs_dynamic_rnn = paddle.transpose(
                inputs_basic_lstm, perm=[1, 0, 2]
            )
            cell = paddle.nn.LSTMCell(
                input_size, hidden_size, name="LSTMCell_for_rnn"
            )
            np_inputs_dynamic_rnn = np.random.random(
                (seq_len, batch_size, input_size)
            ).astype("float32")

            def test_input_Variable():
                dynamic_rnn(
                    cell=cell,
                    inputs=np_inputs_dynamic_rnn,
                    sequence_length=sequence_length,
                    is_reverse=False,
                )

            self.assertRaises(TypeError, test_input_Variable)

            def test_input_list():
                dynamic_rnn(
                    cell=cell,
                    inputs=[np_inputs_dynamic_rnn],
                    sequence_length=sequence_length,
                    is_reverse=False,
                )

            self.assertRaises(TypeError, test_input_list)

            def test_initial_states_type():
                cell = paddle.nn.GRUCell(
                    input_size, hidden_size, name="GRUCell_for_rnn"
                )
                error_initial_states = np.random.random(
                    (batch_size, hidden_size)
                ).astype("float32")
                dynamic_rnn(
                    cell=cell,
                    inputs=inputs_dynamic_rnn,
                    initial_states=error_initial_states,
                    sequence_length=sequence_length,
                    is_reverse=False,
                )

            self.assertRaises(TypeError, test_initial_states_type)

            def test_initial_states_list():
                error_initial_states = [
                    np.random.random((batch_size, hidden_size)).astype(
                        "float32"
                    ),
                    np.random.random((batch_size, hidden_size)).astype(
                        "float32"
                    ),
                ]
                dynamic_rnn(
                    cell=cell,
                    inputs=inputs_dynamic_rnn,
                    initial_states=error_initial_states,
                    sequence_length=sequence_length,
                    is_reverse=False,
                )

            self.assertRaises(TypeError, test_initial_states_type)

            def test_sequence_length_type():
                np_sequence_length = np.random.random((batch_size)).astype(
                    "float32"
                )
                dynamic_rnn(
                    cell=cell,
                    inputs=inputs_dynamic_rnn,
                    sequence_length=np_sequence_length,
                    is_reverse=False,
                )

            self.assertRaises(TypeError, test_sequence_length_type)


class TestRnn(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16
        self.seq_len = 4

    def test_run(self):
        inputs_basic_lstm = fluid.data(
            name='inputs_basic_lstm',
            shape=[None, None, self.input_size],
            dtype='float32',
        )
        sequence_length = fluid.data(
            name="sequence_length", shape=[None], dtype='int64'
        )

        inputs_dynamic_rnn = paddle.transpose(inputs_basic_lstm, perm=[1, 0, 2])
        cell = paddle.nn.LSTMCell(
            self.input_size, self.hidden_size, name="LSTMCell_for_rnn"
        )
        output, final_state = dynamic_rnn(
            cell=cell,
            inputs=inputs_dynamic_rnn,
            sequence_length=sequence_length,
            is_reverse=False,
        )
        output_new = paddle.transpose(output, perm=[1, 0, 2])

        rnn_out, last_hidden, last_cell = basic_lstm(
            inputs_basic_lstm,
            None,
            None,
            self.hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=False,
            sequence_length=sequence_length,
            forget_bias=1.0,
        )

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        inputs_basic_lstm_np = np.random.uniform(
            -0.1, 0.1, (self.seq_len, self.batch_size, self.input_size)
        ).astype('float32')
        sequence_length_np = (
            np.ones(self.batch_size, dtype='int64') * self.seq_len
        )

        inputs_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.input_size)
        ).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)
        ).astype('float32')
        pre_cell_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)
        ).astype('float32')

        param_names = [
            [
                "LSTMCell_for_rnn/BasicLSTMUnit_0.w_0",
                "basic_lstm_layers_0/BasicLSTMUnit_0.w_0",
            ],
            [
                "LSTMCell_for_rnn/BasicLSTMUnit_0.b_0",
                "basic_lstm_layers_0/BasicLSTMUnit_0.b_0",
            ],
        ]

        for names in param_names:
            param = np.array(
                fluid.global_scope().find_var(names[0]).get_tensor()
            )
            param = np.random.uniform(-0.1, 0.1, size=param.shape).astype(
                'float32'
            )
            fluid.global_scope().find_var(names[0]).get_tensor().set(
                param, place
            )
            fluid.global_scope().find_var(names[1]).get_tensor().set(
                param, place
            )

        out = exe.run(
            feed={
                'inputs_basic_lstm': inputs_basic_lstm_np,
                'sequence_length': sequence_length_np,
                'inputs': inputs_np,
                'pre_hidden': pre_hidden_np,
                'pre_cell': pre_cell_np,
            },
            fetch_list=[output_new, rnn_out],
        )

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


if __name__ == '__main__':
    unittest.main()
