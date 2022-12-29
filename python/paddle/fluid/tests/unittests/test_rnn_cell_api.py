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
from rnn.rnn_numpy import LSTMCell
from rnn.rnn_numpy import rnn as numpy_rnn

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.layers.utils as utils
from paddle.fluid import framework
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, program_guard
from paddle.nn.layer.rnn import rnn as dynamic_rnn

paddle.enable_static()


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

        numpy_cell = LSTMCell(self.input_size, self.hidden_size)
        dynamic_cell = paddle.nn.LSTMCell(self.input_size, self.hidden_size)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        state = numpy_cell.parameters
        for k, v in dynamic_cell.named_parameters():
            param = np.random.uniform(-0.1, 0.1, size=state[k].shape).astype(
                'float64'
            )
            setattr(numpy_cell, k, param)
            fluid.global_scope().find_var(v.name).get_tensor().set(param, place)

        sequence_length = fluid.data(
            name="sequence_length", shape=[None], dtype='int64'
        )
        inputs_rnn = fluid.data(
            name='inputs_rnn',
            shape=[None, None, self.input_size],
            dtype='float64',
        )
        pre_hidden = fluid.data(
            name='pre_hidden', shape=[None, self.hidden_size], dtype='float64'
        )
        pre_cell = fluid.data(
            name='pre_cell', shape=[None, self.hidden_size], dtype='float64'
        )

        dynamic_output, dynamic_final_state = dynamic_rnn(
            cell=dynamic_cell,
            inputs=inputs_rnn,
            sequence_length=sequence_length,
            initial_states=(pre_hidden, pre_cell),
            is_reverse=False,
        )

        inputs_rnn_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.seq_len, self.input_size)
        ).astype('float64')
        sequence_length_np = (
            np.ones(self.batch_size, dtype='int64') * self.seq_len
        )
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)
        ).astype('float64')
        pre_cell_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)
        ).astype('float64')

        o1, _ = numpy_rnn(
            cell=numpy_cell,
            inputs=inputs_rnn_np,
            initial_states=(pre_hidden_np, pre_cell_np),
            sequence_length=sequence_length_np,
            is_reverse=False,
        )

        o2 = exe.run(
            feed={
                'inputs_rnn': inputs_rnn_np,
                'sequence_length': sequence_length_np,
                'pre_hidden': pre_hidden_np,
                'pre_cell': pre_cell_np,
            },
            fetch_list=[dynamic_output],
        )[0]
        np.testing.assert_allclose(o1, o2, rtol=0.001)


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
