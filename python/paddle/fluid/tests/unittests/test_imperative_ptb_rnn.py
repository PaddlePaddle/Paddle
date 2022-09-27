#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import TracedLayer
from test_imperative_base import new_program_scope
import numpy as np
import six
from utils import DyGraphProgramDescTracerTestHelper, is_equal_program
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph


class SimpleLSTMRNN(fluid.Layer):

    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []
        self._create_parameter()

    def _create_parameter(self):
        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []

        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(init_hidden,
                                            axes=[0],
                                            starts=[i],
                                            ends=[i + 1])
            pre_cell = fluid.layers.slice(init_cell,
                                          axes=[0],
                                          starts=[i],
                                          ends=[i + 1])
            pre_hidden = fluid.layers.reshape(pre_hidden,
                                              shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(pre_cell,
                                            shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        res = []
        for index in range(self._num_steps):
            self._input = fluid.layers.slice(input_embedding,
                                             axes=[1],
                                             starts=[index],
                                             ends=[index + 1])
            self._input = fluid.layers.reshape(self._input,
                                               shape=[-1, self._hidden_size])
            for k in range(self._num_layers):
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                i, j, f, o = fluid.layers.split(gate_input,
                                                num_or_sections=4,
                                                dim=-1)
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m

                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
            res.append(
                fluid.layers.reshape(self._input,
                                     shape=[1, -1, self._hidden_size]))
        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(fluid.Layer):

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 is_sparse=False,
                 dropout=None):
        super(PtbModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(hidden_size,
                                             num_steps,
                                             num_layers=num_layers,
                                             init_scale=init_scale,
                                             dropout=dropout)
        self.embedding = Embedding(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label, init_hidden, init_cell):
        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)
        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.drop_out,
                dropout_implementation='upscale_in_train')
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(
            x_emb, init_h, init_c)
        rnn_out = fluid.layers.reshape(
            rnn_out, shape=[-1, self.num_steps, self.hidden_size])
        projection = fluid.layers.matmul(rnn_out, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(projection,
                                          shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(logits=projection,
                                                       label=label,
                                                       soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss, last_hidden, last_cell


class TestDygraphPtbRnn(unittest.TestCase):

    def func_test_ptb_rnn(self):
        for is_sparse in [True, False]:
            self.ptb_rnn_cpu_float32(is_sparse)

    def test_ptb_rnn(self):
        with _test_eager_guard():
            self.func_test_ptb_rnn()
        self.func_test_ptb_rnn()

    def ptb_rnn_cpu_float32(self, is_sparse):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200
        traced_layer = None

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(hidden_size=hidden_size,
                                 vocab_size=vocab_size,
                                 num_layers=num_layers,
                                 num_steps=num_steps,
                                 init_scale=init_scale,
                                 is_sparse=is_sparse)

            sgd = SGDOptimizer(learning_rate=1e-3,
                               parameter_list=ptb_model.parameters())
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            helper = DyGraphProgramDescTracerTestHelper(self)
            program = None

            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros((num_layers, batch_size, hidden_size),
                                          dtype='float32')
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                if i % 5 == 0 and _in_legacy_dygraph():
                    outs, traced_layer = TracedLayer.trace(
                        ptb_model, [x, y, init_hidden, init_cell])
                    outs_static = traced_layer([x, y, init_hidden, init_cell])
                    helper.assertEachVar(outs, outs_static)

                    if program is not None:
                        self.assertTrue(
                            is_equal_program(traced_layer.program, program))

                    program = traced_layer.program

                    traced_layer.save_inference_model(
                        './infe_imperative_ptb_rnn', feed=list(range(4)))
                else:
                    outs = ptb_model(x, y, init_hidden, init_cell)

                dy_loss, last_hidden, last_cell = outs

                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                sgd.minimize(dy_loss)
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            dy_loss_value = dy_loss.numpy()
            dy_last_cell_value = last_cell.numpy()
            dy_last_hidden_value = last_hidden.numpy()

        with new_program_scope():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            ptb_model = PtbModel(hidden_size=hidden_size,
                                 vocab_size=vocab_size,
                                 num_layers=num_layers,
                                 num_steps=num_steps,
                                 init_scale=init_scale,
                                 is_sparse=is_sparse)

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            sgd = SGDOptimizer(learning_rate=1e-3)
            x = fluid.layers.data(name="x",
                                  shape=[-1, num_steps],
                                  dtype='int64')
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(name="init_hidden",
                                            shape=[1],
                                            dtype='float32')
            init_cell = fluid.layers.data(name="init_cell",
                                          shape=[1],
                                          dtype='float32')

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell)
            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()
            static_param_name_list = list()
            for param in ptb_model.parameters():
                static_param_name_list.append(param.name)

            out = exe.run(framework.default_startup_program(),
                          fetch_list=static_param_name_list)
            for i in range(len(static_param_name_list)):
                static_param_init[static_param_name_list[i]] = out[i]
            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros((num_layers, batch_size, hidden_size),
                                          dtype='float32')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                fetch_list.extend(static_param_name_list)
                out = exe.run(fluid.default_main_program(),
                              feed={
                                  "x": x_data,
                                  "y": y_data,
                                  "init_hidden": init_hidden_data,
                                  "init_cell": init_cell_data
                              },
                              fetch_list=fetch_list)
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

                if i == batch_num - 1:
                    for k in range(3, len(out)):
                        static_param_updated[static_param_name_list[k -
                                                                    3]] = out[k]

        np.testing.assert_array_equal(static_loss_value, dy_loss_value)
        np.testing.assert_array_equal(static_last_cell_value,
                                      dy_last_cell_value)
        np.testing.assert_array_equal(static_last_hidden_value,
                                      dy_last_hidden_value)
        for key, value in six.iteritems(static_param_init):
            np.testing.assert_array_equal(value, dy_param_init[key])
        for key, value in six.iteritems(static_param_updated):
            np.testing.assert_array_equal(value, dy_param_updated[key])


if __name__ == '__main__':
    unittest.main()
