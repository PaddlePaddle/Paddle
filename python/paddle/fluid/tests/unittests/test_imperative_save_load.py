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

import os
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.optimizer import Adam
from paddle.nn import Embedding


class SimpleLSTMRNN(fluid.Layer):
    def __init__(
        self, hidden_size, num_steps, num_layers=2, init_scale=0.1, dropout=None
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []
        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale
                    )
                ),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale
                ),
            )
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale
                    )
                ),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0),
            )
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []

        for i in range(self._num_layers):
            pre_hidden = paddle.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1]
            )
            pre_cell = paddle.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1]
            )
            pre_hidden = paddle.reshape(
                pre_hidden, shape=[-1, self._hidden_size]
            )
            pre_cell = paddle.reshape(pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        res = []
        for index in range(self._num_steps):
            self._input = paddle.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1]
            )
            self._input = paddle.reshape(
                self._input, shape=[-1, self._hidden_size]
            )
            for k in range(self._num_layers):
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                gate_input = paddle.matmul(x=nn, y=weight_1)

                gate_input = paddle.add(gate_input, bias)
                i, j, f, o = paddle.split(
                    gate_input, num_or_sections=4, axis=-1
                )
                c = pre_cell * paddle.nn.functional.sigmoid(
                    f
                ) + paddle.nn.functional.sigmoid(i) * paddle.tanh(j)
                m = paddle.tanh(c) * paddle.nn.functional.sigmoid(o)
                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m

                if self._dropout is not None and self._dropout > 0.0:
                    self._input = paddle.nn.functional.dropout(
                        self._input,
                        p=self._dropout,
                        mode='upscale_in_train',
                    )
            res.append(
                paddle.reshape(self._input, shape=[1, -1, self._hidden_size])
            )
        real_res = fluid.layers.concat(res, 0)
        real_res = paddle.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = paddle.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size]
        )
        last_hidden = paddle.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = paddle.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size]
        )
        last_cell = paddle.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(fluid.Layer):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        num_layers=2,
        num_steps=20,
        init_scale=0.1,
        dropout=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout,
        )
        self.embedding = Embedding(
            vocab_size,
            hidden_size,
            sparse=False,
            weight_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale
                ),
            ),
        )

        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale
            ),
        )
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale
            ),
        )

    def forward(self, input, label, init_hidden, init_cell):
        init_h = paddle.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size]
        )

        init_c = paddle.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size]
        )

        x_emb = self.embedding(input)
        x_emb = paddle.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size]
        )
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = paddle.nn.functional.dropout(
                x_emb,
                p=self.drop_out,
                mode='upscale_in_train',
            )
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(
            x_emb, init_h, init_c
        )
        rnn_out = paddle.reshape(
            rnn_out, shape=[-1, self.num_steps, self.hidden_size]
        )

        projection = paddle.matmul(rnn_out, self.softmax_weight)
        projection = paddle.add(projection, self.softmax_bias)
        projection = paddle.reshape(projection, shape=[-1, self.vocab_size])
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False
        )
        loss = paddle.reshape(loss, shape=[-1, self.num_steps])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)

        return loss, last_hidden, last_cell


class TestDygraphPtbRnn(unittest.TestCase):
    def func_setUp(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            bd = []
            lr_arr = [1.0]
            # this a fake lr decay strategy
            for i in range(1, 10):
                bd.append(100 * i)
                new_lr = 1.0
                lr_arr.append(new_lr)

            place = (
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=bd, values=lr_arr
                ),
                parameter_list=ptb_model.parameters(),
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            self.opti_dict = adam.state_dict()
            self.base_opti = {}
            for k, v in self.opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    self.base_opti[v.name] = v.numpy()
                    self.assertTrue(np.sum(np.abs(v.numpy())) != 0)
                else:
                    self.base_opti[k] = v

            fluid.save_dygraph(self.opti_dict, "./test_dy")

            self.state_dict = ptb_model.state_dict()

            self.model_base = {}
            for k, v in self.state_dict.items():
                np_t = v.numpy()
                self.model_base[k] = np_t

            fluid.save_dygraph(self.state_dict, "./test_dy")

    def func_testLoadAndSetVarBase(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            bd = []
            lr_arr = [1.0]
            # this a fake lr decay strategy
            for i in range(1, 10):
                bd.append(100 * i)
                new_lr = 1.0
                lr_arr.append(new_lr)

            place = (
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=bd, values=lr_arr
                ),
                parameter_list=ptb_model.parameters(),
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            opti_dict = adam.state_dict()
            # set to zero
            for k, v in opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    np_t = v.numpy()
                    var = v.value().get_tensor()
                    var.set(np.zeros_like(np_t), place)

                    self.assertTrue(np.sum(np.abs(v.numpy())) == 0)

            if isinstance(adam._learning_rate, LearningRateDecay):
                adam._learning_rate.step_num = 0

            para_state_dict, opti_state_dict = fluid.load_dygraph("./test_dy")
            print(opti_state_dict.keys())
            adam.set_state_dict(opti_state_dict)

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name]
                    )
                else:
                    self.assertEqual(v, self.base_opti[k])

            # check parameter
            state_dict = ptb_model.state_dict()
            for k, v in state_dict.items():
                np_t = v.numpy()
                var = v.value().get_tensor()

                var.set(np.zeros_like(np_t), place)

            ptb_model.set_state_dict(stat_dict=para_state_dict)

            state_dict = ptb_model.state_dict()

            for k, v in state_dict.items():
                new_t = v.numpy()

                base_t = self.model_base[k]

                np.testing.assert_array_equal(new_t, base_t)

    def func_testSetVariable(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            bd = []
            lr_arr = [1.0]
            # this a fake lr decay strategy
            for i in range(1, 10):
                bd.append(100 * i)
                new_lr = 1.0
                lr_arr.append(new_lr)

            place = (
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=bd, values=lr_arr
                ),
                parameter_list=ptb_model.parameters(),
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            opti_dict = adam.state_dict()
            # set to zero
            for k, v in opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    np_t = v.numpy()
                    var = v.value().get_tensor()
                    var.set(np.zeros_like(np_t), place)

                    self.assertTrue(np.sum(np.abs(v.numpy())) == 0)

            if isinstance(adam._learning_rate, LearningRateDecay):
                adam._learning_rate.step_num = 0

            adam.set_state_dict(self.opti_dict)
            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name]
                    )
                else:
                    self.assertEqual(v, self.base_opti[k])

            # check parameter
            state_dict = ptb_model.state_dict()
            for k, v in state_dict.items():
                np_t = v.numpy()
                var = v.value().get_tensor()

                var.set(np.zeros_like(np_t), place)

            ptb_model.set_state_dict(self.state_dict)

            state_dict = ptb_model.state_dict()

            for k, v in state_dict.items():
                new_t = v.numpy()

                base_t = self.model_base[k]

                np.testing.assert_array_equal(new_t, base_t)

    def func_testSetNumpy(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            bd = []
            lr_arr = [1.0]
            # this a fake lr decay strategy
            for i in range(1, 10):
                bd.append(100 * i)
                new_lr = 1.0
                lr_arr.append(new_lr)

            place = (
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=bd, values=lr_arr
                ),
                parameter_list=ptb_model.parameters(),
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            opti_dict = adam.state_dict()
            np_opti_dict = {}
            # set to zero
            for k, v in opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    np_t = v.numpy()
                    np_opti_dict[v.name] = np_t
                    var = v.value().get_tensor()
                    var.set(np.zeros_like(np_t), place)
                    self.assertTrue(np.sum(np.abs(v.numpy())) == 0)
                else:
                    np_opti_dict[k] = v

            if isinstance(adam._learning_rate, LearningRateDecay):
                adam._learning_rate.step_num = 0

            adam.set_state_dict(np_opti_dict)

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name]
                    )
                else:
                    self.assertEqual(v, self.base_opti[k])

            # check parameter
            state_dict = ptb_model.state_dict()
            np_state_dict = {}
            for k, v in state_dict.items():
                np_t = v.numpy()
                np_state_dict[k] = np_t
                var = v.value().get_tensor()

                var.set(np.zeros_like(np_t), place)

            ptb_model.set_state_dict(np_state_dict)

            state_dict = ptb_model.state_dict()

            for k, v in state_dict.items():
                new_t = v.numpy()

                base_t = self.model_base[k]

                np.testing.assert_array_equal(new_t, base_t)

    def func_testSetVariableBeforeTrain(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = (
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=0.0,
                beta1=0.8,
                beta2=0.6,
                parameter_list=ptb_model.parameters(),
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            adam.set_state_dict(self.opti_dict)
            ptb_model.set_state_dict(self.state_dict)

            for i in range(1):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )

                dy_loss.backward()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if k == "global_step":
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] + 1
                    )

                if k.find("beta1_pow_acc_0") > 0:
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] * adam._beta1
                    )
                if k.find("beta2_pow_acc_0") > 0:
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] * adam._beta2
                    )

            state_dict = ptb_model.state_dict()

            for k, v in state_dict.items():
                new_t = v.numpy()

                base_t = self.model_base[k]
                np.testing.assert_array_equal(new_t, base_t)

    def func_testLoadAndSetVarBaseBeforeTrain(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            bd = []
            lr_arr = [0.0]
            # this a fake lr decay strategy
            for i in range(1, 10):
                bd.append(100 * i)
                # set lr to zero not update parameter
                new_lr = 0.0
                lr_arr.append(new_lr)

            place = (
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=0.0,
                beta1=0.8,
                beta2=0.6,
                parameter_list=ptb_model.parameters(),
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            state_dict, opti_dict = fluid.load_dygraph("./test_dy")
            adam.set_state_dict(opti_dict)
            ptb_model.set_state_dict(state_dict)

            for i in range(1):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )

                dy_loss.backward()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if k == "global_step":
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] + 1
                    )

                if k.find("beta1_pow_acc_0") > 0:
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] * adam._beta1
                    )
                if k.find("beta2_pow_acc_0") > 0:
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] * adam._beta2
                    )

            # check parameter

            state_dict = ptb_model.state_dict()

            for k, v in state_dict.items():
                new_t = v.numpy()

                base_t = self.model_base[k]
                np.testing.assert_array_equal(new_t, base_t)

    def func_testSetNumpyBeforeTrain(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            # TODO: marsyang1993 Change seed to

            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            bd = []
            lr_arr = [0.0]
            # this a fake lr decay strategy
            for i in range(1, 10):
                bd.append(100 * i)
                # set lr to 0.0, not update parameter
                new_lr = 0.0
                lr_arr.append(new_lr)

            place = (
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=bd, values=lr_arr
                ),
                beta1=0.8,
                beta2=0.6,
                parameter_list=ptb_model.parameters(),
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            np_opti_dict = {}
            np_state_dict = {}

            for k, v in self.opti_dict.items():
                if isinstance(v, (core.VarBase, core.eager.Tensor)):
                    np_opti_dict[v.name] = v.numpy()
                else:
                    np_opti_dict[k] = v

            for k, v in self.state_dict.items():
                np_state_dict[k] = v.numpy()

            adam.set_state_dict(np_opti_dict)
            ptb_model.set_state_dict(np_state_dict)
            for i in range(1):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )

                dy_loss.backward()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if k == "global_step":
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] + 1
                    )

                if k.find("beta1_pow_acc_0") > 0:
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] * adam._beta1
                    )
                if k.find("beta2_pow_acc_0") > 0:
                    np.testing.assert_array_equal(
                        v.numpy(), self.base_opti[v.name] * adam._beta2
                    )

            # check parameter

            state_dict = ptb_model.state_dict()

            for k, v in state_dict.items():
                new_t = v.numpy()

                base_t = self.model_base[k]
                np.testing.assert_array_equal(new_t, base_t)

    def func_testOnlyLoadParams(self):
        with fluid.dygraph.guard():
            emb = paddle.nn.Embedding(10, 10)
            state_dict = emb.state_dict()
            fluid.save_dygraph(state_dict, os.path.join('saved_dy', 'emb_dy'))

            para_state_dict, opti_state_dict = fluid.load_dygraph(
                os.path.join('saved_dy', 'emb_dy')
            )

            self.assertIsNone(opti_state_dict)

            para_state_dict, opti_state_dict = fluid.load_dygraph(
                os.path.join('saved_dy', 'emb_dy.pdparams')
            )

            para_state_dict, opti_state_dict = fluid.load_dygraph(
                os.path.join('saved_dy', 'emb_dy.pdopt')
            )

    def func_test_load_compatible_with_keep_name_table(self):
        with fluid.dygraph.guard():
            emb = paddle.nn.Embedding(10, 10)
            state_dict = emb.state_dict()
            fluid.save_dygraph(state_dict, os.path.join('saved_dy', 'emb_dy'))

            para_state_dict, opti_state_dict = fluid.load_dygraph(
                os.path.join('saved_dy', 'emb_dy'), keep_name_table=True
            )
            self.assertIsNotNone(para_state_dict)
            self.assertIsNone(opti_state_dict)

    def test_main(self):
        self.func_setUp()
        self.func_testLoadAndSetVarBase()
        self.func_testSetVariable()
        self.func_testSetNumpy()
        self.func_testSetVariableBeforeTrain()
        self.func_testLoadAndSetVarBaseBeforeTrain()
        self.func_testSetNumpyBeforeTrain()
        self.func_testOnlyLoadParams()
        self.func_test_load_compatible_with_keep_name_table()
        with _test_eager_guard():
            self.func_setUp()
            self.func_testLoadAndSetVarBase()
            self.func_testSetVariable()
            self.func_testSetNumpy()
            self.func_testSetVariableBeforeTrain()
            self.func_testLoadAndSetVarBaseBeforeTrain()
            self.func_testSetNumpyBeforeTrain()
            self.func_testOnlyLoadParams()
            self.func_test_load_compatible_with_keep_name_table()


if __name__ == '__main__':
    unittest.main()
