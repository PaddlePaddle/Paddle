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
import tempfile
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.nn import Embedding
from paddle.optimizer import Adam
from paddle.optimizer.lr import LRScheduler


class SimpleLSTMRNN(paddle.nn.Layer):
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
                attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(
                        low=-self._init_scale, high=self._init_scale
                    )
                ),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Uniform(
                    low=-self._init_scale, high=self._init_scale
                ),
            )
            self.weight_1_arr.append(self.add_parameter(f'w_{i}', weight_1))
            bias_1 = self.create_parameter(
                attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(
                        low=-self._init_scale, high=self._init_scale
                    )
                ),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            self.bias_arr.append(self.add_parameter(f'b_{i}', bias_1))

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

                nn = paddle.concat([self._input, pre_hidden], 1)
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
        real_res = paddle.concat(res, 0)
        real_res = paddle.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = paddle.concat(self.hidden_array, 1)
        last_hidden = paddle.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size]
        )
        last_hidden = paddle.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = paddle.concat(self.cell_array, 1)
        last_cell = paddle.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size]
        )
        last_cell = paddle.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(paddle.nn.Layer):
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
            weight_attr=base.ParamAttr(
                name='embedding_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale
                ),
            ),
        )

        self.softmax_weight = self.create_parameter(
            attr=base.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale
            ),
        )
        self.softmax_bias = self.create_parameter(
            attr=base.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(
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
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def func_setUp(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with base.dygraph.guard():
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
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            scheduler = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=bd, values=lr_arr
            )
            adam = Adam(
                learning_rate=scheduler, parameters=ptb_model.parameters()
            )
            dy_param_updated = {}
            dy_param_init = {}
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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                scheduler.step()
                ptb_model.clear_gradients()

                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            self.opti_dict = adam.state_dict()
            self.base_opti = {}
            for k, v in self.opti_dict.items():
                if isinstance(v, core.eager.Tensor):
                    self.base_opti[v.name] = v.numpy()
                    self.assertTrue(np.sum(np.abs(v.numpy())) != 0)
                else:
                    self.base_opti[k] = v

            paddle.save(
                self.opti_dict,
                os.path.join(self.temp_dir.name, "test_dy_v2.pdopt"),
            )

            self.state_dict = ptb_model.state_dict()

            self.model_base = {}
            for k, v in self.state_dict.items():
                np_t = v.numpy()
                self.model_base[k] = np_t

            paddle.save(
                self.state_dict,
                os.path.join(self.temp_dir.name, "test_dy_v2.pdparams"),
            )

    def func_testLoadAndSetVarBase(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with base.dygraph.guard():
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
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            scheduler = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=bd, values=lr_arr
            )
            adam = Adam(
                learning_rate=scheduler, parameters=ptb_model.parameters()
            )
            dy_param_updated = {}
            dy_param_init = {}
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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                scheduler.step()
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            opti_dict = adam.state_dict()
            # set to zero
            for k, v in opti_dict.items():
                if isinstance(v, core.eager.Tensor):
                    np_t = v.numpy()
                    var = v.value().get_tensor()
                    var.set(np.zeros_like(np_t), place)

                    self.assertTrue(np.sum(np.abs(v.numpy())) == 0)

            para_state_dict = paddle.load(
                os.path.join(self.temp_dir.name, "test_dy_v2.pdparams")
            )
            opti_state_dict = paddle.load(
                os.path.join(self.temp_dir.name, "test_dy_v2.pdopt")
            )
            adam.set_state_dict(opti_state_dict)

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if isinstance(v, core.eager.Tensor):
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

            ptb_model.set_dict(para_state_dict)

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

        with base.dygraph.guard():
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
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            scheduler = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=bd, values=lr_arr
            )
            adam = Adam(
                learning_rate=scheduler, parameters=ptb_model.parameters()
            )
            dy_param_updated = {}
            dy_param_init = {}
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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                scheduler.step()
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            opti_dict = adam.state_dict()
            # set to zero
            for k, v in opti_dict.items():
                if isinstance(v, core.eager.Tensor):
                    np_t = v.numpy()
                    var = v.value().get_tensor()
                    var.set(np.zeros_like(np_t), place)

                    self.assertTrue(np.sum(np.abs(v.numpy())) == 0)

            if isinstance(adam._learning_rate, LRScheduler):
                adam._learning_rate.step_num = 0

            adam.set_state_dict(self.opti_dict)
            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if isinstance(v, core.eager.Tensor):
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

            ptb_model.set_dict(self.state_dict)

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

        with base.dygraph.guard():
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
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            scheduler = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=bd, values=lr_arr
            )
            adam = Adam(
                learning_rate=scheduler, parameters=ptb_model.parameters()
            )
            dy_param_updated = {}
            dy_param_init = {}
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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                adam.minimize(dy_loss)
                scheduler.step()
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            # check optimizer
            opti_dict = adam.state_dict()
            np_opti_dict = {}
            # set to zero
            for k, v in opti_dict.items():
                if isinstance(v, core.eager.Tensor):
                    np_t = v.numpy()
                    np_opti_dict[v.name] = np_t
                    var = v.value().get_tensor()
                    var.set(np.zeros_like(np_t), place)
                    self.assertTrue(np.sum(np.abs(v.numpy())) == 0)
                else:
                    np_opti_dict[k] = v

            if isinstance(adam._learning_rate, LRScheduler):
                adam._learning_rate.step_num = 0

            adam.set_state_dict(np_opti_dict)

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if isinstance(v, core.eager.Tensor):
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

            ptb_model.set_dict(np_state_dict)

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

        with base.dygraph.guard():
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

            place = (
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=0.0,
                beta1=0.8,
                beta2=0.6,
                parameters=ptb_model.parameters(),
            )
            dy_param_updated = {}
            dy_param_init = {}
            dy_loss = None
            last_hidden = None
            last_cell = None

            adam.set_state_dict(self.opti_dict)
            ptb_model.set_dict(self.state_dict)

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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
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

        with base.dygraph.guard():
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
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            adam = Adam(
                learning_rate=0.0,
                beta1=0.8,
                beta2=0.6,
                parameters=ptb_model.parameters(),
            )
            dy_param_updated = {}
            dy_param_init = {}
            dy_loss = None
            last_hidden = None
            last_cell = None

            model_prefix = os.path.join(self.temp_dir.name, "test_dy_v2")
            state_dict = paddle.load(model_prefix + '.pdparams')
            opti_dict = paddle.load(model_prefix + '.pdopt')
            adam.set_state_dict(opti_dict)
            ptb_model.set_dict(state_dict)

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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
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

        with base.dygraph.guard():
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
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            scheduler = paddle.optimizer.lr.PiecewiseDecay(
                boundaries=bd, values=lr_arr
            )
            adam = Adam(
                learning_rate=scheduler,
                beta1=0.8,
                beta2=0.6,
                parameters=ptb_model.parameters(),
            )
            dy_param_updated = {}
            dy_param_init = {}
            dy_loss = None
            last_hidden = None
            last_cell = None

            np_opti_dict = {}
            np_state_dict = {}

            for k, v in self.opti_dict.items():
                if isinstance(v, core.eager.Tensor):
                    np_opti_dict[v.name] = v.numpy()
                else:
                    np_opti_dict[k] = v

            for k, v in self.state_dict.items():
                np_state_dict[k] = v.numpy()

            adam.set_state_dict(np_opti_dict)
            ptb_model.set_dict(np_state_dict)
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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )

                dy_loss.backward()
                scheduler.step()
                adam.minimize(dy_loss)
                ptb_model.clear_gradients()

            opti_dict = adam.state_dict()
            for k, v in opti_dict.items():
                if k == "LR_Scheduler":
                    np.testing.assert_array_equal(
                        v['last_epoch'], self.base_opti[k]['last_epoch'] + 1
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
        with base.dygraph.guard():
            emb = paddle.nn.Embedding(10, 10)
            state_dict = emb.state_dict()
            paddle.save(
                state_dict,
                os.path.join(self.temp_dir.name, 'saved_dy', 'emb_dy.pdparams'),
            )

            para_state_dict = paddle.load(
                os.path.join(self.temp_dir.name, 'saved_dy', 'emb_dy.pdparams')
            )

    def func_test_no_state_in_input_dict(self):
        with base.dygraph.guard():
            emb = paddle.nn.Embedding(10, 10)
            state_dict = emb.state_dict()
            paddle.save(
                state_dict,
                os.path.join(self.temp_dir.name, 'saved_dy', 'emb_dy.pdparams'),
            )

            para_state_dict = paddle.load(
                os.path.join(self.temp_dir.name, 'saved_dy', 'emb_dy.pdparams')
            )
            para_state_dict.pop('weight')

            emb.set_state_dict(para_state_dict)

    def func_test_state_shape_mismatch(self):
        with base.dygraph.guard():
            emb = paddle.nn.Embedding(10, 10)
            state_dict = emb.state_dict()
            paddle.save(
                state_dict,
                os.path.join(self.temp_dir.name, 'saved_dy', 'emb_dy.pdparams'),
            )

            para_state_dict = paddle.load(
                os.path.join(self.temp_dir.name, 'saved_dy', 'emb_dy.pdparams'),
                return_numpy=True,
            )
            para_state_dict['weight'] = np.expand_dims(
                para_state_dict['weight'], axis=-1
            )

            emb.set_state_dict(para_state_dict)

    def test_main(self):
        self.func_setUp()
        self.func_testLoadAndSetVarBase()
        self.func_testSetVariable()
        self.func_testSetNumpy()
        self.func_testSetVariableBeforeTrain()
        self.func_testLoadAndSetVarBaseBeforeTrain()
        self.func_testSetNumpyBeforeTrain()
        self.func_testOnlyLoadParams()
        self.func_test_no_state_in_input_dict()
        self.func_test_state_shape_mismatch()


if __name__ == '__main__':
    unittest.main()
