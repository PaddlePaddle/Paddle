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


import errno
import os
import pickle
import tempfile
import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base
from paddle.base import core, framework
from paddle.optimizer import Adam

paddle.enable_static()


class SimpleLSTMRNN(paddle.nn.Layer):
    def __init__(
        self,
        name_scope,
        hidden_size,
        num_steps,
        num_layers=2,
        init_scale=0.1,
        dropout=None,
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
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
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
        name_scope,
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
            self.full_name(),
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout,
        )
        self.embedding = paddle.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
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

        # NPU 'tok_k' kernel only support `int32` dtype, so cast `input` from `int64` to `int32`.
        input = paddle.cast(input, "int32")
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


class TestSaveLoadBase(unittest.TestCase):
    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200
        temp_dir = tempfile.TemporaryDirectory()

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )
            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            paddle.static.save(
                main_program, os.path.join(temp_dir.name, "test_1")
            )

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.load(
                main_program,
                os.path.join(temp_dir.name, "test_1.pdparams"),
                exe,
            )

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            temp_dir.cleanup()


class TestSaveLoadPartial(unittest.TestCase):
    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200
        temp_dir = tempfile.TemporaryDirectory()

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )

            test_program = base.default_main_program().clone(for_test=True)

            add_1 = paddle.static.nn.fc(
                static_last_hidden,
                size=hidden_size,
                num_flatten_dims=2,
                bias_attr=False,
            )

            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            paddle.static.save(
                main_program, os.path.join(temp_dir.name, "test_1")
            )

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.load(
                test_program, os.path.join(temp_dir.name, "test_1.pdopt"), None
            )

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            paddle.static.load(
                test_program,
                os.path.join(temp_dir.name, "test_1.pdmodel"),
                None,
            )
            temp_dir.cleanup()


class TestSaveLoadSetStateDict(unittest.TestCase):
    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200
        temp_dir = tempfile.TemporaryDirectory()

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )
            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            paddle.static.save(
                main_program, os.path.join(temp_dir.name, "test_1")
            )

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.load(
                main_program, os.path.join(temp_dir.name, "test_1"), exe
            )

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            temp_dir.cleanup()


class TestProgramStatePartial(unittest.TestCase):
    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200
        temp_dir = tempfile.TemporaryDirectory()

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )

            test_program = base.default_main_program().clone(for_test=True)

            add_1 = paddle.static.nn.fc(
                static_last_hidden,
                size=hidden_size,
                num_flatten_dims=2,
                bias_attr=False,
            )

            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            paddle.static.save(
                main_program, os.path.join(temp_dir.name, 'test_1')
            )

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            # base.load(test_program, "./test_1", None )
            program_state = paddle.static.load_program_state(
                os.path.join(temp_dir.name, 'test_1')
            )

            program_state_1 = paddle.static.load_program_state(
                os.path.join(temp_dir.name, 'test_1.pdparams')
            )

            program_state_2 = paddle.static.load_program_state(
                os.path.join(temp_dir.name, 'test_1.pdopt')
            )

            program_state_3 = paddle.static.load_program_state(
                os.path.join(temp_dir.name, 'test_1.pdmodel')
            )

            paddle.static.set_program_state(test_program, program_state)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

            # check 1
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.set_program_state(test_program, program_state_1)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

            # check 2
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.set_program_state(test_program, program_state_2)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

            # check 3
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.set_program_state(test_program, program_state_3)

            for var in test_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            temp_dir.cleanup()


class TestVariableInit(unittest.TestCase):
    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_variable_init(self):
        x = paddle.static.data(name="x", shape=[10, 10], dtype='float32')
        y = paddle.static.nn.fc(x, 10)
        z = paddle.static.nn.fc(y, 10)

        place = self.set_place()
        exe = base.Executor(place)
        exe.run(base.default_startup_program())

        temp_dir = tempfile.TemporaryDirectory()
        paddle.static.save(
            base.default_main_program(),
            os.path.join(temp_dir.name, "test_path"),
        )

        def set_var(var, ndarray):
            t = var.get_tensor()
            p = t._place()
            if p.is_cpu_place():
                place = paddle.base.CPUPlace()
            elif p.is_cuda_pinned_place():
                place = paddle.base.CUDAPinnedPlace()
            else:
                p = paddle.base.core.Place()
                p.set_place(t._place())
                place = paddle.base.CUDAPlace(p.gpu_device_id())

            t.set(ndarray, place)

        program = base.default_main_program()
        new_scope = base.core.Scope()

        place = self.set_place()
        exe = base.Executor(place)
        parameter_list = list(
            filter(paddle.framework.is_parameter, program.list_vars())
        )

        base.core._create_loaded_parameter(
            parameter_list, new_scope, exe._default_executor
        )
        parameter_file_name = os.path.join(temp_dir.name, "test_path.pdparams")
        with open(parameter_file_name, 'rb') as f:
            load_dict = pickle.load(f)

        for v in parameter_list:
            assert (
                v.name in load_dict
            ), f"Can not find [{v.name}] in model file [{parameter_file_name}]"
            new_v = new_scope.find_var(v.name)
            set_var(new_v, load_dict[v.name])

        opt_list = list(
            filter(
                paddle.framework.io_utils.is_belong_to_optimizer,
                program.list_vars(),
            )
        )

        base.core._create_loaded_parameter(
            opt_list, new_scope, exe._default_executor
        )
        opt_file_name = os.path.join(temp_dir.name, "test_path.pdopt")
        with open(opt_file_name, 'rb') as f:
            load_dict = pickle.load(f)

        for v in opt_list:
            assert (
                v.name in load_dict
            ), f"Can not find [{v.name}] in model file [{opt_file_name}]"

            new_v = new_scope.find_var(v.name)
            set_var(new_v, load_dict[v.name])

        base_map = {}
        for var in program.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                t = np.array(
                    base.global_scope().find_var(var.name).get_tensor()
                )
                # make sure all the paramerter or optimizer var have been update
                base_map[var.name] = t

        for var in program.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                new_t = np.array(new_scope.find_var(var.name).get_tensor())
                base_t = base_map[var.name]

                np.testing.assert_array_equal(new_t, base_t)
        temp_dir.cleanup()


class TestLoadFromOldInterface(unittest.TestCase):
    def setUp(self):
        if os.path.exists("test_path.pdparams"):
            os.remove("test_path.pdparams")

        if os.path.exists("test_static_load_var_list.pdparams"):
            os.remove("test_static_load_var_list.pdparams")

        self.temp_dir = tempfile.TemporaryDirectory()

    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_from_old_interface(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )

            test_clone_program = base.default_main_program().clone()
            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            # base.save(main_program, "./test_1")
            paddle.distributed.io.save_persistables(
                exe, os.path.join(self.temp_dir.name, "test_path"), main_program
            )

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.load(
                main_program, os.path.join(self.temp_dir.name, "test_path"), exe
            )

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    old_shape = np.array(ten).shape
                    new_shape = [e + 10 for e in old_shape]

                    var.desc.set_shape(new_shape)
            with self.assertRaises(RuntimeError):
                paddle.static.load(
                    main_program,
                    os.path.join(self.temp_dir.name, "test_path"),
                    exe,
                )

            # check unused parameter

            paddle.static.load(
                test_clone_program,
                os.path.join(self.temp_dir.name, "test_path"),
                exe,
            )

    def test_load_from_old_interface_var_list(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)
            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )

            test_clone_program = base.default_main_program().clone()
            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            # base.save(main_program, "./test_1")
            paddle.distributed.io.save_persistables(
                exe,
                os.path.join(self.temp_dir.name, "test_static_load_var_list"),
                main_program,
            )

            # set var to zero
            var_list = []
            for i, var in enumerate(main_program.list_vars()):
                if isinstance(var, framework.Parameter) or var.persistable:
                    if i % 2 == 0:
                        var_list.append(var)
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.static.load(
                main_program,
                os.path.join(self.temp_dir.name, "test_static_load_var_list"),
                exe,
                var_list,
            )
            var_list_names = [var.name for var in var_list]
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    if var.name in var_list_names:
                        # loaded vars
                        base_t = base_map[var.name]
                        np.testing.assert_array_equal(new_t, base_t)
                    else:
                        # not loaded vars
                        self.assertTrue(np.sum(np.abs(new_t)) == 0)


class TestLoadFromOldInterfaceSingleFile(unittest.TestCase):
    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_load_from_old_interface(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200
        temp_dir = tempfile.TemporaryDirectory()

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )
            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t
            save_dir = os.path.join(temp_dir.name, "test_path")
            # base.save(main_program, "./test_1")
            paddle.distributed.io.save_persistables(
                exe, save_dir, main_program, filename="model_single"
            )

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            file_model_path = os.path.join(save_dir, "model_single")
            paddle.static.load(
                main_program,
                file_model_path,
                exe,
                paddle.static.io.get_program_persistable_vars(main_program),
            )

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

            # test exception
            # change shape
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    old_shape = np.array(ten).shape
                    new_shape = [e + 10 for e in old_shape]

                    var.desc.set_shape(new_shape)

            with self.assertRaises(RuntimeError):
                paddle.static.load(
                    main_program,
                    file_model_path,
                    exe,
                    paddle.static.io.get_program_persistable_vars(main_program),
                )

            with self.assertRaises(RuntimeError):
                paddle.static.load(
                    main_program,
                    file_model_path,
                    exe,
                    paddle.static.io.get_program_persistable_vars(main_program),
                )

            # check when executor is None
            with self.assertRaises(ValueError):
                paddle.static.load(
                    main_program,
                    file_model_path,
                    None,
                    paddle.static.io.get_program_persistable_vars(main_program),
                )

            # check when var list is None
            with self.assertRaises(ValueError):
                paddle.static.load(main_program, file_model_path, exe, None)

            # check save params, load var_list = get_program_persistable_vars
            with self.assertRaises(RuntimeError):
                temp_var = framework.Variable(
                    main_program.global_block(), shape=[1], name="test_temp_var"
                )
                all_var_list = list(main_program.list_vars())
                paddle.static.load(
                    main_program,
                    file_model_path,
                    exe,
                    all_var_list + [temp_var],
                )
        temp_dir.cleanup()


class TestProgramStateOldSave(unittest.TestCase):
    def setUp(self):
        self.test_dygraph = True
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )

            test_program = base.default_main_program().clone(for_test=True)

            add_1 = paddle.static.nn.fc(
                static_last_hidden,
                size=hidden_size,
                num_flatten_dims=2,
                bias_attr=False,
            )

            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t
            save_dir = os.path.join(self.temp_dir.name, "test_program_1")
            paddle.distributed.io.save_persistables(exe, save_dir, main_program)

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            # case 1: load basic
            program_state = paddle.static.load_program_state(save_dir)
            paddle.static.set_program_state(main_program, program_state)
            self.check_in_static(main_program, base_map)

            # case 2: load with no need file
            def symlink_force(target, link_name):
                try:
                    self.create_symlink(target, link_name)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        os.remove(link_name)
                        self.create_symlink(target, link_name)
                    else:
                        raise e

            program_state = paddle.static.load_program_state(save_dir)
            paddle.static.set_program_state(main_program, program_state)
            self.check_in_static(main_program, base_map)

            # case 3: load with var_list
            program_state = paddle.static.load_program_state(
                save_dir, main_program.all_parameters()
            )
            paddle.static.set_program_state(main_program, program_state)
            self.check_in_static(main_program, base_map)

        if self.test_dygraph:
            # make sure `load_program_state` can be used in dynamic graph mode
            with base.dygraph.guard(place):
                load_state = paddle.static.load_program_state(save_dir)
                for k, v in load_state.items():
                    np.testing.assert_array_equal(base_map[k], v)

    def create_symlink(self, target, link_name):
        try:
            os.symlink(target, link_name)
        except AttributeError:
            import ctypes

            kernel_dll = ctypes.windll.LoadLibrary("kernel32.dll")
            kernel_dll.CreateSymbolicLinkA(target, link_name, 0)

    def check_in_static(self, main_program, base_map):
        for var in main_program.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                new_t = np.array(
                    base.global_scope().find_var(var.name).get_tensor()
                )
                base_t = base_map[var.name]
                np.testing.assert_array_equal(new_t, base_t)


class TestProgramStateOldSaveSingleModel(unittest.TestCase):
    def set_place(self):
        return (
            base.CPUPlace()
            if not core.is_compiled_with_cuda()
            else base.CUDAPlace(0)
        )

    def test_ptb_rnn_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200
        temp_dir = tempfile.TemporaryDirectory()

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel(
                "ptb_model",
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
            )

            place = self.set_place()
            exe = base.Executor(place)
            sgd = Adam(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps], dtype='int64'
            )
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )

            test_program = base.default_main_program().clone(for_test=True)

            add_1 = paddle.static.nn.fc(
                static_last_hidden,
                size=hidden_size,
                num_flatten_dims=2,
                bias_attr=False,
            )

            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}

            out = exe.run(framework.default_startup_program())

            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(
                    base.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

            # get value before save
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            save_dir = os.path.join(temp_dir.name, "test_program_2")
            paddle.distributed.io.save_persistables(
                exe, save_dir, main_program, filename="model_1"
            )

            # set var to zero
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been set to zero
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            # base.load(test_program, "./test_1", None )
            program_state = paddle.static.load_program_state(
                os.path.join(save_dir, "model_1"),
                var_list=paddle.static.io.get_program_persistable_vars(
                    main_program
                ),
            )
            paddle.static.set_program_state(main_program, program_state)

            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

            with self.assertRaises(ValueError):
                paddle.static.load_program_state(
                    os.path.join(save_dir, "model_1")
                )

            with self.assertRaises(TypeError):
                paddle.static.load_program_state(
                    os.path.join(save_dir, "model_1"), var_list=["str"]
                )

            with self.assertRaises(RuntimeError):
                paddle.static.load_program_state(
                    os.path.join(save_dir, "model_1"),
                    var_list=[
                        main_program.global_block().create_var(
                            name="fake_var_name", persistable=True
                        )
                    ],
                )
        temp_dir.cleanup()


class TestStaticSaveLoadPickle(unittest.TestCase):
    def test_pickle_protocol(self):
        # enable static graph mode
        paddle.enable_static()

        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="static_save_load_large_x",
                shape=[None, 10],
                dtype='float32',
            )
            x.desc.set_need_check_feed(False)
            z = paddle.static.nn.fc(x, 10, bias_attr=False)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()

            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t

            temp_dir = tempfile.TemporaryDirectory()
            path = os.path.join(
                temp_dir.name, "test_static_save_load_pickle", "pickle_protocol"
            )

            with self.assertRaises(ValueError):
                paddle.static.save(prog, path, 2.0)

            with self.assertRaises(ValueError):
                paddle.static.save(prog, path, 1)

            with self.assertRaises(ValueError):
                paddle.static.save(prog, path, 5)

            protocols = [2, 3, 4]
            for protocol in protocols:
                paddle.static.save(prog, path, protocol)
                # set var to zero
                for var in prog.list_vars():
                    if isinstance(var, framework.Parameter) or var.persistable:
                        ten = (
                            base.global_scope().find_var(var.name).get_tensor()
                        )
                        ten.set(np.zeros_like(np.array(ten)), place)

                        new_t = np.array(
                            base.global_scope().find_var(var.name).get_tensor()
                        )
                        self.assertTrue(np.sum(np.abs(new_t)) == 0)

                paddle.static.load(prog, path)

                for var in prog.list_vars():
                    if isinstance(var, framework.Parameter) or var.persistable:
                        new_t = np.array(
                            base.global_scope().find_var(var.name).get_tensor()
                        )
                        base_t = base_map[var.name]
                        np.testing.assert_array_equal(new_t, base_t)


class TestSaveLoadInferenceModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'no_params')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_params(self):
        main_program = framework.Program()
        with framework.program_guard(main_program):
            x = paddle.static.data(name="x", shape=[10, 10], dtype='float32')
            x.desc.set_need_check_feed(False)
            y = x + x

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)

            paddle.static.save_inference_model(self.model_path, [x], [y], exe)

            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = paddle.static.load_inference_model(self.model_path, exe)

            self.assertEqual(feed_target_names, ['x'])
            self.assertEqual(fetch_targets[0].shape, (10, 10))
            ops = [op.type for op in inference_program.block(0).ops]
            self.assertEqual(ops, ['feed', 'elementwise_add', 'scale', 'fetch'])


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
