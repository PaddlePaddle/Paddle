#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import os

from paddle.fluid import ParamAttr
from paddle.fluid.contrib.layers import basic_lstm
from paddle.fluid.executor import Executor
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN

os.environ["CPU_NUM"] = "1"


class RNNConfig(object):

    def __init__(self, model_type, rnn_model):
        self.model_type = model_type
        self.rnn_model = rnn_model

        self.vocab_size = 10000
        if self.model_type == "test":
            self.num_layers = 1
            self.batch_size = 2
            self.hidden_size = 10
            self.num_steps = 3
            self.init_scale = 0.1
            self.max_grad_norm = 5.0
            self.epoch_start_decay = 1
            self.max_epoch = 1
            self.dropout = 0.0
            self.lr_decay = 0.5
            self.base_learning_rate = 1.0
        elif self.model_type == "small":
            self.num_layers = 2
            self.batch_size = 20
            self.hidden_size = 200
            self.num_steps = 20
            self.init_scale = 0.1
            self.max_grad_norm = 5.0
            self.epoch_start_decay = 4
            self.max_epoch = 13
            self.dropout = 0.0
            self.lr_decay = 0.5
            self.base_learning_rate = 1.0
        elif self.model_type == "medium":
            self.num_layers = 2
            self.batch_size = 20
            self.hidden_size = 650
            self.num_steps = 35
            self.init_scale = 0.05
            self.max_grad_norm = 5.0
            self.epoch_start_decay = 6
            self.max_epoch = 39
            self.dropout = 0.5
            self.lr_decay = 0.8
            self.base_learning_rate = 1.0
        elif self.model_type == "large":
            self.num_layers = 2
            self.batch_size = 20
            self.hidden_size = 1500
            self.num_steps = 35
            self.init_scale = 0.04
            self.max_grad_norm = 10.0
            self.epoch_start_decay = 14
            self.max_epoch = 55
            self.dropout = 0.65
            self.lr_decay = 1.0 / 1.15
            self.base_learning_rate = 1.0
        else:
            raise ValueError('Unsupported model_type.')

        if rnn_model not in ('static', 'padding', 'cudnn', 'basic_lstm'):
            raise ValueError('Unsupported rnn_model.')

        self.batch_size = 12
        self.max_epoch = 3
        self.random_seed = 123


# Fake data reader for test
class Reader(object):

    def get_data_iter(self, rnn_config):
        for i in range(rnn_config.max_epoch):
            x = np.zeros(shape=(rnn_config.batch_size, rnn_config.num_steps),
                         dtype='int64')
            y = np.ones(shape=(rnn_config.batch_size, rnn_config.num_steps),
                        dtype='int64')
            yield (x, y)


# Model from PaddleNLP/models/language_model/lm_model.py in Paddle Models repo
def lm_model(hidden_size,
             vocab_size,
             batch_size,
             num_layers=2,
             num_steps=20,
             init_scale=0.1,
             dropout=None,
             rnn_model='static'):

    def padding_rnn(input_embedding, len=3, init_hidden=None, init_cell=None):
        weight_1_arr = []
        weight_2_arr = []
        bias_arr = []
        hidden_array = []
        cell_array = []
        mask_array = []
        for i in range(num_layers):
            weight_1 = layers.create_parameter(
                [hidden_size * 2, hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale))
            weight_1_arr.append(weight_1)
            bias_1 = layers.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            bias_arr.append(bias_1)

            pre_hidden = layers.slice(init_hidden,
                                      axes=[0],
                                      starts=[i],
                                      ends=[i + 1])
            pre_cell = layers.slice(init_cell,
                                    axes=[0],
                                    starts=[i],
                                    ends=[i + 1])
            pre_hidden = layers.reshape(pre_hidden, shape=[-1, hidden_size])
            pre_cell = layers.reshape(pre_cell, shape=[-1, hidden_size])
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        input_embedding = layers.transpose(input_embedding, perm=[1, 0, 2])
        rnn = PaddingRNN()

        with rnn.step():
            input = rnn.step_input(input_embedding)
            for k in range(num_layers):
                pre_hidden = rnn.memory(init=hidden_array[k])
                pre_cell = rnn.memory(init=cell_array[k])
                weight_1 = weight_1_arr[k]
                bias = bias_arr[k]

                nn = layers.concat([input, pre_hidden], 1)
                gate_input = layers.matmul(x=nn, y=weight_1)

                gate_input = layers.elementwise_add(gate_input, bias)
                i = layers.slice(gate_input,
                                 axes=[1],
                                 starts=[0],
                                 ends=[hidden_size])
                j = layers.slice(gate_input,
                                 axes=[1],
                                 starts=[hidden_size],
                                 ends=[hidden_size * 2])
                f = layers.slice(gate_input,
                                 axes=[1],
                                 starts=[hidden_size * 2],
                                 ends=[hidden_size * 3])
                o = layers.slice(gate_input,
                                 axes=[1],
                                 starts=[hidden_size * 3],
                                 ends=[hidden_size * 4])

                c = pre_cell * layers.sigmoid(f) + layers.sigmoid(
                    i) * layers.tanh(j)
                m = layers.tanh(c) * layers.sigmoid(o)

                rnn.update_memory(pre_hidden, m)
                rnn.update_memory(pre_cell, c)

                rnn.step_output(m)
                rnn.step_output(c)

                input = m

                if dropout != None and dropout > 0.0:
                    input = layers.dropout(
                        input,
                        dropout_prob=dropout,
                        dropout_implementation='upscale_in_train')

            rnn.step_output(input)
        rnnout = rnn()

        last_hidden_array = []
        last_cell_array = []
        real_res = rnnout[-1]
        for i in range(num_layers):
            m = rnnout[i * 2]
            c = rnnout[i * 2 + 1]
            m.stop_gradient = True
            c.stop_gradient = True
            last_h = layers.slice(m,
                                  axes=[0],
                                  starts=[num_steps - 1],
                                  ends=[num_steps])
            last_hidden_array.append(last_h)
            last_c = layers.slice(c,
                                  axes=[0],
                                  starts=[num_steps - 1],
                                  ends=[num_steps])
            last_cell_array.append(last_c)
        real_res = layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = layers.concat(last_hidden_array, 0)
        last_cell = layers.concat(last_cell_array, 0)

        return real_res, last_hidden, last_cell

    def encoder_static(input_embedding,
                       len=3,
                       init_hidden=None,
                       init_cell=None):

        weight_1_arr = []
        weight_2_arr = []
        bias_arr = []
        hidden_array = []
        cell_array = []
        mask_array = []
        for i in range(num_layers):
            weight_1 = layers.create_parameter(
                [hidden_size * 2, hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale))
            weight_1_arr.append(weight_1)
            bias_1 = layers.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            bias_arr.append(bias_1)

            pre_hidden = layers.slice(init_hidden,
                                      axes=[0],
                                      starts=[i],
                                      ends=[i + 1])
            pre_cell = layers.slice(init_cell,
                                    axes=[0],
                                    starts=[i],
                                    ends=[i + 1])
            pre_hidden = layers.reshape(pre_hidden,
                                        shape=[-1, hidden_size],
                                        inplace=True)
            pre_cell = layers.reshape(pre_cell,
                                      shape=[-1, hidden_size],
                                      inplace=True)
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        res = []
        sliced_inputs = layers.split(input_embedding,
                                     num_or_sections=len,
                                     dim=1)

        for index in range(len):
            input = sliced_inputs[index]
            input = layers.reshape(input, shape=[-1, hidden_size], inplace=True)
            for k in range(num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = weight_1_arr[k]
                bias = bias_arr[k]

                nn = layers.concat([input, pre_hidden], 1)
                gate_input = layers.matmul(x=nn, y=weight_1)

                gate_input = layers.elementwise_add(gate_input, bias)
                i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)

                c = pre_cell * layers.sigmoid(f) + layers.sigmoid(
                    i) * layers.tanh(j)
                m = layers.tanh(c) * layers.sigmoid(o)

                hidden_array[k] = m
                cell_array[k] = c
                input = m

                if dropout != None and dropout > 0.0:
                    input = layers.dropout(
                        input,
                        dropout_prob=dropout,
                        dropout_implementation='upscale_in_train')

            res.append(input)

        last_hidden = layers.concat(hidden_array, 1)
        last_hidden = layers.reshape(last_hidden,
                                     shape=[-1, num_layers, hidden_size],
                                     inplace=True)
        last_hidden = layers.transpose(x=last_hidden, perm=[1, 0, 2])

        last_cell = layers.concat(cell_array, 1)
        last_cell = layers.reshape(last_cell,
                                   shape=[-1, num_layers, hidden_size])
        last_cell = layers.transpose(x=last_cell, perm=[1, 0, 2])

        real_res = layers.concat(res, 0)
        real_res = layers.reshape(real_res,
                                  shape=[len, -1, hidden_size],
                                  inplace=True)
        real_res = layers.transpose(x=real_res, perm=[1, 0, 2])

        return real_res, last_hidden, last_cell

    batch_size_each = batch_size
    x = layers.data(name="x",
                    shape=[batch_size_each, num_steps, 1],
                    dtype='int64',
                    append_batch_size=False)
    y = layers.data(name="y",
                    shape=[batch_size_each * num_steps, 1],
                    dtype='int64',
                    append_batch_size=False)

    init_hidden = layers.data(name="init_hidden",
                              shape=[num_layers, batch_size_each, hidden_size],
                              dtype='float32',
                              append_batch_size=False)
    init_cell = layers.data(name="init_cell",
                            shape=[num_layers, batch_size_each, hidden_size],
                            dtype='float32',
                            append_batch_size=False)

    init_cell.persistable = True
    init_hidden.persistable = True

    init_hidden_reshape = layers.reshape(init_hidden,
                                         shape=[num_layers, -1, hidden_size])
    init_cell_reshape = layers.reshape(init_cell,
                                       shape=[num_layers, -1, hidden_size])

    x_emb = layers.embedding(
        input=x,
        size=[vocab_size, hidden_size],
        dtype='float32',
        is_sparse=False,
        param_attr=fluid.ParamAttr(
            name='embedding_para',
            initializer=fluid.initializer.UniformInitializer(low=-init_scale,
                                                             high=init_scale)))

    x_emb = layers.reshape(x_emb,
                           shape=[-1, num_steps, hidden_size],
                           inplace=True)
    if dropout != None and dropout > 0.0:
        x_emb = layers.dropout(x_emb,
                               dropout_prob=dropout,
                               dropout_implementation='upscale_in_train')

    if rnn_model == "padding":
        rnn_out, last_hidden, last_cell = padding_rnn(
            x_emb,
            len=num_steps,
            init_hidden=init_hidden_reshape,
            init_cell=init_cell_reshape)
    elif rnn_model == "static":
        rnn_out, last_hidden, last_cell = encoder_static(
            x_emb,
            len=num_steps,
            init_hidden=init_hidden_reshape,
            init_cell=init_cell_reshape)
    elif rnn_model == "cudnn":
        x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
        rnn_out, last_hidden, last_cell = layers.lstm(
            x_emb,
            init_hidden_reshape,
            init_cell_reshape,
            num_steps,
            hidden_size,
            num_layers,
            is_bidirec=False,
            default_initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale))
        rnn_out = layers.transpose(rnn_out, perm=[1, 0, 2])
    elif rnn_model == "basic_lstm":
        rnn_out, last_hidden, last_cell = basic_lstm( x_emb, init_hidden, init_cell, hidden_size, \
                num_layers=num_layers, batch_first=True, dropout_prob=dropout, \
                param_attr = ParamAttr( initializer=fluid.initializer.UniformInitializer(low=-init_scale, high=init_scale) ), \
                bias_attr = ParamAttr( initializer = fluid.initializer.Constant(0.0) ), \
                forget_bias = 0.0)
    else:
        print("type not support")
        return

    rnn_out = layers.reshape(rnn_out,
                             shape=[-1, num_steps, hidden_size],
                             inplace=True)

    softmax_weight = layers.create_parameter(
        [hidden_size, vocab_size],
        dtype="float32",
        name="softmax_weight",
        default_initializer=fluid.initializer.UniformInitializer(
            low=-init_scale, high=init_scale))
    softmax_bias = layers.create_parameter(
        [vocab_size],
        dtype="float32",
        name='softmax_bias',
        default_initializer=fluid.initializer.UniformInitializer(
            low=-init_scale, high=init_scale))

    projection = layers.matmul(rnn_out, softmax_weight)
    projection = layers.elementwise_add(projection, softmax_bias)
    projection = layers.reshape(projection,
                                shape=[-1, vocab_size],
                                inplace=True)

    loss = layers.softmax_with_cross_entropy(logits=projection,
                                             label=y,
                                             soft_label=False)

    loss = layers.reshape(loss, shape=[-1, num_steps], inplace=True)
    loss = layers.reduce_mean(loss, dim=[0])
    loss = layers.reduce_sum(loss)

    loss.persistable = True
    last_cell.persistable = True
    last_hidden.persistable = True

    # This will feed last_hidden, last_cell to init_hidden, init_cell, which
    # can be used directly in next batch. This can avoid the fetching of
    # last_hidden and last_cell and feeding of init_hidden and init_cell in
    # each training step.
    layers.assign(input=last_cell, output=init_cell)
    layers.assign(input=last_hidden, output=init_hidden)

    feeding_list = ['x', 'y', 'init_hidden', 'init_cell']
    return loss, last_hidden, last_cell, feeding_list


class PaddingRNNTestBase(unittest.TestCase):

    def setUp(self):
        self.reader = Reader()
        self.device_count = 1

        # The default exec_strategy used for PaddingRNN.
        # You can change it in set_customed_config.
        self.exec_strategy = fluid.ExecutionStrategy()
        self.exec_strategy.num_threads = self.device_count
        self.exec_strategy.num_iteration_per_drop_scope = 100

        # The default build_strategy used for PaddingRNN.
        # You can change it in set_customed_config.
        self.build_strategy = fluid.BuildStrategy()
        self.build_strategy.enable_inplace = True
        self.build_strategy.memory_optimize = False
        self.build_strategy.fuse_all_optimizer_ops = True

        # CPU executor is used for PaddingRNN default.
        # You can change to CUDA executor in set_customed_config.
        self.exe = Executor(fluid.CPUPlace())

    def set_customed_config(self):
        # This function will be called before training.
        # You can override the function to set your own config.
        pass

    def _prepare_program(self, config, parallel=True):
        paddle.seed(config.random_seed)
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        with fluid.program_guard(self.main_program, self.startup_program):
            with fluid.unique_name.guard():
                res_vars = lm_model(config.hidden_size,
                                    config.vocab_size,
                                    config.batch_size,
                                    num_layers=config.num_layers,
                                    num_steps=config.num_steps,
                                    init_scale=config.init_scale,
                                    dropout=config.dropout,
                                    rnn_model=config.rnn_model)
                self.loss, self.last_hidden, self.last_cell, self.feed_order = res_vars

                fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByGlobalNorm(
                        clip_norm=config.max_grad_norm))

                self.learning_rate = fluid.layers.create_global_var(
                    name="learning_rate",
                    shape=[1],
                    value=1.0,
                    dtype='float32',
                    persistable=True)

                optimizer = fluid.optimizer.SGD(
                    learning_rate=self.learning_rate)
                optimizer.minimize(self.loss)

        self.exe.run(self.startup_program)

        if parallel:
            self.train_program = fluid.compiler.CompiledProgram(
                self.main_program).with_data_parallel(
                    loss_name=self.loss.name,
                    build_strategy=self.build_strategy,
                    exec_strategy=self.exec_strategy)
        else:
            self.train_program = self.main_program

    def _generate_init_data(self):
        init_hidden = np.zeros((self.config.num_layers, self.config.batch_size,
                                self.config.hidden_size),
                               dtype='float32')
        init_cell = np.zeros((self.config.num_layers, self.config.batch_size,
                              self.config.hidden_size),
                             dtype='float32')
        return init_hidden, init_cell

    def _generate_new_lr(self, epoch_id=0, device_count=1):
        new_lr = self.config.base_learning_rate * (self.config.lr_decay**max(
            epoch_id + 1 - self.config.epoch_start_decay, 0.0))
        lr = np.ones((self.device_count), dtype='float32') * new_lr
        return lr

    def _prepare_input(self,
                       batch,
                       init_hidden=None,
                       init_cell=None,
                       epoch_id=0,
                       with_lr=True,
                       device_count=1):
        x, y = batch
        x = x.reshape((-1, self.config.num_steps, 1))
        y = y.reshape((-1, 1))

        res = {}
        res['x'] = x
        res['y'] = y
        if init_hidden is not None:
            res['init_hidden'] = init_hidden
        if init_cell is not None:
            res['init_cell'] = init_cell
        if with_lr:
            res['learning_rate'] = self._generate_new_lr(epoch_id, device_count)
        return res

    def _train_an_epoch(self, epoch_id, use_program_cache=True):
        train_data_iter = self.reader.get_data_iter(self.config)

        total_loss = 0
        iters = 0

        init_hidden, init_cell = self._generate_init_data()
        ppl = np.zeros(shape=(0))
        for batch_id, batch in enumerate(train_data_iter):
            input_data_feed = self._prepare_input(
                batch,
                init_hidden=init_hidden,
                init_cell=init_cell,
                epoch_id=epoch_id,
                with_lr=True,
                device_count=self.device_count)

            fetch_outs = self.exe.run(self.train_program,
                                      feed=input_data_feed,
                                      fetch_list=[
                                          self.loss.name, "learning_rate",
                                          self.last_hidden.name,
                                          self.last_cell.name
                                      ],
                                      use_program_cache=use_program_cache)

            cost_train = np.array(fetch_outs[0])
            lr = np.array(fetch_outs[1])
            init_hidden = np.array(fetch_outs[2])
            init_cell = np.array(fetch_outs[3])

            total_loss += cost_train
            iters += self.config.num_steps

            batch_ppl = np.exp(total_loss / iters)
            ppl = np.append(ppl, batch_ppl)
        return ppl

    def train(self, config, parallel=True, use_program_cache=True):
        self.set_customed_config()

        self.config = config
        self._prepare_program(config, parallel)
        ppl = np.zeros(shape=(0, config.batch_size))
        for epoch_id in range(config.max_epoch):
            train_ppl = self._train_an_epoch(epoch_id, use_program_cache)
            ppl = np.append(ppl, train_ppl)
        return ppl

    def compare_padding_static_mode(self,
                                    parallel=True,
                                    use_program_cache=True):
        '''
        Test that train ppl of padding mode is same to that of static mode
        '''
        config = RNNConfig('test', 'padding')
        with fluid.scope_guard(fluid.Scope()):
            padding_rnn_ppl = self.train(config, parallel, use_program_cache)
        config = RNNConfig('test', 'static')
        with fluid.scope_guard(fluid.Scope()):
            static_rnn_ppl = self.train(config, parallel, use_program_cache)
        np.testing.assert_allclose(padding_rnn_ppl, static_rnn_ppl, rtol=0.001)


class EagerDeletionPaddingRNNTest(PaddingRNNTestBase):

    def test_padding_mode_no_eager_deletion(self):
        '''
        Test that train ppl of padding mode is same to that of static mode without eager deletion
        '''
        fluid.core._set_eager_deletion_mode(-1.0, 1.0, True)
        # When parallel is True, use_program_cache does not make a difference.
        self.compare_padding_static_mode(parallel=True, use_program_cache=True)

    def test_padding_mode_eager_deletion(self):
        '''
        Test that train ppl of padding mode is same to that of static mode under eager deletion
        '''
        fluid.core._set_eager_deletion_mode(0.0, 1.0, True)
        # When parallel is True, use_program_cache does not make a difference.
        self.compare_padding_static_mode(parallel=True, use_program_cache=True)


if __name__ == '__main__':
    unittest.main()
