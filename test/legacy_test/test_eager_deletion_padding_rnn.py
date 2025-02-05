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

import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base.executor import Executor

os.environ["CPU_NUM"] = "1"


class RNNConfig:
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

        if rnn_model not in ('static', 'cudnn'):
            raise ValueError('Unsupported rnn_model.')

        self.batch_size = 12
        self.max_epoch = 3
        self.random_seed = 123


# Fake data reader for test
class Reader:
    def get_data_iter(self, rnn_config):
        for i in range(rnn_config.max_epoch):
            x = np.zeros(
                shape=(rnn_config.batch_size, rnn_config.num_steps),
                dtype='int64',
            )
            y = np.ones(
                shape=(rnn_config.batch_size, rnn_config.num_steps),
                dtype='int64',
            )
            yield (x, y)


# Model from PaddleNLP/models/language_model/lm_model.py in Paddle Models repo
def lm_model(
    hidden_size,
    vocab_size,
    batch_size,
    num_layers=2,
    num_steps=20,
    init_scale=0.1,
    dropout=None,
    rnn_model='static',
):
    def encoder_static(
        input_embedding, len=3, init_hidden=None, init_cell=None
    ):
        weight_1_arr = []
        weight_2_arr = []
        bias_arr = []
        hidden_array = []
        cell_array = []
        mask_array = []
        for i in range(num_layers):
            weight_1 = paddle.create_parameter(
                [hidden_size * 2, hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale
                ),
            )
            weight_1_arr.append(weight_1)
            bias_1 = paddle.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            bias_arr.append(bias_1)

            pre_hidden = paddle.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1]
            )
            pre_cell = paddle.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1]
            )
            pre_hidden = paddle.reshape(pre_hidden, shape=[-1, hidden_size])
            pre_cell = paddle.reshape(pre_cell, shape=[-1, hidden_size])
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        res = []
        sliced_inputs = paddle.split(
            input_embedding, num_or_sections=len, axis=1
        )

        for index in range(len):
            input = sliced_inputs[index]
            input = paddle.reshape(input, shape=[-1, hidden_size])
            for k in range(num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = weight_1_arr[k]
                bias = bias_arr[k]

                nn = paddle.concat([input, pre_hidden], 1)
                gate_input = paddle.matmul(x=nn, y=weight_1)

                gate_input = paddle.add(gate_input, bias)
                i, j, f, o = paddle.split(
                    gate_input, num_or_sections=4, axis=-1
                )

                c = pre_cell * paddle.nn.functional.sigmoid(
                    f
                ) + paddle.nn.functional.sigmoid(i) * paddle.tanh(j)
                m = paddle.tanh(c) * paddle.nn.functional.sigmoid(o)

                hidden_array[k] = m
                cell_array[k] = c
                input = m

                if dropout is not None and dropout > 0.0:
                    input = paddle.nn.functional.dropout(
                        input,
                        p=dropout,
                        mode='upscale_in_train',
                    )

            res.append(input)

        last_hidden = paddle.concat(hidden_array, 1)
        last_hidden = paddle.reshape(
            last_hidden, shape=[-1, num_layers, hidden_size]
        )
        last_hidden = paddle.transpose(x=last_hidden, perm=[1, 0, 2])

        last_cell = paddle.concat(cell_array, 1)
        last_cell = paddle.reshape(
            last_cell, shape=[-1, num_layers, hidden_size]
        )
        last_cell = paddle.transpose(x=last_cell, perm=[1, 0, 2])

        real_res = paddle.concat(res, 0)
        real_res = paddle.reshape(real_res, shape=[len, -1, hidden_size])
        real_res = paddle.transpose(x=real_res, perm=[1, 0, 2])

        return real_res, last_hidden, last_cell

    batch_size_each = batch_size
    x = paddle.static.data(
        name="x", shape=[batch_size_each, num_steps, 1], dtype='int64'
    )
    y = paddle.static.data(
        name="y", shape=[batch_size_each * num_steps, 1], dtype='int64'
    )

    init_hidden = paddle.static.data(
        name="init_hidden",
        shape=[num_layers, batch_size_each, hidden_size],
        dtype='float32',
    )
    init_cell = paddle.static.data(
        name="init_cell",
        shape=[num_layers, batch_size_each, hidden_size],
        dtype='float32',
    )

    init_cell.persistable = True
    init_hidden.persistable = True

    init_hidden_reshape = paddle.reshape(
        init_hidden, shape=[num_layers, -1, hidden_size]
    )
    init_cell_reshape = paddle.reshape(
        init_cell, shape=[num_layers, -1, hidden_size]
    )

    if paddle.framework.in_pir_mode():
        Emb = paddle.nn.Embedding(
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
        x_emb = Emb(x)
    else:
        x_emb = paddle.static.nn.embedding(
            input=x,
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=base.ParamAttr(
                name='embedding_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale
                ),
            ),
        )

    x_emb = paddle.reshape(x_emb, shape=[-1, num_steps, hidden_size])
    if dropout is not None and dropout > 0.0:
        x_emb = paddle.nn.functional.dropout(
            x_emb,
            p=dropout,
            mode='upscale_in_train',
        )

    if rnn_model == "static":
        rnn_out, last_hidden, last_cell = encoder_static(
            x_emb,
            len=num_steps,
            init_hidden=init_hidden_reshape,
            init_cell=init_cell_reshape,
        )
    else:
        print("type not support")
        return

    rnn_out = paddle.reshape(rnn_out, shape=[-1, num_steps, hidden_size])

    softmax_weight = paddle.create_parameter(
        [hidden_size, vocab_size],
        dtype="float32",
        name="softmax_weight",
        default_initializer=paddle.nn.initializer.Uniform(
            low=-init_scale, high=init_scale
        ),
    )
    softmax_bias = paddle.create_parameter(
        [vocab_size],
        dtype="float32",
        name='softmax_bias',
        default_initializer=paddle.nn.initializer.Uniform(
            low=-init_scale, high=init_scale
        ),
    )

    projection = paddle.matmul(rnn_out, softmax_weight)
    projection = paddle.add(projection, softmax_bias)
    projection = paddle.reshape(projection, shape=[-1, vocab_size])

    loss = paddle.nn.functional.softmax_with_cross_entropy(
        logits=projection, label=y, soft_label=False
    )

    loss = paddle.reshape(loss, shape=[-1, num_steps])
    loss = paddle.mean(loss, axis=[0])
    loss = paddle.sum(loss)

    loss.persistable = True
    last_cell.persistable = True
    last_hidden.persistable = True

    # This will feed last_hidden, last_cell to init_hidden, init_cell, which
    # can be used directly in next batch. This can avoid the fetching of
    # last_hidden and last_cell and feeding of init_hidden and init_cell in
    # each training step.
    paddle.assign(last_cell, output=init_cell)
    paddle.assign(last_hidden, output=init_hidden)

    feeding_list = [x, y, init_hidden, init_cell]
    return loss, last_hidden, last_cell, feeding_list


class PaddingRNNTestBase(unittest.TestCase):
    def setUp(self):
        self.reader = Reader()
        self.device_count = 1

        # The default build_strategy used for PaddingRNN.
        # You can change it in set_customed_config.
        self.build_strategy = base.BuildStrategy()
        self.build_strategy.enable_inplace = True
        self.build_strategy.memory_optimize = False
        self.build_strategy.fuse_all_optimizer_ops = True

        # CPU executor is used for PaddingRNN default.
        # You can change to CUDA executor in set_customed_config.
        self.exe = Executor(base.CPUPlace())

    def set_customed_config(self):
        # This function will be called before training.
        # You can override the function to set your own config.
        pass

    def _prepare_program(self, config):
        paddle.seed(config.random_seed)
        self.main_program = base.Program()
        self.startup_program = base.Program()
        with base.program_guard(self.main_program, self.startup_program):
            with base.unique_name.guard():
                res_vars = lm_model(
                    config.hidden_size,
                    config.vocab_size,
                    config.batch_size,
                    num_layers=config.num_layers,
                    num_steps=config.num_steps,
                    init_scale=config.init_scale,
                    dropout=config.dropout,
                    rnn_model=config.rnn_model,
                )
                (
                    self.loss,
                    self.last_hidden,
                    self.last_cell,
                    self.feed_list,
                ) = res_vars

                paddle.nn.clip.set_gradient_clip(
                    clip=paddle.nn.ClipGradByGlobalNorm(
                        clip_norm=config.max_grad_norm
                    )
                )

                optimizer = paddle.optimizer.SGD(learning_rate=1.0)
                optimizer.minimize(self.loss)

        self.exe.run(self.startup_program)

        self.train_program = self.main_program

    def _generate_init_data(self):
        init_hidden = np.zeros(
            (
                self.config.num_layers,
                self.config.batch_size,
                self.config.hidden_size,
            ),
            dtype='float32',
        )
        init_cell = np.zeros(
            (
                self.config.num_layers,
                self.config.batch_size,
                self.config.hidden_size,
            ),
            dtype='float32',
        )
        return init_hidden, init_cell

    def _generate_new_lr(self, epoch_id=0, device_count=1):
        new_lr = self.config.base_learning_rate * (
            self.config.lr_decay
            ** max(epoch_id + 1 - self.config.epoch_start_decay, 0.0)
        )
        lr = np.ones((self.device_count), dtype='float32') * new_lr
        return lr

    def _prepare_input(
        self,
        batch,
        init_hidden=None,
        init_cell=None,
        epoch_id=0,
        with_lr=True,
        device_count=1,
    ):
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
                device_count=self.device_count,
            )

            fetch_outs = self.exe.run(
                self.train_program,
                feed=input_data_feed,
                fetch_list=[
                    self.loss,
                    self.last_hidden,
                    self.last_cell,
                ],
                use_program_cache=use_program_cache,
            )

            cost_train = np.array(fetch_outs[0])
            init_hidden = np.array(fetch_outs[1])
            init_cell = np.array(fetch_outs[2])

            total_loss += cost_train
            iters += self.config.num_steps

            batch_ppl = np.exp(total_loss / iters)
            ppl = np.append(ppl, batch_ppl)
        return ppl

    def train(self, config, use_program_cache=True):
        self.set_customed_config()

        self.config = config
        self._prepare_program(config)
        ppl = np.zeros(shape=(0, config.batch_size))
        for epoch_id in range(config.max_epoch):
            train_ppl = self._train_an_epoch(epoch_id, use_program_cache)
            ppl = np.append(ppl, train_ppl)
        return ppl


if __name__ == '__main__':
    unittest.main()
