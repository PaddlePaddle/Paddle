# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
import unittest

import numpy as np
from test_lac import DynamicGRU

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.jit import ProgramTranslator
from paddle.jit.api import declarative
from paddle.nn import Embedding, Linear

SEED = 2020
program_translator = ProgramTranslator()

# Note: Set True to eliminate randomness.
#     1. For one operation, cuDNN has several algorithms,
#        some algorithm results are non-deterministic, like convolution algorithms.
if fluid.is_compiled_with_cuda():
    fluid.set_flags({'FLAGS_cudnn_deterministic': True})


class SimpleConvPool(fluid.dygraph.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        use_cudnn=True,
        batch_size=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self._conv2d = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=[1, 1],
        )

    def forward(self, inputs):
        x = paddle.tanh(self._conv2d(inputs))
        x = paddle.max(x, axis=-1)
        x = paddle.reshape(x, shape=[self.batch_size, -1])
        return x


class CNN(fluid.dygraph.Layer):
    def __init__(self, dict_dim, batch_size, seq_len):
        super().__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.channels = 1
        self.win_size = [3, self.hid_dim]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            self.dict_dim + 1,
            self.emb_dim,
            sparse=False,
        )
        self._simple_conv_pool_1 = SimpleConvPool(
            self.channels,
            self.hid_dim,
            self.win_size,
            batch_size=self.batch_size,
        )
        self._fc1 = Linear(
            self.hid_dim * self.seq_len,
            self.fc_hid_dim,
        )
        self._fc1_act = paddle.nn.Softmax()
        self._fc_prediction = Linear(self.fc_hid_dim, self.class_dim)

    @declarative
    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (paddle.reshape(inputs, [-1, 1]) != self.dict_dim).astype(
            dtype='float32'
        )
        mask_emb = paddle.expand(o_np_mask, [-1, self.hid_dim])
        emb = emb * mask_emb
        emb = paddle.reshape(
            emb, shape=[-1, self.channels, self.seq_len, self.hid_dim]
        )
        conv_3 = self._simple_conv_pool_1(emb)
        fc_1 = self._fc1(conv_3)
        fc_1 = self._fc1_act(fc_1)
        prediction = self._fc_prediction(fc_1)
        prediction = self._fc1_act(prediction)

        cost = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)
        acc = paddle.static.accuracy(input=prediction, label=label)
        return avg_cost, prediction, acc


class BOW(fluid.dygraph.Layer):
    def __init__(self, dict_dim, batch_size, seq_len):
        super().__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            self.dict_dim + 1,
            self.emb_dim,
            sparse=False,
        )
        self._fc1 = Linear(self.hid_dim, self.hid_dim)
        self._fc2 = Linear(self.hid_dim, self.fc_hid_dim)
        self._fc_prediction = Linear(self.fc_hid_dim, self.class_dim)

    @declarative
    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (paddle.reshape(inputs, [-1, 1]) != self.dict_dim).astype(
            dtype='float32'
        )
        mask_emb = paddle.expand(o_np_mask, [-1, self.hid_dim])
        emb = emb * mask_emb
        emb = paddle.reshape(emb, shape=[-1, self.seq_len, self.hid_dim])
        bow_1 = paddle.sum(emb, axis=1)
        bow_1 = paddle.tanh(bow_1)
        fc_1 = self._fc1(bow_1)
        fc_1 = paddle.tanh(fc_1)
        fc_2 = self._fc2(fc_1)
        fc_2 = paddle.tanh(fc_2)
        prediction = self._fc_prediction(fc_2)
        prediction = paddle.nn.functional.softmax(prediction)

        cost = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)
        acc = paddle.static.accuracy(input=prediction, label=label)
        return avg_cost, prediction, acc


class GRU(fluid.dygraph.Layer):
    def __init__(self, dict_dim, batch_size, seq_len):
        super().__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            self.dict_dim + 1,
            self.emb_dim,
            weight_attr=fluid.ParamAttr(learning_rate=30),
            sparse=False,
        )
        h_0 = np.zeros((self.batch_size, self.hid_dim), dtype="float32")
        h_0 = to_variable(h_0)
        self._fc1 = Linear(self.hid_dim, self.hid_dim * 3)
        self._fc2 = Linear(self.hid_dim, self.fc_hid_dim)
        self._fc_prediction = Linear(self.fc_hid_dim, self.class_dim)
        self._gru = DynamicGRU(size=self.hid_dim, h_0=h_0)

    @declarative
    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (paddle.reshape(inputs, [-1, 1]) != self.dict_dim).astype(
            'float32'
        )
        mask_emb = paddle.expand(o_np_mask, [-1, self.hid_dim])
        emb = emb * mask_emb
        emb = paddle.reshape(emb, shape=[self.batch_size, -1, self.hid_dim])
        fc_1 = self._fc1(emb)
        gru_hidden = self._gru(fc_1)
        gru_hidden = paddle.max(gru_hidden, axis=1)
        tanh_1 = paddle.tanh(gru_hidden)
        fc_2 = self._fc2(tanh_1)
        fc_2 = paddle.tanh(fc_2)
        prediction = self._fc_prediction(fc_2)
        prediction = paddle.nn.functional.softmax(prediction)
        cost = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)
        acc = paddle.static.accuracy(input=prediction, label=label)
        return avg_cost, prediction, acc


class BiGRU(fluid.dygraph.Layer):
    def __init__(self, dict_dim, batch_size, seq_len):
        super().__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding = Embedding(
            self.dict_dim + 1,
            self.emb_dim,
            weight_attr=fluid.ParamAttr(learning_rate=30),
            sparse=False,
        )
        h_0 = np.zeros((self.batch_size, self.hid_dim), dtype="float32")
        h_0 = to_variable(h_0)
        self._fc1 = Linear(self.hid_dim, self.hid_dim * 3)
        self._fc2 = Linear(self.hid_dim * 2, self.fc_hid_dim)
        self._fc_prediction = Linear(self.fc_hid_dim, self.class_dim)
        self._gru_forward = DynamicGRU(
            size=self.hid_dim, h_0=h_0, is_reverse=False
        )
        self._gru_backward = DynamicGRU(
            size=self.hid_dim, h_0=h_0, is_reverse=True
        )

    @declarative
    def forward(self, inputs, label=None):
        emb = self.embedding(inputs)
        o_np_mask = (paddle.reshape(inputs, [-1, 1]) != self.dict_dim).astype(
            'float32'
        )
        mask_emb = paddle.expand(o_np_mask, [-1, self.hid_dim])

        emb = emb * mask_emb
        emb = paddle.reshape(emb, shape=[self.batch_size, -1, self.hid_dim])
        fc_1 = self._fc1(emb)
        gru_forward = self._gru_forward(fc_1)
        gru_backward = self._gru_backward(fc_1)
        gru_forward_tanh = paddle.tanh(gru_forward)
        gru_backward_tanh = paddle.tanh(gru_backward)
        encoded_vector = fluid.layers.concat(
            input=[gru_forward_tanh, gru_backward_tanh], axis=2
        )
        encoded_vector = paddle.max(encoded_vector, axis=1)
        fc_2 = self._fc2(encoded_vector)
        fc_2 = paddle.tanh(fc_2)
        prediction = self._fc_prediction(fc_2)
        prediction = paddle.nn.functional.softmax(prediction)
        # TODO(Aurelius84): Uncomment the following codes when we support return variable-length vars.
        # if label is not None:
        cost = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)
        acc = paddle.static.accuracy(input=prediction, label=label)
        return avg_cost, prediction, acc
        # else:
        #     return prediction


def fake_data_reader(class_num, vocab_size, batch_size, padding_size):
    local_random = np.random.RandomState(SEED)

    def reader():
        batch_data = []
        while True:
            label = local_random.randint(0, class_num)
            seq_len = local_random.randint(
                padding_size // 2, int(padding_size * 1.2)
            )
            word_ids = local_random.randint(0, vocab_size, [seq_len]).tolist()
            word_ids = word_ids[:padding_size] + [vocab_size] * (
                padding_size - seq_len
            )
            batch_data.append((word_ids, [label], seq_len))
            if len(batch_data) == batch_size:
                yield batch_data
                batch_data = []

    return reader


class Args:
    epoch = 1
    batch_size = 4
    class_num = 2
    lr = 0.01
    vocab_size = 1000
    padding_size = 50
    log_step = 5
    train_step = 10


def train(args, to_static):
    program_translator.enable(to_static)
    place = (
        fluid.CUDAPlace(0)
        if fluid.is_compiled_with_cuda()
        else fluid.CPUPlace()
    )

    with fluid.dygraph.guard(place):
        np.random.seed(SEED)
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        train_reader = fake_data_reader(
            args.class_num, args.vocab_size, args.batch_size, args.padding_size
        )
        train_loader = fluid.io.DataLoader.from_generator(capacity=24)
        train_loader.set_sample_list_generator(train_reader)

        if args.model_type == 'cnn_net':
            model = CNN(args.vocab_size, args.batch_size, args.padding_size)
        elif args.model_type == 'bow_net':
            model = BOW(args.vocab_size, args.batch_size, args.padding_size)
        elif args.model_type == 'gru_net':
            model = GRU(args.vocab_size, args.batch_size, args.padding_size)
        elif args.model_type == 'bigru_net':
            model = BiGRU(args.vocab_size, args.batch_size, args.padding_size)
        sgd_optimizer = fluid.optimizer.Adagrad(
            learning_rate=args.lr, parameter_list=model.parameters()
        )

        loss_data = []
        for eop in range(args.epoch):
            time_begin = time.time()
            for batch_id, data in enumerate(train_loader()):
                word_ids, labels, seq_lens = data
                doc = to_variable(word_ids.numpy().reshape(-1)).astype('int64')
                label = labels.astype('int64')

                model.train()
                avg_cost, prediction, acc = model(doc, label)
                loss_data.append(avg_cost.numpy()[0])

                avg_cost.backward()
                sgd_optimizer.minimize(avg_cost)
                model.clear_gradients()

                if batch_id % args.log_step == 0:
                    time_end = time.time()
                    used_time = time_end - time_begin
                    # used_time may be 0.0, cause zero division error
                    if used_time < 1e-5:
                        used_time = 1e-5
                    print(
                        "step: %d, ave loss: %f, speed: %f steps/s"
                        % (
                            batch_id,
                            avg_cost.numpy()[0],
                            args.log_step / used_time,
                        )
                    )
                    time_begin = time.time()

                if batch_id == args.train_step:
                    break
                batch_id += 1
    return loss_data


class TestSentiment(unittest.TestCase):
    def setUp(self):
        self.args = Args()

    def train_model(self, model_type='cnn_net'):
        self.args.model_type = model_type
        st_out = train(self.args, True)
        dy_out = train(self.args, False)
        np.testing.assert_allclose(
            dy_out,
            st_out,
            rtol=1e-05,
            err_msg='dy_out:\n {}\n st_out:\n {}'.format(dy_out, st_out),
        )

    def test_train(self):
        model_types = ['cnn_net', 'bow_net', 'gru_net', 'bigru_net']
        for model_type in model_types:
            print('training %s ....' % model_type)
            self.train_model(model_type)


if __name__ == '__main__':
    unittest.main()
