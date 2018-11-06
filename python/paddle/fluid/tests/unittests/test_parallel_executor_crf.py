# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle.dataset.conll05 as conll05
import paddle.fluid as fluid
import unittest
import paddle
import numpy as np
import os

word_dict, verb_dict, label_dict = conll05.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_dict_len = len(verb_dict)
mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8
mix_hidden_lr = 1e-3
embedding_name = 'emb'


def db_lstm(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark,
            is_sparse, **ignored):
    # 8 features
    predicate_embedding = fluid.layers.embedding(
        input=predicate,
        is_sparse=is_sparse,
        size=[pred_dict_len, word_dim],
        dtype='float32',
        param_attr='vemb')

    mark_embedding = fluid.layers.embedding(
        input=mark,
        is_sparse=is_sparse,
        size=[mark_dict_len, mark_dim],
        dtype='float32')

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        fluid.layers.embedding(
            size=[word_dict_len, word_dim],
            is_sparse=is_sparse,
            input=x,
            param_attr=fluid.ParamAttr(
                name=embedding_name, trainable=False)) for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0_layers = [
        fluid.layers.fc(input=emb, size=hidden_dim, act='tanh')
        for emb in emb_layers
    ]

    hidden_0 = fluid.layers.sums(input=hidden_0_layers)

    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    feature_out = fluid.layers.sums(input=[
        fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')
    ])

    return feature_out


class TestCRFModel(unittest.TestCase):
    def check_network_convergence(self,
                                  is_sparse,
                                  build_strategy=None,
                                  use_cuda=True):
        os.environ['CPU_NUM'] = str(4)
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            word = fluid.layers.data(
                name='word_data', shape=[1], dtype='int64', lod_level=1)
            predicate = fluid.layers.data(
                name='verb_data', shape=[1], dtype='int64', lod_level=1)
            ctx_n2 = fluid.layers.data(
                name='ctx_n2_data', shape=[1], dtype='int64', lod_level=1)
            ctx_n1 = fluid.layers.data(
                name='ctx_n1_data', shape=[1], dtype='int64', lod_level=1)
            ctx_0 = fluid.layers.data(
                name='ctx_0_data', shape=[1], dtype='int64', lod_level=1)
            ctx_p1 = fluid.layers.data(
                name='ctx_p1_data', shape=[1], dtype='int64', lod_level=1)
            ctx_p2 = fluid.layers.data(
                name='ctx_p2_data', shape=[1], dtype='int64', lod_level=1)
            mark = fluid.layers.data(
                name='mark_data', shape=[1], dtype='int64', lod_level=1)

            feature_out = db_lstm(**locals())
            target = fluid.layers.data(
                name='target', shape=[1], dtype='int64', lod_level=1)
            crf_cost = fluid.layers.linear_chain_crf(
                input=feature_out,
                label=target,
                param_attr=fluid.ParamAttr(
                    name='crfw', learning_rate=1e-1))
            avg_cost = fluid.layers.mean(crf_cost)

            sgd_optimizer = fluid.optimizer.SGD(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=0.01,
                    decay_steps=100000,
                    decay_rate=0.5,
                    staircase=True))
            sgd_optimizer.minimize(avg_cost)

            train_data = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.conll05.test(), buf_size=8192),
                batch_size=16)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup)

            pe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                loss_name=avg_cost.name,
                build_strategy=build_strategy)

            feeder = fluid.DataFeeder(
                feed_list=[
                    word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, predicate,
                    mark, target
                ],
                place=fluid.CPUPlace())

            data = train_data()
            for i in range(10):
                cur_batch = next(data)
                print(pe.run(feed=feeder.feed(cur_batch),
                             fetch_list=[avg_cost.name])[0])

    def test_update_sparse_parameter_all_reduce(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
        self.check_network_convergence(
            is_sparse=True, build_strategy=build_strategy, use_cuda=True)
        self.check_network_convergence(
            is_sparse=True, build_strategy=build_strategy, use_cuda=False)

    def test_update_dense_parameter_all_reduce(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce
        self.check_network_convergence(
            is_sparse=False, build_strategy=build_strategy, use_cuda=True)
        self.check_network_convergence(
            is_sparse=False, build_strategy=build_strategy, use_cuda=False)

    def test_update_sparse_parameter_reduce(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        self.check_network_convergence(
            is_sparse=True, build_strategy=build_strategy, use_cuda=True)
        self.check_network_convergence(
            is_sparse=True, build_strategy=build_strategy, use_cuda=False)

    def test_update_dense_parameter_reduce(self):
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        self.check_network_convergence(
            is_sparse=False, build_strategy=build_strategy, use_cuda=True)
        self.check_network_convergence(
            is_sparse=False, build_strategy=build_strategy, use_cuda=False)


if __name__ == '__main__':
    unittest.main()
