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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle
import os
import random
import time

import numpy
import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

word_dict = imdb.word_dict()


def crop_sentence(reader, crop_size):
    unk_value = word_dict['<unk>']

    def __impl__():
        for item in reader():
            if len([x for x in item[0] if x != unk_value]) < crop_size:
                yield item

    return __impl__


def lstm_net(sentence, lstm_size):
    sentence = fluid.layers.fc(input=sentence, size=lstm_size, act='tanh')

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        prev_hidden = rnn.memory(value=0.0, shape=[lstm_size])
        prev_cell = rnn.memory(value=0.0, shape=[lstm_size])

        def gate_common(
                ipt,
                hidden,
                size, ):
            gate0 = fluid.layers.fc(input=ipt, size=size, bias_attr=True)
            gate1 = fluid.layers.fc(input=hidden, size=size, bias_attr=False)
            gate = fluid.layers.sums(input=[gate0, gate1])
            return gate

        forget_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, lstm_size))
        input_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, lstm_size))
        output_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, lstm_size))
        cell_gate = fluid.layers.tanh(
            x=gate_common(word, prev_hidden, lstm_size))

        cell = fluid.layers.sums(input=[
            fluid.layers.elementwise_mul(
                x=forget_gate, y=prev_cell), fluid.layers.elementwise_mul(
                    x=input_gate, y=cell_gate)
        ])

        hidden = fluid.layers.elementwise_mul(
            x=output_gate, y=fluid.layers.tanh(x=cell))

        rnn.update_memory(prev_cell, cell)
        rnn.update_memory(prev_hidden, hidden)
        rnn.output(hidden)

    last = fluid.layers.sequence_pool(rnn(), 'last')
    logit = fluid.layers.fc(input=last, size=2, act='softmax')
    return logit


def get_model(args, is_train, main_prog, startup_prog):
    if args.use_reader_op:
        raise Exception(
            "stacked_dynamic_lstm do not support reader op for now.")
    lstm_size = 512
    emb_dim = 512
    crop_size = 1500

    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            data = fluid.layers.data(
                name="words", shape=[1], lod_level=1, dtype='int64')
            sentence = fluid.layers.embedding(
                input=data, size=[len(word_dict), emb_dim])
            logit = lstm_net(sentence, lstm_size)
            loss = fluid.layers.cross_entropy(
                input=logit,
                label=fluid.layers.data(
                    name='label', shape=[1], dtype='int64'))
            loss = fluid.layers.mean(x=loss)

            # add acc
            batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
            batch_acc = fluid.layers.accuracy(input=logit, label=fluid.layers.data(name='label', \
                        shape=[1], dtype='int64'), total=batch_size_tensor)

            if is_train:
                adam = fluid.optimizer.Adam()
                adam.minimize(loss)

    if is_train:
        reader = crop_sentence(imdb.train(word_dict), crop_size)
    else:
        reader = crop_sentence(imdb.test(word_dict), crop_size)

    batched_reader = paddle.batch(
        paddle.reader.shuffle(
            reader, buf_size=25000),
        batch_size=args.batch_size * args.gpus)

    return loss, adam, [batch_acc], batched_reader, None
