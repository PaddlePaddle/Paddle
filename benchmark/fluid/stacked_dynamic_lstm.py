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
import paddle.v2 as paddle
import paddle.v2.dataset.imdb as imdb
import paddle.fluid as fluid
from paddle.v2 import batch
import paddle.fluid.profiler as profiler


def parse_args():
    parser = argparse.ArgumentParser("Understand Sentiment by Dynamic RNN.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--emb_dim',
        type=int,
        default=512,
        help='Dimension of embedding table. (default: %(default)d)')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512,
        help='Hidden size of lstm unit. (default: %(default)d)')
    parser.add_argument(
        '--pass_num',
        type=int,
        default=100,
        help='Epoch number to train. (default: %(default)d)')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--crop_size',
        type=int,
        default=int(os.environ.get('CROP_SIZE', '1500')),
        help='The max sentence length of input. Since this model use plain RNN,'
        ' Gradient could be explored if sentence is too long')
    args = parser.parse_args()
    return args


word_dict = imdb.word_dict()


def crop_sentence(reader, crop_size):
    unk_value = word_dict['<unk>']

    def __impl__():
        for item in reader():
            if len([x for x in item[0] if x != unk_value]) < crop_size:
                yield item

    return __impl__


def main():
    args = parse_args()
    lstm_size = args.hidden_dim

    data = fluid.layers.data(
        name="words", shape=[1], lod_level=1, dtype='int64')
    sentence = fluid.layers.embedding(
        input=data, size=[len(word_dict), args.emb_dim])

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
    loss = fluid.layers.cross_entropy(
        input=logit,
        label=fluid.layers.data(
            name='label', shape=[1], dtype='int64'))
    loss = fluid.layers.mean(x=loss)

    # add acc
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(input=logit, label=fluid.layers.data(name='label', \
                shape=[1], dtype='int64'), total=batch_size_tensor)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            target_vars=[batch_acc, batch_size_tensor])

    adam = fluid.optimizer.Adam()
    adam.minimize(loss)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    def train_loop(pass_num, crop_size):
        with profiler.profiler(args.device, 'total') as prof:
            for pass_id in range(pass_num):
                train_reader = batch(
                    paddle.reader.shuffle(
                        crop_sentence(imdb.train(word_dict), crop_size),
                        buf_size=25000),
                    batch_size=args.batch_size)
                word_nums = 0
                pass_start_time = time.time()
                for batch_id, data in enumerate(train_reader()):
                    tensor_words = to_lodtensor([x[0] for x in data], place)
                    for x in data:
                        word_nums += len(x[0])
                    label = numpy.array([x[1] for x in data]).astype("int64")
                    label = label.reshape((-1, 1))
                    loss_np, acc, weight = exe.run(
                        fluid.default_main_program(),
                        feed={"words": tensor_words,
                              "label": label},
                        fetch_list=[loss, batch_acc, batch_size_tensor])
                    print("pass_id=%d, batch_id=%d, loss=%f, acc=%f" %
                          (pass_id, batch_id, loss_np, acc))

                pass_end_time = time.time()
                time_consumed = pass_end_time - pass_start_time
                words_per_sec = word_nums / time_consumed
                print("pass_id=%d, sec/pass: %f, words/s: %f" %
                      (pass_id, time_consumed, words_per_sec))

    train_loop(args.pass_num, args.crop_size)


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = numpy.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


if __name__ == '__main__':
    main()
