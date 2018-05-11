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

import paddle
import paddle.fluid as fluid
from functools import partial

CLASS_DIM = 2
EMB_DIM = 128
LSTM_SIZE = 128


def dyn_rnn_lstm(data, input_dim, class_dim, emb_dim, lstm_size):
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)
    sentence = fluid.layers.fc(input=emb, size=lstm_size, act='tanh')

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        prev_hidden = rnn.memory(value=0.0, shape=[lstm_size])
        prev_cell = rnn.memory(value=0.0, shape=[lstm_size])

        def gate_common(ipt, hidden, size):
            gate0 = fluid.layers.fc(input=ipt, size=size, bias_attr=True)
            gate1 = fluid.layers.fc(input=hidden, size=size, bias_attr=False)
            return gate0 + gate1

        forget_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                         lstm_size))
        input_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                        lstm_size))
        output_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                         lstm_size))
        cell_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                       lstm_size))

        cell = forget_gate * prev_cell + input_gate * cell_gate
        hidden = output_gate * fluid.layers.tanh(x=cell)
        rnn.update_memory(prev_cell, cell)
        rnn.update_memory(prev_hidden, hidden)
        rnn.output(hidden)

    last = fluid.layers.sequence_last_step(rnn())
    prediction = fluid.layers.fc(input=last, size=class_dim, act="softmax")
    return prediction


def inference_network(word_dict):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    dict_dim = len(word_dict)
    net = dyn_rnn_lstm(data, dict_dim, CLASS_DIM, EMB_DIM, LSTM_SIZE)
    return net


def train_network(word_dict):
    prediction = inference_network(word_dict)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, accuracy


def train(use_cuda, save_path):
    BATCH_SIZE = 128
    EPOCH_NUM = 5

    word_dict = paddle.dataset.imdb.word_dict()

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=BATCH_SIZE)

    test_data = paddle.batch(
        paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)

    def event_handler(event):
        if isinstance(event, fluid.EndIteration):
            if (event.batch_id % 10) == 0:
                avg_cost, accuracy = trainer.test(reader=test_data)

                print('BatchID {1:04}, Loss {2:2.2}, Acc {3:2.2}'.format(
                    event.batch_id + 1, avg_cost, accuracy))

                if accuracy > 0.01:  # Low threshold for speeding up CI
                    trainer.params.save(save_path)
                    return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    trainer = fluid.Trainer(
        partial(train_network, word_dict),
        optimizer=fluid.optimizer.Adagrad(learning_rate=0.002),
        place=place,
        event_handler=event_handler)

    trainer.train(train_data, EPOCH_NUM, event_handler=event_handler)


def infer(use_cuda, save_path):
    params = fluid.Params(save_path)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    word_dict = paddle.dataset.imdb.word_dict()
    inferencer = fluid.Inferencer(
        partial(inference_network, word_dict), params, place=place)

    def create_random_lodtensor(lod, place, low, high):
        data = np.random.random_integers(low, high,
                                         [lod[-1], 1]).astype("int64")
        res = fluid.LoDTensor()
        res.set(data, place)
        res.set_lod([lod])
        return res

    lod = [0, 4, 10]
    tensor_words = create_random_lodtensor(
        lod, place, low=0, high=len(word_dict) - 1)
    results = inferencer.infer({'words': tensor_words})
    print("infer results: ", results)


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_path = "understand_sentiment_stacked_lstm.inference.model"
    train(use_cuda, save_path)
    infer(use_cuda, save_path)


if __name__ == '__main__':
    for use_cuda in (False, True):
        main(use_cuda=use_cuda)
