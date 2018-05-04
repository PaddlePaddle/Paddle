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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy

WORD_DICT, VERB_DICT, LABEL_DICT = paddle.dataset.conll05.get_dict()
WORD_DICT_LEN = len(WORD_DICT)
LABEL_DICT_LEN = len(LABEL_DICT)
PRED_DICT_LEN = len(VERB_DICT)
MARK_DICT_LEN = 2


def lstm_net(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark):
    WORD_DIM = 32
    MARK_DIM = 5
    HIDDEN_DIM = 512
    DEPTH = 8
    EMBEDDING_NAME = 'emb'

    # Data definitions
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

    # 8 features
    predicate_embedding = fluid.layers.embedding(
        input=predicate,
        size=[PRED_DICT_LEN, WORD_DIM],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr='vemb')

    mark_embedding = fluid.layers.embedding(
        input=mark,
        size=[MARK_DICT_LEN, MARK_DIM],
        dtype='float32',
        is_sparse=IS_SPARSE)

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        fluid.layers.embedding(
            size=[WORD_DICT_LEN, WORD_DIM],
            input=x,
            param_attr=fluid.ParamAttr(
                name=EMBEDDING_NAME, trainable=False)) for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0_layers = [
        fluid.layers.fc(input=emb, size=HIDDEN_DIM, act='tanh')
        for emb in emb_layers
    ]

    hidden_0 = fluid.layers.sums(input=hidden_0_layers)

    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=HIDDEN_DIM,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, DEPTH):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=HIDDEN_DIM, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=HIDDEN_DIM, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=HIDDEN_DIM,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    feature_out = fluid.layers.sums(input=[
        fluid.layers.fc(input=input_tmp[0], size=LABEL_DICT_LEN, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=LABEL_DICT_LEN, act='tanh')
    ])

    return feature_out


def inference_network():
    predict = lstm_net(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2,
                       mark)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    return crf_decode


def train_network():
    MIX_HIDDEN_LR = 1e-3

    predict = lstm_net(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2,
                       mark)
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)
    crf_cost = fluid.layers.linear_chain_crf(
        input=predict,
        label=target,
        param_attr=fluid.ParamAttr(
            name='crfw', learning_rate=MIX_HIDDEN_LR))
    avg_cost = fluid.layers.mean(crf_cost)

    return avg_cost


def train(use_cuda, save_path):
    BATCH_SIZE = 128
    EPOCH_NUM = 1

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.conll05.train(), buf_size=8192),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.conll05.test(), batch_size=BATCH_SIZE)

    def event_handler(event):
        if isinstance(event, fluid.EndIteration):
            if (event.batch_id % 10) == 0:
                avg_cost = trainer.test(reader=test_reader)

                print('BatchID {1:04}, Loss {2:2.2}'.format(event.batch_id + 1,
                                                            avg_cost))

                if avg_cost > 0.01:  # Low threshold for speeding up CI
                    trainer.save_params(save_path)
                    return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=0.01,
            decay_steps=100000,
            decay_rate=0.5,
            staircase=True))
    trainer = fluid.Trainer(train_network, optimizer=sgd_optimizer, place=place)
    trainer.train(train_reader, EPOCH_NUM, event_handler=event_handler)


def infer(use_cuda, save_path):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        inference_program, param_path=save_path, place=place)

    def create_random_lodtensor(lod, place, low, high):
        data = np.random.random_integers(low, high,
                                         [lod[-1], 1]).astype("int64")
        res = fluid.LoDTensor()
        res.set(data, place)
        res.set_lod([lod])
        return res

    # Create an input example
    lod = [0, 4, 10]
    word = create_random_lodtensor(lod, place, low=0, high=WORD_DICT_LEN - 1)
    pred = create_random_lodtensor(lod, place, low=0, high=PRED_DICT_LEN - 1)
    ctx_n2 = create_random_lodtensor(lod, place, low=0, high=WORD_DICT_LEN - 1)
    ctx_n1 = create_random_lodtensor(lod, place, low=0, high=WORD_DICT_LEN - 1)
    ctx_0 = create_random_lodtensor(lod, place, low=0, high=WORD_DICT_LEN - 1)
    ctx_p1 = create_random_lodtensor(lod, place, low=0, high=WORD_DICT_LEN - 1)
    ctx_p2 = create_random_lodtensor(lod, place, low=0, high=WORD_DICT_LEN - 1)
    mark = create_random_lodtensor(lod, place, low=0, high=MARK_DICT_LEN - 1)

    results = inferencer.infer({
        'word_data': word,
        'verb_data': pred,
        'ctx_n2_data': ctx_n2,
        'ctx_n1_data': ctx_n1,
        'ctx_0_data': ctx_0,
        'ctx_p1_data': ctx_p1,
        'ctx_p2_data': ctx_p2,
        'mark_data': mark
    })

    print("infer results: ", results)


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    save_path = "label_semantic_roles.inference.model"
    train(use_cuda, save_path)
    infer(use_cuda, save_path)


if __name__ == '__main__':
    for use_cuda in (False, True):
        main(use_cuda=use_cuda)
