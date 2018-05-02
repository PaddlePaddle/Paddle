#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.fluid as fluid
import numpy as np
import math
import sys
from functools import partial

PASS_NUM = 100
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32


def create_random_lodtensor(lod, place, low, high):
    # The range of data elements is [low, high]
    data = np.random.random_integers(low, high, [lod[-1], 1]).astype("int64")
    res = fluid.LoDTensor()
    res.set(data, place)
    res.set_lod([lod])
    return res


word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)


def inference_network(is_sparse):
    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')

    embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=second_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=third_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_forth = fluid.layers.embedding(
        input=forth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
    hidden1 = fluid.layers.fc(input=concat_embed,
                              size=HIDDEN_SIZE,
                              act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word


def train_network(is_sparse):
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    predict_word = inference_network(is_sparse)
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def train(use_cuda, is_sparse, save_path):
    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    def event_handler(event):
        print type(event)
        if isinstance(event, fluid.EndEpochEvent):
            avg_cost = trainer.test(reader=paddle.dataset.imikolov.test(
                word_dict, N))

            if avg_cost < 5.0:
                trainer.params.save(save_path)
                return
            if math.isnan(avg_cost):
                sys.exit("got NaN loss, training failed.")

    trainer = fluid.Trainer(
        partial(train_network, is_sparse),
        fluid.optimizer.SGD(learning_rate=0.001),
        place=place)
    trainer.train(
        reader=train_reader, num_epochs=100, event_handler=event_handler)


def infer(use_cuda, save_path):
    params = fluid.Params(save_path)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(inference_network, params, place=place)

    lod = [0, 1]
    first_word = create_random_lodtensor(lod, place, low=0, high=dict_size - 1)
    second_word = create_random_lodtensor(lod, place, low=0, high=dict_size - 1)
    third_word = create_random_lodtensor(lod, place, low=0, high=dict_size - 1)
    fourth_word = create_random_lodtensor(lod, place, low=0, high=dict_size - 1)
    result = inferencer.infer({
        'firstw': first_word,
        'secondw': second_word,
        'thirdw': third_word,
        'forthw': fourth_word
    })
    print(result)


def main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    save_path = "word2vec.inference.model"
    train(use_cuda, is_sparse, save_path)
    infer(use_cuda, save_path)


if __name__ == '__main__':
    for use_cuda in (False, True):
        for is_sparse in (False, True):
            main(use_cuda=use_cuda, is_sparse=is_sparse)
