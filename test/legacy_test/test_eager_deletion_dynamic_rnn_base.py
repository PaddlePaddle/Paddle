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

import os

os.environ['CPU_NUM'] = '2'

import unittest

from fake_reader import fake_imdb_reader

import paddle
from paddle import base
from paddle.base import core


def train(network, use_cuda, batch_size=32, pass_num=2):
    if use_cuda and not core.is_compiled_with_cuda():
        print('Skip use_cuda=True because Paddle is not compiled with cuda')
        return

    word_dict_size = 5147
    reader = fake_imdb_reader(word_dict_size, batch_size * 40)
    train_reader = paddle.batch(reader, batch_size=batch_size)

    data = paddle.static.data(name="words", shape=[-1, 1], dtype="int64")

    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    cost = network(data, label, word_dict_size)
    cost.persistable = True
    optimizer = paddle.optimizer.Adagrad(learning_rate=0.2)
    optimizer.minimize(cost)

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    feeder = base.DataFeeder(feed_list=[data, label], place=place)
    reader = feeder.feed(train_reader())

    exe = base.Executor(place)
    paddle.seed(1)
    exe.run(base.default_startup_program())

    train_cp = base.default_main_program()
    fetch_list = [cost]

    for pass_id in range(pass_num):
        batch_id = 0
        for data in reader():
            exe.run(
                train_cp,
                feed=data,
                fetch_list=fetch_list if batch_id % 4 == 0 else [],
            )
            batch_id += 1
            if batch_id > 16:
                break


class TestBase(unittest.TestCase):
    def setUp(self):
        self.net = None

    def test_network(self):
        if self.net is None:
            return

        for use_cuda in [True, False]:
            print(f'network: {self.net.__name__}, use_cuda: {use_cuda}')
            with base.program_guard(base.Program(), base.Program()):
                with base.scope_guard(core.Scope()):
                    train(self.net, use_cuda)
