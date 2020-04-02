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

import six
import unittest

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler
import numpy as np
from fake_reader import fake_imdb_reader


def train(network, use_cuda, use_parallel_executor, batch_size=32, pass_num=2):
    if use_cuda and not core.is_compiled_with_cuda():
        print('Skip use_cuda=True because Paddle is not compiled with cuda')
        return

    if use_parallel_executor and os.name == 'nt':
        print(
            'Skip use_parallel_executor=True because Paddle comes without parallel support on windows'
        )
        return

    word_dict_size = 5147
    reader = fake_imdb_reader(word_dict_size, batch_size * 40)
    train_reader = paddle.batch(reader, batch_size=batch_size)

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    cost = network(data, label, word_dict_size)
    cost.persistable = True
    optimizer = fluid.optimizer.Adagrad(learning_rate=0.2)
    optimizer.minimize(cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
    reader = feeder.decorate_reader(
        train_reader, multi_devices=use_parallel_executor)

    exe = fluid.Executor(place)
    fluid.default_startup_program().random_seed = 1
    fluid.default_main_program().random_seed = 1
    exe.run(fluid.default_startup_program())

    train_cp = fluid.default_main_program()
    if use_parallel_executor:
        train_cp = compiler.CompiledProgram(fluid.default_main_program(
        )).with_data_parallel(loss_name=cost.name)
        fetch_list = [cost.name]
    else:
        fetch_list = [cost]

    for pass_id in six.moves.xrange(pass_num):
        batch_id = 0
        for data in reader():
            exe.run(train_cp,
                    feed=data,
                    fetch_list=fetch_list if batch_id % 4 == 0 else [])
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
            for use_parallel_executor in [False, True]:
                print('network: {}, use_cuda: {}, use_parallel_executor: {}'.
                      format(self.net.__name__, use_cuda,
                             use_parallel_executor))
                with fluid.program_guard(fluid.Program(), fluid.Program()):
                    with fluid.scope_guard(core.Scope()):
                        train(self.net, use_cuda, use_parallel_executor)
