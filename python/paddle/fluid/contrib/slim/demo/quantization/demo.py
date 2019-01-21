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

import paddle.fluid as fluid
import paddle
import os
import sys
from collections import OrderedDict
from paddle.fluid.contrib.slim import CompressPass
from paddle.fluid.contrib.slim import build_compressor
from paddle.fluid.contrib.slim import ImitationGraph


class LinearModel(object):
    def __init__(slef):
        pass

    def train(self, use_cuda=False):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        startup_program.random_seed = 10
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=predict, label=y)
            avg_cost = fluid.layers.mean(cost)
            eval_program = train_program.clone(for_test=True)
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost)

        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        eval_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=1)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        train_feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        eval_feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(startup_program)

        train_feed_list = OrderedDict([('x', x.name), ('y', y.name)])
        train_fetch_list = OrderedDict([('cost', avg_cost.name)])
        eval_feed_list = OrderedDict([('x', x.name), ('y', y.name)])
        eval_fetch_list = OrderedDict([('cost', avg_cost.name)])

        comp = CompressPass(
            place,
            fluid.global_scope(),
            train_program,
            train_reader=train_reader,
            train_feed_list=train_feed_list,
            train_fetch_list=train_fetch_list,
            eval_program=eval_program,
            eval_reader=eval_reader,
            eval_feed_list=eval_feed_list,
            eval_fetch_list=eval_fetch_list)
        comp.config('./config_static.yaml')
        comp.run()


if __name__ == "__main__":
    model = LinearModel()
    model.train(use_cuda=True)
