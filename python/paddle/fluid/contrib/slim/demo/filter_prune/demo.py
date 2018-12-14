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
from paddle.fluid.contrib.slim import CompressPass
from paddle.fluid.contrib.slim import build_compressor
from paddle.fluid.contrib.slim import ImitationGraph


class LinearModel(object):
    def __init__(slef):
        pass

    def train(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        startup_program.random_seed = 10
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=predict, label=y)
            avg_cost = fluid.layers.mean(cost)
            eval_program = train_program.clone()
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost)

        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        eval_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=1)
        place = fluid.CPUPlace()
        train_feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        eval_feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(startup_program)
        train_metrics = {"loss": avg_cost.name}
        eval_metrics = {"loss": avg_cost.name}

        graph = ImitationGraph(train_program)
        config = './config.yaml'
        comp_pass = build_compressor(
            place,
            data_reader=train_reader,
            data_feeder=train_feeder,
            scope=fluid.global_scope(),
            metrics=train_metrics,
            epoch=1,
            config=config)
        comp_pass.apply(graph)


if __name__ == "__main__":
    model = LinearModel()
    model.train()
