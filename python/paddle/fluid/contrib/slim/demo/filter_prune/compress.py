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
from vgg import *
from paddle.fluid.contrib.slim import CompressPass
from paddle.fluid.contrib.slim import build_compressor
from paddle.fluid.contrib.slim import ImitationGraph


class TrainGraphPass(object):
    def __init__(self, optimizer, executor):
        self.optimizer = optimizer
        self.exe = executor

    def apply(self, graph):
        train_graph = graph.clone()
        startup_program = fluid.Program()
        with fluid.program_guard(train_graph.program, startup_program):
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            predict = train_graph.get_var(train_graph.out_nodes.pop('predict'))
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(cost)
            self.optimizer.minimize(avg_cost)
        train_graph.out_nodes['loss'] = avg_cost.name
        train_graph.in_nodes['label'] = label.name
        self.exe.run(startup_program, scope=train_graph.scope)
        return train_graph


class EvalGraphPass(object):
    def apply(self, graph):
        eval_graph = graph.clone()
        with fluid.program_guard(eval_graph.program):
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            predict = eval_graph.get_var(eval_graph.out_nodes.pop('predict'))
            acc = fluid.layers.accuracy(input=predict, label=label)
        eval_graph.out_nodes['acc'] = acc.name
        eval_graph.in_nodes['label'] = label.name
        return eval_graph


class Model(object):
    def __init__(slef):
        pass

    def _build_program(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            img = fluid.layers.data(
                name='img', shape=[1, 28, 28], dtype='float32')
            vgg = VGG11()
            predict = vgg.net(img, class_dim=10)
        return main_program, startup_program, {
            'img': img.name
        }, {
            'predict': predict.name
        }

    def compress(self):
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=128)

        eval_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=1)

        place = fluid.CUDAPlace(0)
        train_feed_list = ['img', 'label']
        eval_feed_list = ['img', 'label']
        exe = fluid.Executor(place)
        main_program, startup_program, in_dict, out_dict = self._build_program()
        exe.run(startup_program, scope=fluid.global_scope())

        tartget_graph = ImitationGraph(
            main_program,
            scope=fluid.global_scope(),
            in_nodes=in_dict,
            out_nodes=out_dict)
        scope = fluid.global_scope()
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        train_graph_pass = TrainGraphPass(optimizer, exe)
        eval_graph_pass = EvalGraphPass()

        com_pass = CompressPass(
            place=place,
            train_graph_pass=train_graph_pass,
            train_reader=train_reader,
            train_feed_list=train_feed_list,
            eval_graph_pass=eval_graph_pass,
            eval_reader=eval_reader,
            eval_feed_list=eval_feed_list)
        com_pass.config('./config.yaml')
        com_pass.apply(tartget_graph)


if __name__ == "__main__":
    model = Model()
    model.compress()
