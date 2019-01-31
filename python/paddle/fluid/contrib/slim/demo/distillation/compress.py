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
from resnet import *
from paddle.fluid.contrib.slim import CompressPass
from paddle.fluid.contrib.slim import build_compressor
from paddle.fluid.contrib.slim import ImitationGraph


class Model(object):
    def __init__(slef):
        pass

    def compress(self):

        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        resnet50 = ResNet50()
        predict = resnet50.net(img, class_dim=10)
        eval_program = fluid.default_main_program().clone(for_test=False)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)

        with fluid.program_guard(main_program=eval_program):
            acc = fluid.layers.accuracy(input=predict, label=label)

        optimizer = fluid.optimizer.SGD(0.001)
        optimizer.minimize(avg_cost)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=32)
        eval_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=1)

        train_feed_list = {'img': img.name, 'label': label.name}
        train_fetch_list = {'cost': avg_cost.name}
        eval_feed_list = {'img': img.name, 'label': label.name}
        eval_fetch_list = {'acc': acc.name}

        # define teacher program 
        teacher_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(teacher_program, startup_program):
            img = fluid.layers.data(
                name='img', shape=[1, 28, 28], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            resnet101 = ResNet101()
            predict = resnet101.net(img, class_dim=10)
        exe.run(startup_program)

        com_pass = CompressPass(
            place,
            fluid.global_scope(),
            fluid.default_main_program(),
            train_reader=train_reader,
            train_feed_list=train_feed_list,
            train_fetch_list=train_fetch_list,
            eval_program=eval_program,
            eval_reader=eval_reader,
            eval_feed_list=eval_feed_list,
            eval_fetch_list=eval_fetch_list,
            teacher_programs=[teacher_program],
            optimizer=optimizer)
        com_pass.model_save_dir = './checkpoints'
        com_pass.config('./config.yaml')
        com_pass.run()


if __name__ == "__main__":
    model = Model()
    model.compress()
