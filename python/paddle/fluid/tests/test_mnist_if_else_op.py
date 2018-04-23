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

import paddle
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard, default_main_program, default_startup_program
from paddle.fluid.executor import Executor
from paddle.fluid.optimizer import MomentumOptimizer
import paddle.fluid.core as core
import unittest
import numpy as np


class TestMNISTIfElseOp(unittest.TestCase):
    def test_raw_api(self):
        prog = Program()
        startup_prog = Program()
        with program_guard(prog, startup_prog):
            image = layers.data(name='x', shape=[784], dtype='float32')

            label = layers.data(name='y', shape=[1], dtype='int64')

            limit = layers.fill_constant_batch_size_like(
                input=label, dtype='int64', shape=[1], value=5.0)
            cond = layers.less_than(x=label, y=limit)
            true_image, false_image = layers.split_lod_tensor(
                input=image, mask=cond)

            true_out = layers.create_tensor(dtype='float32')
            true_cond = layers.ConditionalBlock([true_image])

            with true_cond.block():
                hidden = layers.fc(input=true_image, size=100, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                layers.assign(input=prob, output=true_out)

            false_out = layers.create_tensor(dtype='float32')
            false_cond = layers.ConditionalBlock([false_image])

            with false_cond.block():
                hidden = layers.fc(input=false_image, size=200, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                layers.assign(input=prob, output=false_out)

            prob = layers.merge_lod_tensor(
                in_true=true_out, in_false=false_out, mask=cond, x=image)
            loss = layers.cross_entropy(input=prob, label=label)
            avg_loss = layers.mean(loss)

            optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
            optimizer.minimize(avg_loss, startup_prog)

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=200)

        place = core.CPUPlace()
        exe = Executor(place)

        exe.run(startup_prog)
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                x_data = np.array(map(lambda x: x[0], data)).astype("float32")
                y_data = np.array(map(lambda x: x[1], data)).astype("int64")
                y_data = np.expand_dims(y_data, axis=1)

                outs = exe.run(prog,
                               feed={'x': x_data,
                                     'y': y_data},
                               fetch_list=[avg_loss])
                print outs[0]
                if outs[0] < 1.0:
                    return
        self.assertFalse(True)

    def test_ifelse(self):
        prog = Program()
        startup_prog = Program()
        with program_guard(prog, startup_prog):
            image = layers.data(name='x', shape=[784], dtype='float32')

            label = layers.data(name='y', shape=[1], dtype='int64')

            limit = layers.fill_constant_batch_size_like(
                input=label, dtype='int64', shape=[1], value=5.0)
            cond = layers.less_than(x=label, y=limit)
            ie = layers.IfElse(cond)

            with ie.true_block():
                true_image = ie.input(image)
                hidden = layers.fc(input=true_image, size=100, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                ie.output(prob)

            with ie.false_block():
                false_image = ie.input(image)
                hidden = layers.fc(input=false_image, size=200, act='tanh')
                prob = layers.fc(input=hidden, size=10, act='softmax')
                ie.output(prob)

            prob = ie()
            loss = layers.cross_entropy(input=prob[0], label=label)
            avg_loss = layers.mean(loss)

            optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)
            optimizer.minimize(avg_loss, startup_prog)
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=200)

        place = core.CPUPlace()
        exe = Executor(place)

        exe.run(kwargs['startup_program'])
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                x_data = np.array(map(lambda x: x[0], data)).astype("float32")
                y_data = np.array(map(lambda x: x[1], data)).astype("int64")
                y_data = y_data.reshape((y_data.shape[0], 1))

                outs = exe.run(kwargs['main_program'],
                               feed={'x': x_data,
                                     'y': y_data},
                               fetch_list=[avg_loss])
                print outs[0]
                if outs[0] < 1.0:
                    return
        self.assertFalse(True)


if __name__ == '__main__':
    # temp disable if else unittest since it could be buggy.
    exit(0)
