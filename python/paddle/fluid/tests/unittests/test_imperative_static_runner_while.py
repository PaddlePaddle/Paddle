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

import unittest

import contextlib
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward
from test_imperative_base import new_program_scope


def simple_while_net():
    d0 = layers.data("d0", shape=[10], append_batch_size=False, dtype='float32')
    d1 = layers.data("d1", shape=[10], append_batch_size=False, dtype='float32')
    d2 = layers.data("d2", shape=[10], append_batch_size=False, dtype='float32')
    i = layers.zeros(shape=[1], dtype='int64')
    i.stop_gradient = True
    init = layers.zeros(shape=[10], dtype='float32')
    mem_array = layers.array_write(x=init, i=i)
    data_array = layers.array_write(x=d0, i=i)
    i = layers.increment(i)
    layers.array_write(d1, i, array=data_array)
    i = layers.increment(i)
    layers.array_write(d2, i, array=data_array)
    i = layers.zeros(shape=[1], dtype='int64')
    i.stop_gradient = True
    array_len = layers.fill_constant(shape=[1], dtype='int64', value=1)
    array_len.stop_gradient = True
    cond = layers.less_than(x=i, y=array_len)
    j = layers.fill_constant(shape=[1], dtype='int64', value=1)
    j.stop_gradient = True
    array_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
    array_len2.stop_gradient = True
    cond2 = layers.less_than(x=j, y=array_len2)
    while_op = layers.While(cond=cond)
    while_op2 = layers.While(cond=cond2)
    with while_op.block():
        d = layers.array_read(array=data_array, i=i)
        prev = layers.array_read(array=mem_array, i=i)
        result = layers.sums(input=[d, prev])

        i = layers.increment(x=i, in_place=True)
        layers.array_write(result, i=i, array=mem_array)
        layers.less_than(x=i, y=array_len, cond=cond)

        with while_op2.block():
            d2 = layers.array_read(array=data_array, i=j)
            prev2 = layers.array_read(array=mem_array, i=j)
            result2 = layers.sums(input=[d2, prev2])

            j = layers.increment(x=j, in_place=True)
            layers.array_write(result2, i=j, array=mem_array)
            layers.less_than(x=j, y=array_len2, cond=cond2)
    sum_result = layers.array_read(array=mem_array, i=j)
    return sum_result


class TestImperativeStaticModelRunnerWhile(unittest.TestCase):
    def setUp(self):
        self.seed = 90
        self.save_dirname = "while.inference.model"

    def random_feed(self):
        data = []
        for i in range(3):
            np.random.seed(90)
            data.append(np.random.random(size=[10]).astype('float32'))
        return data

    def train_and_save_model(self):
        sum_result = simple_while_net()
        loss = layers.mean(sum_result)

        append_backward(loss)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        d = self.random_feed()

        outs = exe.run(feed={'d0': d[0],
                             'd1': d[1],
                             'd2': d[2]},
                       fetch_list=[sum_result])

        fluid.io.save_inference_model(
            self.save_dirname, ['d0', 'd1', 'd2'], [sum_result],
            exe,
            main_program=fluid.default_main_program(),
            model_filename=None,
            params_filename=None)

    def load_and_train_dygraph(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True

            while_net = fluid.dygraph.StaticModelRunner(self.save_dirname)

            d = self.random_feed()

            loss = while_net(inputs={'d0': d[0], 'd1': d[1], 'd2': d[2]})

            avg_loss = fluid.layers.mean(loss)

            avg_loss.backward()

        return d, avg_loss.numpy()

    def load_and_train_static(self):
        with new_program_scope():
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed

            loss = simple_while_net()
            avg_loss = layers.mean(loss)

            append_backward(avg_loss)

            cpu = fluid.CPUPlace()
            exe = Executor(cpu)

            d = self.random_feed()

            outs = exe.run(feed={'d0': d[0],
                                 'd1': d[1],
                                 'd2': d[2]},
                           fetch_list=[avg_loss])

        return d, outs[0]

    def test_while_no_params_filename(self):
        # Phase 1. run and save static model
        self.train_and_save_model()

        # Phase 2. load model & train dygraph
        dy_input, dy_out = self.load_and_train_dygraph()

        static_input, static_out = self.load_and_train_static()

        # Phase 3. compare
        self.assertTrue(np.array_equal(static_input, dy_input))

        np_out = np.mean(np.sum(dy_input, axis=0))
        self.assertEqual(static_out, np_out)
        self.assertEqual(static_out, dy_out)


if __name__ == '__main__':
    unittest.main()
