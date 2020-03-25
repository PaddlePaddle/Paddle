#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import paddle.fluid.core as core
import paddle.fluid as fluid
from test_eager_deletion_padding_rnn import RNNConfig, PaddingRNNTestBase


class TestExecutor(unittest.TestCase):
    def net(self):
        lr = fluid.layers.data(
            name="lr", shape=[1], dtype='float32', append_batch_size=False)
        x = fluid.data(name="x", shape=[None, 1], dtype='float32')
        y = fluid.data(name="y", shape=[None, 1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

        opt = fluid.optimizer.Adam(learning_rate=lr)
        opt.minimize(avg_cost)

        return lr, avg_cost

    #NOTE(zhiqiu): Historically, Program supports feeding scalar.
    def test_program_feed_scalar(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                cpu = fluid.CPUPlace()
                exe = fluid.Executor(cpu)
                lr, cost = self.net()
                exe.run(startup_program)
                train_data = numpy.array(
                    [[1.0], [2.0], [3.0], [4.0]]).astype('float32')
                y_true = numpy.array(
                    [[2.0], [4.0], [6.0], [8.0]]).astype('float32')
                _lr, _, _ = exe.run(
                    feed={'x': train_data,
                          'y': y_true,
                          'lr': 0.01},
                    fetch_list=[lr, cost])
            self.assertEqual(_lr._dtype(), lr.dtype)

    #NOTE(zhiqiu): CompiledProgram does not support feeding scalar.
    def test_program_feed_scalar(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                cpu = fluid.CPUPlace()
                exe = fluid.Executor(cpu)
                lr, cost = self.net()
                compiled_prog = fluid.CompiledProgram(
                    main_program).with_data_parallel(loss_name=cost.name)
                exe.run(startup_program)
                train_data = numpy.array(
                    [[1.0], [2.0], [3.0], [4.0]]).astype('float32')
                y_true = numpy.array(
                    [[2.0], [4.0], [6.0], [8.0]]).astype('float32')
            self.assertRaises(
                AssertionError,
                exe.run,
                compiled_prog,
                feed={'x': train_data,
                      'y': y_true,
                      'lr': 0.01},
                fetch_list=[lr, cost])


if __name__ == '__main__':
    unittest.main()
