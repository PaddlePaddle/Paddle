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

import unittest

import numpy
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid


class TestExecutor(unittest.TestCase):

    def net(self):
        lr = fluid.data(name="lr", shape=[1], dtype='float32')
        x = fluid.data(name="x", shape=[None, 1], dtype='float32')
        y = fluid.data(name="y", shape=[None, 1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)

        opt = fluid.optimizer.Adam(learning_rate=lr)
        opt.minimize(avg_cost)

        return lr, avg_cost

    def test_program_check_feed(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                cpu = fluid.CPUPlace()
                exe = fluid.Executor(cpu)
                lr, cost = self.net()
                exe.run(startup_program)
                train_data = [[1.0], [2.0], [3.0], [4.0]]
                y_true = [[2.0], [4.0], [6.0], [8.0]]
                a = 0
                with self.assertRaises(ValueError):
                    exe.run(feed={
                        'x': train_data,
                        'lr': a
                    },
                            fetch_list=[lr, cost],
                            return_numpy=False,
                            use_prune=True)

    def test_compiled_program_check_feed(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                cpu = fluid.CPUPlace()
                exe = fluid.Executor(cpu)
                lr, cost = self.net()
                exe.run(startup_program)
                compiled_prog = fluid.CompiledProgram(
                    main_program).with_data_parallel(loss_name=cost.name)
                train_data = [[1.0], [2.0], [3.0], [4.0]]
                y_true = [[2.0], [4.0], [6.0], [8.0]]
                a = 0
                with self.assertRaises(ValueError):
                    exe.run(compiled_prog,
                            feed={
                                'x': train_data,
                                'lr': a
                            },
                            fetch_list=[lr, cost],
                            return_numpy=False,
                            use_prune=True)


if __name__ == '__main__':
    unittest.main()
