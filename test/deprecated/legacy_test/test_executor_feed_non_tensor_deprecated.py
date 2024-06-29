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
from paddle import base

paddle.enable_static()


class TestExecutor(unittest.TestCase):
    def net(self):
        lr = 0.0
        x = paddle.static.data(name="x", shape=[None, 1], dtype='float32')
        y = paddle.static.data(name="y", shape=[None, 1], dtype='float32')
        y_predict = paddle.static.nn.fc(x, size=1)

        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)

        opt = paddle.optimizer.Adam(learning_rate=lr)
        opt.minimize(avg_cost)

        return paddle.to_tensor(lr), avg_cost

    def test_program_feed_float(self):
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                lr, cost = self.net()
                exe.run(startup_program)
                train_data = numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype(
                    'float32'
                )
                y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype(
                    'float32'
                )
                a = 0.01
                _lr, _ = exe.run(
                    feed={'x': train_data, 'y': y_true, 'lr': a},
                    fetch_list=[lr, cost],
                    return_numpy=False,
                )
            self.assertEqual(_lr._dtype(), lr.dtype)
            self.assertEqual(_lr._dtype(), paddle.float32)
            self.assertEqual(type(a), float)

    def test_program_feed_int(self):
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                lr, cost = self.net()
                exe.run(startup_program)
                train_data = numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype(
                    'float32'
                )
                y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype(
                    'float32'
                )
                a = 0
                _lr, _ = exe.run(
                    feed={'x': train_data, 'y': y_true, 'lr': a},
                    fetch_list=[lr, cost],
                    return_numpy=False,
                )
            self.assertEqual(_lr._dtype(), lr.dtype)
            self.assertEqual(_lr._dtype(), paddle.float32)
            self.assertEqual(type(a), int)

    def test_program_feed_list(self):
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                lr, cost = self.net()
                exe.run(startup_program)
                train_data = [[1.0], [2.0], [3.0], [4.0]]
                y_true = [[2.0], [4.0], [6.0], [8.0]]
                a = 0
                _lr, _ = exe.run(
                    feed={'x': train_data, 'y': y_true, 'lr': a},
                    fetch_list=[lr, cost],
                    return_numpy=False,
                )
            self.assertEqual(_lr._dtype(), lr.dtype)
            self.assertEqual(_lr._dtype(), paddle.float32)
            self.assertEqual(type(y_true), list)

    def test_compiled_program_feed_scalar(self):
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                lr, cost = self.net()
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                exe.run(startup_program)
                compiled_prog = base.CompiledProgram(main_program)
                train_data = numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype(
                    'float32'
                )
                y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype(
                    'float32'
                )
                a = 0.01
                _lr, _ = exe.run(
                    compiled_prog,
                    feed={'x': train_data, 'y': y_true, 'lr': a},
                    fetch_list=[lr, cost],
                    return_numpy=False,
                )
                self.assertEqual(_lr._dtype(), lr.dtype)
                self.assertEqual(_lr._dtype(), paddle.float32)
                self.assertEqual(type(a), float)


if __name__ == '__main__':
    unittest.main()
