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
import paddle.fluid as fluid


class TestExecutor(unittest.TestCase):
    def net(self):
        lr = fluid.data(name="lr", shape=[1], dtype='float32')
        x = fluid.data(name="x", shape=[None, 1], dtype='float32')
        y = fluid.data(name="y", shape=[None, 1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)

        opt = fluid.optimizer.Adam(learning_rate=lr)
        opt.minimize(avg_cost)

        return lr, avg_cost

    def test_program_feed_float(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                cpu = fluid.CPUPlace()
                exe = fluid.Executor(cpu)
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
            self.assertEqual(_lr._dtype(), fluid.core.VarDesc.VarType.FP32)
            self.assertEqual(type(a), float)

    def test_program_feed_int(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                cpu = fluid.CPUPlace()
                exe = fluid.Executor(cpu)
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
            self.assertEqual(_lr._dtype(), fluid.core.VarDesc.VarType.FP32)
            self.assertEqual(type(a), int)

    def test_program_feed_list(self):
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
                _lr, _ = exe.run(
                    feed={'x': train_data, 'y': y_true, 'lr': a},
                    fetch_list=[lr, cost],
                    return_numpy=False,
                )
            self.assertEqual(_lr._dtype(), lr.dtype)
            self.assertEqual(_lr._dtype(), fluid.core.VarDesc.VarType.FP32)
            self.assertEqual(type(y_true), list)

    def test_compiled_program_feed_scalar(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                lr, cost = self.net()
                cpu = fluid.CPUPlace()
                exe = fluid.Executor(cpu)
                exe.run(startup_program)
                compiled_prog = fluid.CompiledProgram(
                    main_program
                ).with_data_parallel(loss_name=cost.name)
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
                self.assertEqual(_lr._dtype(), fluid.core.VarDesc.VarType.FP32)
                self.assertEqual(type(a), float)


class TestAsLodTensor(unittest.TestCase):
    def test_as_lodtensor_int32(self):
        cpu = fluid.CPUPlace()
        tensor = fluid.executor._as_lodtensor(
            1.0, cpu, fluid.core.VarDesc.VarType.INT32
        )
        self.assertEqual(tensor._dtype(), fluid.core.VarDesc.VarType.INT32)

    def test_as_lodtensor_fp64(self):
        cpu = fluid.CPUPlace()
        tensor = fluid.executor._as_lodtensor(
            1, cpu, fluid.core.VarDesc.VarType.FP64
        )
        self.assertEqual(tensor._dtype(), fluid.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_assertion_error(self):
        cpu = fluid.CPUPlace()
        self.assertRaises(AssertionError, fluid.executor._as_lodtensor, 1, cpu)

    def test_as_lodtensor_type_error(self):
        cpu = fluid.CPUPlace()
        self.assertRaises(
            TypeError,
            fluid.executor._as_lodtensor,
            {"a": 1},
            cpu,
            fluid.core.VarDesc.VarType.INT32,
        )

    def test_as_lodtensor_list(self):
        cpu = fluid.CPUPlace()
        tensor = fluid.executor._as_lodtensor(
            [1, 2], cpu, fluid.core.VarDesc.VarType.FP64
        )
        self.assertEqual(tensor._dtype(), fluid.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_tuple(self):
        cpu = fluid.CPUPlace()
        tensor = fluid.executor._as_lodtensor(
            (1, 2), cpu, fluid.core.VarDesc.VarType.FP64
        )
        self.assertEqual(tensor._dtype(), fluid.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_nested_list(self):
        cpu = fluid.CPUPlace()
        self.assertRaises(
            TypeError,
            fluid.executor._as_lodtensor,
            [[1], [1, 2]],
            cpu,
            fluid.core.VarDesc.VarType.INT32,
        )


if __name__ == '__main__':
    unittest.main()
