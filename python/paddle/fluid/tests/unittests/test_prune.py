#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.compat as cpt
import numpy as np
import os
import contextlib


class TestPrune(unittest.TestCase):
    def net(self):
        x = fluid.layers.data(name='x', shape=[2], dtype='float32')
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        y = fluid.layers.fc(input=[x], size=2, act="softmax")
        loss = fluid.layers.cross_entropy(input=y, label=label)
        loss = fluid.layers.mean(x=loss)
        return x, y, label, loss

    def test_prune_with_input(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        pruned_program = program._prune_with_input(
            feeded_var_names=[y.name, label.name], targets=[loss])
        self.assertEqual(len(pruned_program.global_block().ops), 2)
        self.assertEqual([op.type for op in pruned_program.global_block().ops],
                         ["cross_entropy2", "mean"])

    def test_prune(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        pruned_program = program._prune(targets=[loss])
        self.assertEqual(len(pruned_program.global_block().ops), 5)
        self.assertEqual(
            [op.type for op in pruned_program.global_block().ops],
            ["mul", "elementwise_add", "softmax", "cross_entropy2", "mean"])

    def test_prune_target_not_list(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        pruned_program = program._prune(targets=loss)
        self.assertEqual(len(pruned_program.global_block().ops), 5)
        self.assertEqual(
            [op.type for op in pruned_program.global_block().ops],
            ["mul", "elementwise_add", "softmax", "cross_entropy2", "mean"])

    def test_prune_target_none(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "mean"
        ])
        try:
            pruned_program = program._prune(targets=None)
        except ValueError as e:
            self.assertEqual(
                "All targets of prune() can only be Variable or Operator.",
                cpt.get_exception_message(e))


def mock(self, program, feed, fetch):
    self.prune_called_times += 1
    return program


@contextlib.contextmanager
def _mock_guard(mock):

    original = fluid.Executor._prune_program
    fluid.Executor._prune_program = mock

    yield

    fluid.Executor._prune_program = original


class TestExecutorRunAutoPrune(unittest.TestCase):
    def net1(self):
        x = fluid.layers.data(name='x', shape=[2], dtype='float32')
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        w_param_attrs = fluid.ParamAttr(
            name="fc_weight",
            learning_rate=0.5,
            initializer=fluid.initializer.Constant(1.0),
            trainable=True)
        y = fluid.layers.fc(input=[x],
                            size=2,
                            act="softmax",
                            param_attr=w_param_attrs)
        loss1 = fluid.layers.cross_entropy(input=y, label=label)
        loss1 = fluid.layers.mean(x=loss1)
        loss2 = fluid.layers.cross_entropy(input=y, label=label)
        loss2 = fluid.layers.mean(x=loss2)
        loss1.persistable = True
        loss2.persistable = True
        return x, y, label, loss1, loss2, w_param_attrs

    def test_not_prune(self):
        """
        use_prune = False
        """
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={'x': x_np,
                                    'label': label_np},
                              fetch_list=[loss1.name],
                              use_prune=False)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNotNone(scope.find_var(loss2.name))

    def test_prune_fetches_without_optimizer(self):
        """
        Prune operators and operators which are not needed to generate 'fetches'. 
        """
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight1 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={'x': x_np,
                                    'label': label_np},
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))

                weight2 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                self.assertTrue(np.array_equal(weight1,
                                               weight2))  # weight not changed

    def test_prune_fetches_with_optimizer(self):
        """
        Prune operators and operators which are not needed to generate 'fetches'. 
        In train mode, the operators and operators in backward and optimization should be kept.
        """
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                x.persistable = True
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight1 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={'x': x_np,
                                    'label': label_np},
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))

                weight2 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                self.assertFalse(np.array_equal(weight1,
                                                weight2))  # weight changed

    def test_prune_compiled_program(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                compiled_prog = fluid.CompiledProgram(
                    program).with_data_parallel(
                        loss_name=loss1.name, places=fluid.CPUPlace())
                weight1 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(compiled_prog,
                              feed={'x': x_np,
                                    'label': label_np},
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))

                weight2 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                self.assertFalse(np.array_equal(weight1,
                                                weight2))  # weight changed

    def test_prune_feed_without_optimizer(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                #compiled_prog = fluid.CompiledProgram(program).with_data_parallel(loss_name=loss1.name, places=fluid.CPUPlace())
                weight1 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={y.name: x_np,
                                    'label': label_np},
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))

                weight2 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                self.assertTrue(np.array_equal(weight1,
                                               weight2))  # weight unchanged

    def test_prune_feed_with_optimizer(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                #compiled_prog = fluid.CompiledProgram(program).with_data_parallel(loss_name=loss1.name, places=fluid.CPUPlace())
                weight1 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                self.assertRaises(
                    Exception,
                    exe.run,
                    program,
                    feed={y.name: x_np,
                          'label': label_np},
                    fetch_list=[loss1.name],
                    use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))

    def test_prune_with_cache_program(self):
        '''
        When use_prune=True and use_program_cache=True, Executor should cache the pruned program.
        If in next run, the program, feed, fetch are not changed, Executor use the cached pruned program,
        and needn't to call  _prune_program() to prune the program.
        In this test, we hack the Executor._prune_program with a mock function which do nothing but increase
        Executor.prune_called_times, and we check prune_called_times equals 1 even if we called exe.run() 
        10 times with the same input arguments.
        '''

        with _mock_guard(mock):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.prune_called_times = 0

            program = framework.Program()
            startup_program = framework.Program()
            block = program.global_block()

            scope = fluid.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(program, startup_program):
                    (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                    x.persistable = True
                    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                    sgd_optimizer.minimize(loss1)
                    exe.run(startup_program)
                    weight1 = np.array(
                        scope.find_var(w_param_attrs.name).get_tensor())
                    x_np = np.random.random(size=(10, 2)).astype('float32')
                    label_np = np.random.randint(
                        1, size=(10, 1)).astype('int64')
                    for i in range(10):
                        res = exe.run(program,
                                      feed={'x': x_np,
                                            'label': label_np},
                                      fetch_list=[loss1.name],
                                      use_prune=True,
                                      use_program_cache=True)
                        if i == 0:
                            self.assertEqual(exe.prune_called_times, 1)
                        else:
                            self.assertEqual(exe.prune_called_times, 1)

    def test_prune_with_cache_compiled_program(self):
        '''
        When use_prune=True and use_program_cache=True, Executor should cache the pruned program.
        If in next run, the program, feed, fetch are not changed, Executor use the cached pruned program,
        and needn't to call  _prune_program() to prune the program.
        In this test, we hack the Executor._prune_program with a mock function which do nothing but increase
        Executor.prune_called_times, and we check prune_called_times equals 1 even if we called exe.run() 
        10 times with the same input arguments.
        '''

        with _mock_guard(mock):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.prune_called_times = 0
            program = framework.Program()
            startup_program = framework.Program()
            block = program.global_block()

            scope = fluid.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(program, startup_program):
                    (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                    x.persistable = True
                    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                    sgd_optimizer.minimize(loss1)
                    exe.run(startup_program)
                    weight1 = np.array(
                        scope.find_var(w_param_attrs.name).get_tensor())
                    x_np = np.random.random(size=(10, 2)).astype('float32')
                    label_np = np.random.randint(
                        1, size=(10, 1)).astype('int64')
                    compiled_prog = fluid.CompiledProgram(
                        program).with_data_parallel(
                            loss_name=loss1.name, places=fluid.CPUPlace())
                    for i in range(10):
                        res = exe.run(compiled_prog,
                                      feed={'x': x_np,
                                            'label': label_np},
                                      fetch_list=[loss1.name],
                                      use_prune=True,
                                      use_program_cache=True)
                        if i == 0:
                            self.assertEqual(exe.prune_called_times, 1)
                        else:
                            self.assertEqual(exe.prune_called_times, 1)

    def test_prune_with_multi_optimizers(self):
        '''
        If there are multiple optimizers in the program, we can run specific one by 
        pass the return of optimize.minimize() to fetch_list.
        '''
        exe = fluid.Executor(fluid.CPUPlace())

        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        # do not use_prune
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                x.persistable = True
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                train1, _ = sgd_optimizer.minimize(loss1)
                cloned_program = program.clone()

                train2, _ = sgd_optimizer.minimize(loss2)
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')

                res = exe.run(program,
                              feed={'x': x_np,
                                    'label': label_np},
                              fetch_list=[loss1.name, train1],
                              use_prune=False)

                weight_without_prune = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())

        scope = fluid.Scope()
        # use_prune
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            res = exe.run(program,
                          feed={'x': x_np,
                                'label': label_np},
                          fetch_list=[loss1.name, train1],
                          use_prune=True)
            weight_with_prune = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        # expected
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            exe.run(cloned_program,
                    feed={'x': x_np,
                          'label': label_np},
                    fetch_list=[loss1.name],
                    use_prune=False)
            weight_expected = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        self.assertTrue(np.array_equal(weight_with_prune, weight_expected))
        self.assertFalse(np.array_equal(weight_without_prune, weight_expected))

    def test_prune_with_multi_devices(self):
        '''
        When training model with multi_devices, the pruned CompiledProgram should share same local scopes.
        This test the correctness.
        '''
        exe = fluid.Executor(fluid.CPUPlace())

        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()

        scope = fluid.Scope()
        os.environ['CPU_NUM'] = str(2)
        # do not use_prune
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                x.persistable = True
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                train1, _ = sgd_optimizer.minimize(loss1)
                cloned_program = program.clone()

                train2, _ = sgd_optimizer.minimize(loss2)
                exe.run(startup_program)
                weight1 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                compiled_prog = fluid.CompiledProgram(
                    program).with_data_parallel(
                        loss_name=loss1.name, places=[fluid.CPUPlace()] * 1)
                for i in range(10):
                    if i % 2:
                        res = exe.run(compiled_prog,
                                      feed={'x': x_np,
                                            'label': label_np},
                                      fetch_list=[loss1.name, train1],
                                      use_prune=True)
                    else:
                        res = exe.run(compiled_prog,
                                      feed={'x': x_np,
                                            'label': label_np},
                                      fetch_list=[loss2.name, train2],
                                      use_prune=True)

                weight1 = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())

        # expected
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            weight2 = np.array(scope.find_var(w_param_attrs.name).get_tensor())
            for i in range(10):
                exe.run(cloned_program,
                        feed={'x': x_np,
                              'label': label_np},
                        fetch_list=[loss1.name],
                        use_prune=False)
            weight2 = np.array(scope.find_var(w_param_attrs.name).get_tensor())
        self.assertTrue(np.array_equal(weight1, weight2))


if __name__ == '__main__':
    unittest.main()
