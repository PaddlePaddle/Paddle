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

import unittest

import paddle
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
        loss = paddle.mean(x=loss)
        return x, y, label, loss

    def test_prune_with_input(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "reduce_mean"
        ])
        pruned_program = program._prune_with_input(
            feeded_var_names=[y.name, label.name], targets=[loss])
        self.assertEqual(len(pruned_program.global_block().ops), 2)
        self.assertEqual([op.type for op in pruned_program.global_block().ops],
                         ["cross_entropy2", "reduce_mean"])

    def test_prune(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "reduce_mean"
        ])
        pruned_program = program._prune(targets=[loss])
        self.assertEqual(len(pruned_program.global_block().ops), 5)
        self.assertEqual([op.type for op in pruned_program.global_block().ops],
                         [
                             "mul", "elementwise_add", "softmax",
                             "cross_entropy2", "reduce_mean"
                         ])

    def test_prune_target_not_list(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "reduce_mean"
        ])
        pruned_program = program._prune(targets=loss)
        self.assertEqual(len(pruned_program.global_block().ops), 5)
        self.assertEqual([op.type for op in pruned_program.global_block().ops],
                         [
                             "mul", "elementwise_add", "softmax",
                             "cross_entropy2", "reduce_mean"
                         ])

    def test_prune_target_none(self):
        program = framework.Program()
        startup_program = framework.Program()
        block = program.global_block()
        with fluid.program_guard(program, startup_program):
            (x, y, label, loss) = self.net()
        self.assertEqual(len(block.ops), 5)
        self.assertEqual([op.type for op in block.ops], [
            "mul", "elementwise_add", "softmax", "cross_entropy2", "reduce_mean"
        ])
        try:
            pruned_program = program._prune(targets=None)
        except ValueError as e:
            self.assertIn(
                "All targets of Program._prune_with_input() can only be Variable or Operator",
                cpt.get_exception_message(e))


def mock(self, program, feed, fetch, optimize_ops):
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
        loss1 = paddle.mean(x=loss1)
        loss2 = fluid.layers.cross_entropy(input=y, label=label)
        loss2 = paddle.mean(x=loss2)
        loss1.persistable = True
        loss2.persistable = True
        return x, y, label, loss1, loss2, w_param_attrs

    def net2(self):
        x1 = fluid.layers.data(name='x1', shape=[2], dtype='float32')
        x2 = fluid.layers.data(name='x2', shape=[2], dtype='float32')
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        w1_param_attrs = fluid.ParamAttr(
            name="fc_weight1",
            learning_rate=0.5,
            initializer=fluid.initializer.Constant(1.0),
            trainable=True)
        w2_param_attrs = fluid.ParamAttr(
            name="fc_weight2",
            learning_rate=0.5,
            initializer=fluid.initializer.Constant(1.0),
            trainable=True)
        y1 = fluid.layers.fc(input=[x1],
                             size=2,
                             act="softmax",
                             param_attr=w1_param_attrs)
        y2 = fluid.layers.fc(input=[x2],
                             size=2,
                             act="softmax",
                             param_attr=w2_param_attrs)
        loss1 = fluid.layers.cross_entropy(input=y1, label=label)
        loss1 = paddle.mean(x=loss1)
        loss2 = fluid.layers.cross_entropy(input=y2, label=label)
        loss2 = paddle.mean(x=loss2)
        return x1, x2, y1, y2, label, loss1, loss2, w1_param_attrs, w2_param_attrs

    def test_not_prune(self):
        """
        If use_prune = False, the targets which is not fetched will be calculated.
        """
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  'x': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=False)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNotNone(scope.find_var(loss2.name))

    def test_prune_fetches_without_optimizer(self):
        """
        Prune operators and variables which are not needed to generate 'fetches'.
        """
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  'x': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))  #loss2 is pruned
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                np.testing.assert_array_equal(weight_init,
                                              weight)  # weight not changed

    def test_prune_fetches_with_optimizer(self):
        """
        Prune operators and operators which are not needed to generate 'fetches'.
        In train mode, the operators and operators in backward and optimization should be kept.
        """
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  'x': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))  #loss2 is pruned
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                self.assertFalse(np.array_equal(weight_init,
                                                weight))  # weight changed

    def test_prune_compiled_program(self):
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                compiled_prog = fluid.CompiledProgram(
                    program).with_data_parallel(loss_name=loss1.name,
                                                places=fluid.CPUPlace())
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(compiled_prog,
                              feed={
                                  'x': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                self.assertFalse(np.array_equal(weight_init,
                                                weight))  # weight changed

    def test_prune_feed_without_optimizer(self):
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  y.name: x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                np.testing.assert_array_equal(weight_init,
                                              weight)  # weight unchanged

    def test_prune_feed_with_optimizer(self):
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                sgd_optimizer.minimize(loss1)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                self.assertRaises(Exception,
                                  exe.run,
                                  program,
                                  feed={
                                      y.name: x_np,
                                      'label': label_np
                                  },
                                  fetch_list=[loss1.name],
                                  use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))

    def test_prune_with_cache_program(self):
        '''
        When use_prune=True, Executor should cache the pruned program.
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
            scope = fluid.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(program, startup_program):
                    (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                    sgd_optimizer.minimize(loss1)
                    exe.run(startup_program)
                    x_np = np.random.random(size=(10, 2)).astype('float32')
                    label_np = np.random.randint(1,
                                                 size=(10, 1)).astype('int64')
                    for i in range(10):
                        res = exe.run(program,
                                      feed={
                                          'x': x_np,
                                          'label': label_np
                                      },
                                      fetch_list=[loss1.name],
                                      use_prune=True)
                        if i == 0:
                            self.assertEqual(exe.prune_called_times, 1)
                        else:
                            self.assertEqual(exe.prune_called_times, 1)

    def test_prune_with_cache_program2(self):
        '''
        When use_prune=True, Executor should cache the pruned program.
        If the only difference in fetch_list is  optimize_ops during multiple runs,
        the cache_keys should be different and get different pruned program.
        '''
        with _mock_guard(mock):
            exe = fluid.Executor(fluid.CPUPlace())
            exe.prune_called_times = 0
            program = framework.Program()
            startup_program = framework.Program()
            scope = fluid.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(program, startup_program):
                    (x1, x2, y1, y2, label, loss1, loss2, w1_param_attrs,
                     w2_param_attrs) = self.net2()
                    adam_optimizer1 = fluid.optimizer.AdamOptimizer(
                        learning_rate=0.5)
                    train1 = adam_optimizer1.minimize(loss1)
                    adam_optimizer2 = fluid.optimizer.AdamOptimizer(
                        learning_rate=0.5)
                    train2 = adam_optimizer2.minimize(loss2)
                    exe.run(startup_program)
                    x_np = np.random.random(size=(10, 2)).astype('float32')
                    label_np = np.random.randint(1,
                                                 size=(10, 1)).astype('int64')

                    for i in range(10):
                        if i % 2:
                            res = exe.run(program,
                                          feed={
                                              'x1': x_np,
                                              'x2': x_np,
                                              'label': label_np
                                          },
                                          fetch_list=[loss1, loss2, train1],
                                          use_prune=True)
                        else:
                            res = exe.run(program,
                                          feed={
                                              'x1': x_np,
                                              'x2': x_np,
                                              'label': label_np
                                          },
                                          fetch_list=[loss1, loss2, train2],
                                          use_prune=True)
                        if i == 0:
                            self.assertEqual(exe.prune_called_times, 1)
                        elif i == 1:
                            self.assertEqual(exe.prune_called_times, 2)
                        else:
                            self.assertEqual(exe.prune_called_times, 2)

    def test_prune_with_cache_compiled_program(self):
        '''
        When use_prune=True, Executor should cache the pruned program.
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
            scope = fluid.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(program, startup_program):
                    (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                    sgd_optimizer.minimize(loss1)
                    exe.run(startup_program)
                    x_np = np.random.random(size=(10, 2)).astype('float32')
                    label_np = np.random.randint(1,
                                                 size=(10, 1)).astype('int64')
                    compiled_prog = fluid.CompiledProgram(
                        program).with_data_parallel(loss_name=loss1.name,
                                                    places=fluid.CPUPlace())
                    for i in range(10):
                        res = exe.run(compiled_prog,
                                      feed={
                                          'x': x_np,
                                          'label': label_np
                                      },
                                      fetch_list=[loss1.name],
                                      use_prune=True)
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
        scope = fluid.Scope()
        # do not use_prune
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                train1, _ = sgd_optimizer.minimize(loss1)
                cloned_program = program.clone()
                train2, _ = sgd_optimizer.minimize(loss2)
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  'x': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=False)
                weight_without_prune = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())

        scope = fluid.Scope()
        # use_prune
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            res = exe.run(program,
                          feed={
                              'x': x_np,
                              'label': label_np
                          },
                          fetch_list=[loss1.name, train1],
                          use_prune=True)
            weight_with_prune = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        # expected
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            exe.run(cloned_program,
                    feed={
                        'x': x_np,
                        'label': label_np
                    },
                    fetch_list=[loss1.name],
                    use_prune=False)
            weight_expected = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        np.testing.assert_array_equal(weight_with_prune, weight_expected)
        self.assertFalse(np.array_equal(weight_without_prune, weight_expected))

    def test_prune_with_multi_devices(self):
        '''
        When training model with multi_devices, the pruned CompiledProgram should share same local scopes.
        This test the correctness.
        '''
        exe = fluid.Executor(fluid.CPUPlace())
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        os.environ['CPU_NUM'] = str(2)
        # do not use_prune
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x1, x2, y1, y2, label, loss1, loss2, w1_param_attrs,
                 w2_param_attrs) = self.net2()
                adam_optimizer1 = fluid.optimizer.AdamOptimizer(
                    learning_rate=0.5)
                train1 = adam_optimizer1.minimize(loss1)
                cloned_program = program.clone()
                adam_optimizer2 = fluid.optimizer.AdamOptimizer(
                    learning_rate=0.5)
                train2 = adam_optimizer2.minimize(loss2)
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                compiled_prog1 = fluid.CompiledProgram(
                    program).with_data_parallel(loss_name=loss1.name,
                                                places=[fluid.CPUPlace()] * 2)
                compiled_prog2 = fluid.CompiledProgram(
                    program).with_data_parallel(loss_name=loss2.name,
                                                places=[fluid.CPUPlace()] * 2)
                for i in range(10):
                    if i % 2 == 1:
                        res = exe.run(compiled_prog1,
                                      feed=[{
                                          'x1': x_np[0:5, :],
                                          'label': label_np[0:5, :]
                                      }, {
                                          'x1': x_np[5:, :],
                                          'label': label_np[5:, :]
                                      }],
                                      fetch_list=[loss1.name, train1],
                                      use_prune=True)
                    else:
                        res = exe.run(compiled_prog2,
                                      feed={
                                          'x2': x_np,
                                          'label': label_np
                                      },
                                      fetch_list=[loss2.name, train2],
                                      use_prune=True)
                weight1 = np.array(
                    scope.find_var(w1_param_attrs.name).get_tensor())
        # expected
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            for i in range(10):
                if i % 2 == 1:
                    exe.run(cloned_program,
                            feed={
                                'x1': x_np,
                                'x2': x_np,
                                'label': label_np
                            },
                            fetch_list=[loss1.name],
                            use_prune=False)
            weight2 = np.array(scope.find_var(w1_param_attrs.name).get_tensor())
        np.testing.assert_allclose(weight1, weight2, rtol=1e-05)

    def test_prune_program_with_tupe_in_fetch_list(self):
        '''
        If there are multiple optimizers in the program, we can run specific one by
        pass the return of optimize.minimize() to fetch_list.
        '''
        exe = fluid.Executor(fluid.CPUPlace())
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        # do not use_prune
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                train1 = sgd_optimizer.minimize(loss1)
                cloned_program = program.clone()

                train2 = sgd_optimizer.minimize(loss2)
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')

                res = exe.run(program,
                              feed={
                                  'x': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=False)

                weight_without_prune = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())

        scope = fluid.Scope()
        # use_prune
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            res = exe.run(program,
                          feed={
                              'x': x_np,
                              'label': label_np
                          },
                          fetch_list=[loss1.name, train1],
                          use_prune=True)
            weight_with_prune = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        # expected
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            exe.run(cloned_program,
                    feed={
                        'x': x_np,
                        'label': label_np
                    },
                    fetch_list=[loss1.name],
                    use_prune=False)
            weight_expected = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        np.testing.assert_array_equal(weight_with_prune, weight_expected)
        self.assertFalse(np.array_equal(weight_without_prune, weight_expected))

    def test_prune_program_partial_parameter_updated(self):
        """
        When running startup program, all parameters declared will be initialized.
        When running main program with prune=True, the pruned parameters will exist in scope and stay unchanged.
        """
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x1, x2, y1, y2, label, loss1, loss2, w1_param_attrs,
                 w2_param_attrs) = self.net2()
                loss1.persistable = True
                loss2.persistable = True
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                train1 = sgd_optimizer.minimize(loss1)
                sgd_optimizer1 = fluid.optimizer.SGD(learning_rate=0.5)
                train2 = sgd_optimizer1.minimize(loss2)
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight1_init = np.array(
                    scope.find_var(w1_param_attrs.name).get_tensor())
                weight2_init = np.array(
                    scope.find_var(w2_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')

                res = exe.run(program,
                              feed={
                                  'x1': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name, train1],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(w1_param_attrs.name))
                self.assertIsNotNone(scope.find_var(w2_param_attrs.name))
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))
                weight1 = np.array(
                    scope.find_var(w1_param_attrs.name).get_tensor())
                weight2 = np.array(
                    scope.find_var(w2_param_attrs.name).get_tensor())
                self.assertFalse(np.array_equal(weight1_init,
                                                weight1))  # weight changed
                np.testing.assert_array_equal(weight2_init,
                                              weight2)  # weight2 unchanged

    def test_prune_override_use_prune(self):
        '''
        If optimize_ops in provided in the fetch_list, the argument use_prune is always override to True.
        '''
        exe = fluid.Executor(fluid.CPUPlace())
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        # do not use_prune
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.5)
                train1, _ = sgd_optimizer.minimize(loss1)
                cloned_program = program.clone()
                train2, _ = sgd_optimizer.minimize(loss2)
                exe.run(startup_program)
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  'x': x_np,
                                  'label': label_np
                              },
                              fetch_list=[loss1.name],
                              use_prune=False)

                weight_without_prune = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())

        scope = fluid.Scope()
        # use_prune
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            res = exe.run(program,
                          feed={
                              'x': x_np,
                              'label': label_np
                          },
                          fetch_list=[loss1.name, train1])
            weight_with_prune = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        # expected
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
            exe.run(cloned_program,
                    feed={
                        'x': x_np,
                        'label': label_np
                    },
                    fetch_list=[loss1.name],
                    use_prune=False)
            weight_expected = np.array(
                scope.find_var(w_param_attrs.name).get_tensor())

        np.testing.assert_array_equal(weight_with_prune, weight_expected)
        self.assertFalse(np.array_equal(weight_without_prune, weight_expected))

    def test_prune_feed_var_in_fetchlist_1(self):
        # the variable to be fed is not leaf
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  y.name: x_np,
                                  'label': label_np
                              },
                              fetch_list=[y.name, loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))
                self.assertIsNone(scope.find_var(x.name))
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                np.testing.assert_array_equal(weight_init,
                                              weight)  # weight unchanged

    def test_prune_feed_var_in_fetchlist_2(self):
        # the variable to be fed is leaf
        program = framework.Program()
        startup_program = framework.Program()
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program, startup_program):
                (x, y, label, loss1, loss2, w_param_attrs) = self.net1()
                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                weight_init = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                x_np = np.random.random(size=(10, 2)).astype('float32')
                label_np = np.random.randint(1, size=(10, 1)).astype('int64')
                res = exe.run(program,
                              feed={
                                  x.name: x_np,
                                  'label': label_np
                              },
                              fetch_list=[x.name, loss1.name],
                              use_prune=True)
                self.assertIsNotNone(scope.find_var(loss1.name))
                self.assertIsNone(scope.find_var(loss2.name))
                weight = np.array(
                    scope.find_var(w_param_attrs.name).get_tensor())
                np.testing.assert_array_equal(weight_init,
                                              weight)  # weight unchanged


if __name__ == '__main__':
    unittest.main()
