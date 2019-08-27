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

import contextlib
import numpy
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.compiler as compiler


class TestPersistableVarsInitCheck(unittest.TestCase):
    def forward_net_func(self):
        x = fluid.layers.data(name='X', shape=[13], dtype='float32')
        y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
        y_ = fluid.layers.fc(input=x, size=1, act=None)
        loss = fluid.layers.square_error_cost(input=y_, label=y)
        avg_loss = fluid.layers.mean(loss)
        return avg_loss

    def test_run_startup_after_minimize(self):
        with self.program_scope_guard():
            place = core.CPUPlace()
            exe = fluid.Executor(place)

            avg_loss = self.forward_net_func()
            fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)
            exe.run(fluid.default_startup_program())

            input_x = numpy.random.random(size=(10, 13)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')
            exe.run(fluid.default_main_program(),
                    feed={'X': input_x,
                          'Y': input_y},
                    fetch_list=[avg_loss.name])

    def test_run_startup_before_minimize(self):
        with self.program_scope_guard():
            place = core.CPUPlace()
            exe = fluid.Executor(place)

            avg_loss = self.forward_net_func()
            exe.run(fluid.default_startup_program())
            fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

            input_x = numpy.random.random(size=(10, 13)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')
            with self.assertRaisesRegexp(RuntimeError,
                "There are persistable variables in the current program that are not initialized. \n"\
                "Please confirm that you have run startup_program and run it after fluid.optimizer.minimize()."):
                exe.run(fluid.default_main_program(),
                        feed={'X': input_x,
                              'Y': input_y},
                        fetch_list=[avg_loss.name])

    def test_run_startup_after_minimize_for_compiled_program(self):
        with self.program_scope_guard():
            place = core.CPUPlace()
            exe = fluid.Executor(place)

            avg_loss = self.forward_net_func()
            fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)
            exe.run(fluid.default_startup_program())

            input_x = numpy.random.random(size=(10, 13)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')
            compiled_prog = compiler.CompiledProgram(fluid.default_main_program(
            )).with_data_parallel(loss_name=avg_loss.name)
            exe.run(compiled_prog,
                    feed={'X': input_x,
                          'Y': input_y},
                    fetch_list=[avg_loss.name])

    def test_run_startup_before_minimize_for_compiled_program(self):
        with self.program_scope_guard():
            place = core.CPUPlace()
            exe = fluid.Executor(place)

            avg_loss = self.forward_net_func()
            exe.run(fluid.default_startup_program())
            fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

            input_x = numpy.random.random(size=(10, 13)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')
            compiled_prog = compiler.CompiledProgram(fluid.default_main_program(
            )).with_data_parallel(loss_name=avg_loss.name)
            with self.assertRaisesRegexp(RuntimeError,
                "There are persistable variables in the current program that are not initialized. \n"\
                "Please confirm that you have run startup_program and run it after fluid.optimizer.minimize()."):
                exe.run(compiled_prog,
                        feed={'X': input_x,
                              'Y': input_y},
                        fetch_list=[avg_loss.name])

    def test_add_new_elements_flag_state(self):
        with self.program_scope_guard():
            place = core.CPUPlace()
            exe = fluid.Executor(place)
            main_program = fluid.default_main_program()

            avg_loss = self.forward_net_func()
            self.assertTrue(main_program._add_new_elements)

            input_x = numpy.random.random(size=(10, 13)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')

            exe.run(fluid.default_startup_program())
            exe.run(fluid.default_main_program(),
                    feed={'X': input_x,
                          'Y': input_y},
                    fetch_list=[avg_loss.name])
            self.assertFalse(main_program._add_new_elements)

            fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)
            self.assertTrue(main_program._add_new_elements)

            exe.run(fluid.default_startup_program())
            exe.run(fluid.default_main_program(),
                    feed={'X': input_x,
                          'Y': input_y},
                    fetch_list=[avg_loss.name])
            self.assertFalse(main_program._add_new_elements)

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
