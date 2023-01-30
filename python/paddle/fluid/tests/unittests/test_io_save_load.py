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

<<<<<<< HEAD
import os
import tempfile
import unittest

import paddle
import paddle.fluid as fluid
from paddle.fluid import core


class TestSaveLoadAPIError(unittest.TestCase):
=======
from __future__ import print_function

import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph
import tempfile
import os


class TestSaveLoadAPIError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = os.path.join(self.temp_dir.name, "fake_dir")

    def tearDown(self):
        self.temp_dir.cleanup()

<<<<<<< HEAD
    def test_get_valid_program_error(self):
=======
    def func_test_get_valid_program_error(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # case 1: CompiledProgram no program
        graph = core.Graph(core.ProgramDesc())
        compiled_program = fluid.CompiledProgram(graph)
        with self.assertRaises(TypeError):
            fluid.io._get_valid_program(compiled_program)

        # case 2: main_program type error
        with self.assertRaises(TypeError):
            fluid.io._get_valid_program("program")

<<<<<<< HEAD
    def test_load_vars_error(self):
=======
    def test_get_valid_program_error(self):
        with _test_eager_guard():
            self.func_test_get_valid_program_error()
        self.func_test_get_valid_program_error()

    def func_test_load_vars_error(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        # case 1: main_program type error when vars None
        with self.assertRaises(TypeError):
<<<<<<< HEAD
            fluid.io.load_vars(
                executor=exe, dirname=self.save_dir, main_program="program"
            )

        # case 2: main_program type error when vars not None
        with self.assertRaises(TypeError):
            fluid.io.load_vars(
                executor=exe,
                dirname=self.save_dir,
                main_program="program",
                vars="vars",
            )


class TestSaveInferenceModelAPIError(unittest.TestCase):
=======
            fluid.io.load_vars(executor=exe,
                               dirname=self.save_dir,
                               main_program="program")

        # case 2: main_program type error when vars not None
        with self.assertRaises(TypeError):
            fluid.io.load_vars(executor=exe,
                               dirname=self.save_dir,
                               main_program="program",
                               vars="vars")

    def test_load_vars_error(self):
        with _test_eager_guard():
            self.func_test_load_vars_error()
        self.func_test_load_vars_error()


class TestSaveInferenceModelAPIError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

<<<<<<< HEAD
    def test_useless_feeded_var_names(self):
=======
    def func_test_useless_feeded_var_names(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        start_prog = fluid.Program()
        main_prog = fluid.Program()
        with fluid.program_guard(main_prog, start_prog):
            x = fluid.data(name='x', shape=[10, 16], dtype='float32')
            y = fluid.data(name='y', shape=[10, 16], dtype='float32')
<<<<<<< HEAD
            z = paddle.static.nn.fc(x, 4)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(start_prog)
        with self.assertRaisesRegex(
            ValueError, "not involved in the target_vars calculation"
        ):
            fluid.io.save_inference_model(
                dirname=os.path.join(self.temp_dir.name, 'model'),
                feeded_var_names=['x', 'y'],
                target_vars=[z],
                executor=exe,
                main_program=main_prog,
            )


class TestWhenTrainWithNoGrad(unittest.TestCase):
=======
            z = fluid.layers.fc(x, 4)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(start_prog)
        with self.assertRaisesRegexp(
                ValueError, "not involved in the target_vars calculation"):
            fluid.io.save_inference_model(dirname=os.path.join(
                self.temp_dir.name, 'model'),
                                          feeded_var_names=['x', 'y'],
                                          target_vars=[z],
                                          executor=exe,
                                          main_program=main_prog)

    def test_useless_feeded_var_names(self):
        with _test_eager_guard():
            self.func_test_useless_feeded_var_names()
        self.func_test_useless_feeded_var_names()


class TestWhenTrainWithNoGrad(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

<<<<<<< HEAD
    def test_when_train_with_no_grad(self):
=======
    def func_test_when_train_with_no_grad(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.disable_static()
        net = paddle.nn.Linear(1024, 1)
        net = paddle.jit.to_static(net)
        x = paddle.rand([1024], 'float32')
        net(x)
        save_path = os.path.join(self.temp_dir.name, 'train_with_no_grad')

        paddle.jit.save(net, save_path)
        net = paddle.jit.load(save_path)
        net.train()

        with paddle.no_grad():
            x = paddle.rand([1024], 'float32')
            net(x)

<<<<<<< HEAD
=======
    def test_when_train_with_no_grad(self):
        with _test_eager_guard():
            self.func_test_when_train_with_no_grad()
        self.func_test_when_train_with_no_grad()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
