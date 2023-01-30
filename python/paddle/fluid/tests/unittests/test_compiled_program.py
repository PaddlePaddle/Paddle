#   copyright (c) 2020 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

<<<<<<< HEAD
import unittest

import numpy as np
from simple_nets import simple_fc_net
from test_imperative_base import new_program_scope

import paddle
import paddle.fluid as fluid
from paddle.fluid import core


class TestCompiledProgram(unittest.TestCase):
    def setUp(self):
        self.seed = 100
        self.img = np.random.random(size=(16, 784)).astype('float32')
        self.label = np.random.randint(
            low=0, high=10, size=[16, 1], dtype=np.int64
        )
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from test_imperative_base import new_program_scope
from simple_nets import simple_fc_net


class TestCompiledProgram(unittest.TestCase):

    def setUp(self):
        self.seed = 100
        self.img = np.random.random(size=(16, 784)).astype('float32')
        self.label = np.random.randint(low=0,
                                       high=10,
                                       size=[16, 1],
                                       dtype=np.int64)
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            exe = fluid.Executor(place)

            loss = simple_fc_net()
            exe.run(fluid.default_startup_program())

<<<<<<< HEAD
            (loss_data,) = exe.run(
                fluid.default_main_program(),
                feed={"image": self.img, "label": self.label},
                fetch_list=[loss.name],
            )
=======
            loss_data, = exe.run(fluid.default_main_program(),
                                 feed={
                                     "image": self.img,
                                     "label": self.label
                                 },
                                 fetch_list=[loss.name])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.loss = loss_data[0]

    def test_compiled_program_base(self):
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
<<<<<<< HEAD
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
=======
            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            exe = fluid.Executor(place)

            loss = simple_fc_net()
            exe.run(fluid.default_startup_program())
            compiled_prog = fluid.CompiledProgram(fluid.default_main_program())

<<<<<<< HEAD
            (loss_data,) = exe.run(
                compiled_prog,
                feed={"image": self.img, "label": self.label},
                fetch_list=[loss.name],
            )
=======
            loss_data, = exe.run(compiled_prog,
                                 feed={
                                     "image": self.img,
                                     "label": self.label
                                 },
                                 fetch_list=[loss.name])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_array_equal(loss_data[0], self.loss)

    def test_compiled_program_with_data_parallel(self):
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
<<<<<<< HEAD
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
=======
            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            exe = fluid.Executor(place)

            loss = simple_fc_net()
            exe.run(fluid.default_startup_program())
            compiled_prog = fluid.CompiledProgram(
<<<<<<< HEAD
                fluid.default_main_program()
            ).with_data_parallel(loss_name=loss.name, places=[place])

            (loss_data,) = exe.run(
                compiled_prog,
                feed={"image": self.img, "label": self.label},
                fetch_list=[loss.name],
            )
=======
                fluid.default_main_program()).with_data_parallel(
                    loss_name=loss.name, places=[place])

            loss_data, = exe.run(compiled_prog,
                                 feed={
                                     "image": self.img,
                                     "label": self.label
                                 },
                                 fetch_list=[loss.name])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_array_equal(loss_data[0], self.loss)


class TestCompiledProgramError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_program_or_graph_error(self):
        self.assertRaises(TypeError, fluid.CompiledProgram, "program")

    def build_simple_model(self):
<<<<<<< HEAD
        img = paddle.static.data(
            name='image', shape=[-1, 1, 28, 28], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        prediction = paddle.static.nn.fc(x=img, size=10, activation='softmax')
        loss = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
=======
        img = fluid.layers.data(name='image',
                                shape=[1, 28, 28],
                                dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        prediction = fluid.layers.fc(input=img, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        avg_loss = paddle.mean(loss)

    def compile_program_not_compiled(self):
        with fluid.program_guard(fluid.Program()):
            # build model
            self.build_simple_model()
            # compile program
            program = fluid.default_main_program()
            compiled_program = fluid.CompiledProgram(
<<<<<<< HEAD
                program
            ).with_data_parallel()
=======
                program).with_data_parallel()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return compiled_program

    def compile_program(self):
        with fluid.program_guard(fluid.Program()):
            # build model
            self.build_simple_model()
            # compile program
            program = fluid.default_main_program()
            compiled_program = fluid.CompiledProgram(program)
            scope = fluid.global_scope()
            place = fluid.CPUPlace()
            compiled_program._compile(scope, place)
            return compiled_program, scope, place

    def test_compile_scope_error(self):
        compiled_program, _, place = self.compile_program()
        new_scope = core.Scope()
        with self.assertRaises(ValueError):
            compiled_program._compile(new_scope, place)

    def test_compile_place_error(self):
        # need create different place
        if core.is_compiled_with_cuda():
            compiled_program, scope, _ = self.compile_program()
            new_place = fluid.CUDAPlace(0)
            with self.assertRaises(ValueError):
                compiled_program._compile(scope, new_place)

    def test_share_vars_from_error_no_parallel(self):
        with fluid.program_guard(fluid.Program()):
            source_program, _, _ = self.compile_program()
            self.build_simple_model()
            # compile program
            program = fluid.default_main_program()
            compiled_program = fluid.CompiledProgram(
<<<<<<< HEAD
                program
            ).with_data_parallel(share_vars_from=source_program)
=======
                program).with_data_parallel(share_vars_from=source_program)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            scope = fluid.global_scope()
            place = fluid.CPUPlace()
            with self.assertRaises(ValueError):
                compiled_program._compile(scope, place)

    def test_share_vars_from_error_no_executor(self):
        with fluid.program_guard(fluid.Program()):
            source_program = self.compile_program_not_compiled()
            self.build_simple_model()
            # compile program
            program = fluid.default_main_program()
            compiled_program = fluid.CompiledProgram(
<<<<<<< HEAD
                program
            ).with_data_parallel(share_vars_from=source_program)
=======
                program).with_data_parallel(share_vars_from=source_program)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            scope = fluid.global_scope()
            place = fluid.CPUPlace()
            with self.assertRaises(ValueError):
                compiled_program._compile(scope, place)


if __name__ == '__main__':
    unittest.main()
