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
            exe = fluid.Executor(place)

            loss = simple_fc_net()
            exe.run(fluid.default_startup_program())

            (loss_data,) = exe.run(
                fluid.default_main_program(),
                feed={"image": self.img, "label": self.label},
                fetch_list=[loss.name],
            )
            self.loss = loss_data[0]

    def test_compiled_program_base(self):
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
            exe = fluid.Executor(place)

            loss = simple_fc_net()
            exe.run(fluid.default_startup_program())
            compiled_prog = fluid.CompiledProgram(fluid.default_main_program())

            (loss_data,) = exe.run(
                compiled_prog,
                feed={"image": self.img, "label": self.label},
                fetch_list=[loss.name],
            )
            np.testing.assert_array_equal(loss_data[0], self.loss)


class TestCompiledProgramError(unittest.TestCase):
    def test_program_or_graph_error(self):
        self.assertRaises(TypeError, fluid.CompiledProgram, "program")

    def build_simple_model(self):
        img = paddle.static.data(
            name='image', shape=[-1, 1, 28, 28], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        prediction = paddle.static.nn.fc(x=img, size=10, activation='softmax')
        loss = paddle.nn.functional.cross_entropy(
            input=prediction, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)

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


if __name__ == '__main__':
    unittest.main()
