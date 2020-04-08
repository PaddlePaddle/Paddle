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

from __future__ import print_function

import unittest
import paddle.fluid as fluid
from paddle.fluid import core


class TestCompiledProgram(unittest.TestCase):
    def test_program_or_graph_error(self):
        self.assertRaises(TypeError, fluid.CompiledProgram, "program")

    def build_simple_model(self):
        img = fluid.layers.data(
            name='image', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        prediction = fluid.layers.fc(input=img, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)

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

    def test_share_vars_from_error(self):
        with fluid.program_guard(fluid.Program()):
            self.build_simple_model()
            source_program, _, _ = self.compile_program()
            # compile program
            program = fluid.default_main_program()
            compiled_program = fluid.CompiledProgram(
                program).with_data_parallel(share_vars_from=source_program)
            scope = fluid.global_scope()
            place = fluid.CPUPlace()
            with self.assertRaises(ValueError):
                compiled_program._compile(scope, place)


if __name__ == '__main__':
    unittest.main()
