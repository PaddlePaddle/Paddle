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

import sys
import unittest

sys.path.append("../../legacy_test")

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()


class TestCompiledProgramError(unittest.TestCase):
    def test_program_or_graph_error(self):
        self.assertRaises(TypeError, base.CompiledProgram, "program")

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
        with base.program_guard(base.Program()):
            # build model
            self.build_simple_model()
            # compile program
            program = base.default_main_program()
            compiled_program = base.CompiledProgram(program)
            scope = base.global_scope()
            place = base.CPUPlace()
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
            new_place = base.CUDAPlace(0)
            with self.assertRaises(ValueError):
                compiled_program._compile(scope, new_place)


if __name__ == '__main__':
    unittest.main()
