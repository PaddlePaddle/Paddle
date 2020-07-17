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

from __future__ import print_function

import unittest
import paddle.fluid as fluid
from paddle.fluid import core


class TestSaveLoadAPIError(unittest.TestCase):
    def test_get_valid_program_error(self):
        # case 1: CompiledProgram no program
        graph = core.Graph(core.ProgramDesc())
        compiled_program = fluid.CompiledProgram(graph)
        with self.assertRaises(TypeError):
            fluid.io._get_valid_program(compiled_program)

        # case 2: main_program type error
        with self.assertRaises(TypeError):
            fluid.io._get_valid_program("program")

    def test_load_vars_error(self):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        # case 1: main_program type error when vars None
        with self.assertRaises(TypeError):
            fluid.io.load_vars(
                executor=exe, dirname="./fake_dir", main_program="program")

        # case 2: main_program type error when vars not None
        with self.assertRaises(TypeError):
            fluid.io.load_vars(
                executor=exe,
                dirname="./fake_dir",
                main_program="program",
                vars="vars")


if __name__ == '__main__':
    unittest.main()
