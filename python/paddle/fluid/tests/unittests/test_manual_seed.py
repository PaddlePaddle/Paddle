#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import manual_seed
from paddle.fluid.framework import Program, default_main_program, default_startup_program


class TestManualSeed(unittest.TestCase):
    def test_manual_seed(self):
        local_program = Program()
        local_main_prog = default_main_program()
        local_start_prog = default_startup_program()

        self.assertEqual(0, local_program.random_seed)
        self.assertEqual(0, local_main_prog.random_seed)
        self.assertEqual(0, local_start_prog.random_seed)

        manual_seed(102)
        global_program1 = Program()
        global_program2 = Program()
        global_main_prog = default_main_program()
        global_start_prog = default_startup_program()
        self.assertEqual(102, global_program1.random_seed)
        self.assertEqual(102, global_program2.random_seed)
        self.assertEqual(102, global_main_prog.random_seed)
        self.assertEqual(102, global_start_prog.random_seed)


if __name__ == '__main__':
    unittest.main()
