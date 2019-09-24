# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from paddle.fluid.executor import Executor


class TestLoadVarsShapeCheck(unittest.TestCase):
    def test_shape_check_save(self):
        program_1 = fluid.Program()
        startup_program_1 = fluid.Program()

        with fluid.program_guard(program_1, startup_program_1):
            input = fluid.layers.data(name="x", shape=[-1, 10], dtype='float32')
            out = fluid.layers.fc(input, 20)
        place = fluid.CPUPlace()
        exe = Executor(place)
        exe.run(startup_program_1)

        fluid.io.save_params(exe, "./model_temp", main_program=program_1)

    def test_shape_check_load(self):
        program_1 = fluid.Program()
        startup_program_1 = fluid.Program()

        with fluid.program_guard(program_1, startup_program_1):
            input = fluid.layers.data(name="x", shape=[-1, 10], dtype='float32')
            out = fluid.layers.fc(input, 10)
        place = fluid.CPUPlace()
        exe = Executor(place)
        exe.run(startup_program_1)

        #fluid.io.save_params(exe, "./model_temp", main_program=program_1)
        try:
            fluid.io.load_params(exe, "./model_temp", main_program=program_1)
        except RuntimeError, e:
            self.assertTrue(e.message.startswith("Shape not matching"))
        except:
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
