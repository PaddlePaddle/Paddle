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
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestRunTimeException(OpTest):
    def test_run_time_exception(self):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(name="x", shape=[1], dtype="float32")
            fluid.layers.fc(input=x, size=3)

        def _run_program():
            x = np.random.random(size=(10)).astype('float32')
            exe.run(train_program, feed={"x": x})

        self.assertRaises(core.EnforceNotMet, _run_program)


class TestCompileTimeException(OpTest):
    def test_compile_time_exception(self):
        self.assertRaises(core.EnforceNotMet, self.build_model)

    def build_model(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            x = fluid.layers.data(
                name="x", shape=[1], dtype="float32", append_batch_size=False)
            fluid.layers.fc(input=x, size=100)


if __name__ == '__main__':
    unittest.main()
