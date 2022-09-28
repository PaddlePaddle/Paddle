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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestRunTimeException(unittest.TestCase):

    def test_run_time_exception(self):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            fluid.layers.one_hot(input=label, depth=100)

        def _run_program():
            x = np.random.random(size=(10)).astype('int64')
            exe.run(train_program, feed={"label": x})

        self.assertRaises(ValueError, _run_program)


class TestCompileTimeException(unittest.TestCase):

    def test_compile_time_exception(self):
        self.assertRaises(ValueError, self.build_model)

    def build_model(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            label = fluid.layers.data(name="label",
                                      shape=[1],
                                      dtype="int64",
                                      append_batch_size=False)
            fluid.layers.one_hot(input=label, depth=100)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
