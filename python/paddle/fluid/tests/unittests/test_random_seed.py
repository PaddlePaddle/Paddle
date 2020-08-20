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
"""Test cloud role maker."""

from __future__ import print_function
import os
import unittest
import paddle.fluid.generator as generator

import time  # temp for debug
import paddle.fluid as fluid
import numpy as np
import paddle
import paddle.fluid.core as core


class TestGeneratorSeed(unittest.TestCase):
    """
    Test cases for cpu generator seed.
    """

    def test_generator_uniform_random_dygraph(self):
        """Test Generator seed."""
        gen = generator.Generator()

        fluid.enable_dygraph()

        gen.manual_seed(12312321111)
        x = fluid.layers.uniform_random([10], dtype="float32", min=0.0, max=1.0)
        st1 = gen.get_state()
        x1 = fluid.layers.uniform_random(
            [10], dtype="float32", min=0.0, max=1.0)
        gen.set_state(st1)
        x2 = fluid.layers.uniform_random(
            [10], dtype="float32", min=0.0, max=1.0)
        gen.manual_seed(12312321111)
        x3 = fluid.layers.uniform_random(
            [10], dtype="float32", min=0.0, max=1.0)
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if not core.is_compiled_with_cuda():
            self.assertTrue(np.allclose(x1_np, x2_np))
            self.assertTrue(np.allclose(x_np, x3_np))

    def test_generator_uniform_random_static(self):

        fluid.disable_dygraph()

        gen = generator.Generator()
        gen.manual_seed(123123143)

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = fluid.layers.uniform_random(shape=[3, 4])
            result_2 = fluid.layers.uniform_random(shape=[3, 4])

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(train_program,
                           feed={},
                           fetch_list=[result_1, result_2])
            #gen.set_state(cur_state)
            gen.manual_seed(123123143)
            out2 = exe.run(train_program,
                           feed={},
                           fetch_list=[result_1, result_2])

            out1_res1 = np.array(out1[0])
            out1_res2 = np.array(out1[1])
            out2_res1 = np.array(out2[0])
            out2_res2 = np.array(out2[1])

            if not core.is_compiled_with_cuda():
                self.assertTrue(np.allclose(out1_res1, out2_res1))
                self.assertTrue(np.allclose(out1_res2, out2_res2))
                self.assertTrue(not np.allclose(out1_res2, out1_res1))

    def test_generator_randint_dygraph(self):
        """Test Generator seed."""
        gen = generator.Generator()

        fluid.enable_dygraph()

        gen.manual_seed(12312321111)
        x = paddle.randint(low=1)
        st1 = gen.get_state()
        x1 = paddle.randint(low=1)
        gen.set_state(st1)
        x2 = paddle.randint(low=1)
        gen.manual_seed(12312321111)
        x3 = paddle.randint(low=1)
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()
        if not core.is_compiled_with_cuda():
            self.assertTrue(np.allclose(x1_np, x2_np))
            self.assertTrue(np.allclose(x_np, x3_np))


if __name__ == "__main__":
    unittest.main()
