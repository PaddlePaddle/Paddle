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


class TestGeneratorSeed(unittest.TestCase):
    """
    Test cases for cpu generator seed.
    """

    def test_basic_generator(self):
        """Test Generator seed."""
        gen = generator.Generator()

        with fluid.dygraph.guard():
            gen.manual_seed(12312321111)
            x = fluid.layers.uniform_random(
                [10], dtype="float32", min=0.0, max=1.0)
            st1 = gen.get_state()
            x1 = fluid.layers.uniform_random(
                [10], dtype="float32", min=0.0, max=1.0)
            #gen.set_state(st1)
            x2 = fluid.layers.uniform_random(
                [10], dtype="float32", min=0.0, max=1.0)
            x_np = x.numpy()
            x1_np = x1.numpy()
            x2_np = x2.numpy()
            print("x:: {}".format(x_np))
            print("x1:: {}".format(x1_np))
            print("x2:: {}".format(x2_np))


if __name__ == "__main__":
    unittest.main()
'''
startup_program = fluid.Program()
train_program = fluid.Program()
with fluid.program_guard(train_program, startup_program):
    # example 1:
    # attr shape is a list which doesn't contain tensor Variable.
    result_1 = fluid.layers.uniform_random(shape=[3, 4])
    result_2 = fluid.layers.uniform_random(shape=[3, 4])

    # example 2:
    # attr shape is a list which contains tensor Variable.
    dim_1 = fluid.layers.fill_constant([1],"int64",3)

    dim_2 = fluid.layers.fill_constant([1],"int32",5)
    result_2 = fluid.layers.uniform_random(shape=[dim_1, dim_2])

    # example 3:
    # attr shape is a Variable, the data type must be int32 or int64
    var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
    result_3 = fluid.layers.uniform_random(var_shape)
    var_shape_int32 = fluid.data(name='var_shape_int32', shape=[2], dtype="int32")
    result_4 = fluid.layers.uniform_random(var_shape_int32)
    shape_1 = np.array([3,4]).astype("int64")
    shape_2 = np.array([3,4]).astype("int32")

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(startup_program)
    outs = exe.run(train_program, feed = {'var_shape':shape_1, 'var_shape_int32':shape_2},
                   fetch_list=[result_1, result_2])
    #gen.set_state(cur_state)
    gen.manual_seed(123123143)
    out2 = exe.run(train_program, feed = {'var_shape':shape_1, 'var_shape_int32':shape_2},
                   fetch_list=[result_1, result_2])

    print(outs)
    print(out2)
'''
