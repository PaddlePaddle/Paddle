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

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.framework as framework

from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.nn import Linear

np.random.seed(2020)

place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace(
)


def simple_func(x, weight_numpy):
    weight_initalizer = fluid.initializer.NumpyArrayInitializer(weight_numpy)
    linear = Linear(32, 64, param_attr=weight_initalizer)
    x = fluid.dygraph.to_variable(x)
    y = linear(x)
    z = linear(x)
    return z


def decorated_simple_func(x, weight_numpy):
    weight_initalizer = fluid.initializer.NumpyArrayInitializer(weight_numpy)
    linear = Linear(32, 64, param_attr=weight_initalizer)
    x = fluid.dygraph.to_variable(x)
    y = linear(x)
    z = linear(x)
    return z


class TestDyToStaticSaveLoad(unittest.TestCase):
    def test_save_load_same_result(self):
        x = np.random.randn(30, 10, 32).astype('float32')
        weight = np.random.randn(32, 64).astype('float32')
        with fluid.dygraph.guard(place):
            dygraph_result = simple_func(x, weight)

        main_program, startup_program, inputs, outputs = ProgramTranslator(
        ).get_program(decorated_simple_func, x, weight)
        exe = fluid.Executor(place)
        exe.run(startup_program)
        fluid.save(main_program, "./test_dy2stat_save_load")

        # set vars to zero so that we can test load in same file
        for var in main_program.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                tensor = fluid.global_scope().find_var(var.name).get_tensor()
                tensor.set(np.zeros_like(np.array(tensor)), place)

                # make sure all the paramerter or optimizer var have been set to zero
                tensor_np = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                self.assertEqual(0, np.sum(np.abs(tensor_np)))

        fluid.load(main_program, "./test_dy2stat_save_load")
        static_result = exe.run(main_program,
                                feed={inputs[0].name: x},
                                fetch_list=outputs)
        self.assertTrue(np.allclose(dygraph_result.numpy(), static_result))


if __name__ == '__main__':
    unittest.main()
