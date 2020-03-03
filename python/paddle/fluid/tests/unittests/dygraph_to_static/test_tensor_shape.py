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

import numpy
import paddle.fluid as fluid
import unittest
from paddle.fluid.dygraph.jit import dygraph_to_static_graph

SEED = 2020
numpy.random.seed(SEED)

# def dyfunc_tensor_shape(x):
#
#
#     x = fluid.dygraph.to_variable(x)
#     expand_times = [1] * len(x.shape)
#     fluid.layers.reshape(x, shape=(-1, x.shape[0]))
#     a = x.shape
#     res1 = fluid.layers.reshape(x, shape=x.shape)
#     b = numpy.ones(5)
#     c = fluid.layers.reshape(x, shape=b.shape)
#     res3 = fluid.layers.reshape(x, x.shape)
#     return res1


def dyfunc_tensor_shape_1(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, shape=x.shape)
    return res


def dyfunc_tensor_shape_2(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, x.shape)
    return res


def dyfunc_tensor_shape_3(x):
    x = fluid.dygraph.to_variable(x)
    y = numpy.ones(5)
    res = fluid.layers.reshape(x, shape=y.shape)
    return res


def dyfunc_tensor_shape_4(x):
    x = fluid.dygraph.to_variable(x)
    # res = fluid.layers.reshape(x, shape=(-1, len(x.shape[0])))
    res = fluid.layers.reshape(x, shape=(-1, x.shape[0]))
    return res


def dyfunc_tensor_shape(x):
    x = fluid.dygraph.to_variable(x)
    s = x.shape[0]
    res = fluid.layers.reshape(x, shape=(-1, s))
    return res


test_funcs = [dyfunc_tensor_shape]

# test_funcs = [
#     dyfunc_tensor_shape_1, dyfunc_tensor_shape_2, dyfunc_tensor_shape_3,dyfunc_tensor_shape_4
# ]


class TestTensorShape(unittest.TestCase):
    def setUp(self):
        self.input = numpy.ones(5).astype("int32")

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input).numpy()
            return res

    def get_static_output(self):
        main_program = fluid.Program()
        main_program.random_seed = SEED
        with fluid.program_guard(main_program):
            static_out = dygraph_to_static_graph(self.dygraph_func)(self.input)

        exe = fluid.Executor(fluid.CPUPlace())
        static_res = exe.run(main_program, fetch_list=static_out)

        return static_res[0]

    def test_transformed_static_result(self):
        for func in test_funcs:
            self.dygraph_func = func
            static_res = self.get_static_output()
            dygraph_res = self.get_dygraph_output()
            self.assertTrue(
                numpy.allclose(dygraph_res, static_res),
                msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                                 static_res))


if __name__ == '__main__':
    unittest.main()
