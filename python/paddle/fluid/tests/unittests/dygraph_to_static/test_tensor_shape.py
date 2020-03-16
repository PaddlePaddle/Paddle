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

import unittest
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import dygraph_to_static_graph


def dyfunc_tensor_shape_1(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, shape=x.shape)
    return res


def dyfunc_tensor_shape_2(x):
    x = fluid.dygraph.to_variable(x)
    shape = x.shape
    shape2 = shape
    res = fluid.layers.reshape(x, shape2)
    return res


def dyfunc_tensor_shape_3(x):
    # Don't transform y.shape because y is numpy.ndarray
    x = fluid.dygraph.to_variable(x)
    y = numpy.ones(5)
    res = fluid.layers.reshape(x, shape=y.shape)
    return res


def dyfunc_tensor_shape_4(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, shape=(-1, x.shape[0], len(x.shape)))
    return res


def dyfunc_tensor_shape_5(x):
    # `res = fluid.layers.reshape(x, shape=(-1, s))` to
    # `res = fluid.layers.reshape(x, shape=(-1, fluid.layers.shape(x)[0]))`
    x = fluid.dygraph.to_variable(x)
    s = x.shape[0]
    res = fluid.layers.reshape(x, shape=(-1, s))
    return res


test_funcs = [
    dyfunc_tensor_shape_1, dyfunc_tensor_shape_2, dyfunc_tensor_shape_3,
    dyfunc_tensor_shape_4, dyfunc_tensor_shape_5
]


class TestTensorShape(unittest.TestCase):
    def setUp(self):
        self.input = numpy.ones(5).astype("int32")
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input).numpy()
            return res

    def get_static_output(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            static_out = dygraph_to_static_graph(self.dygraph_func)(self.input)

        exe = fluid.Executor(self.place)
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
