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

import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import dygraph_to_static_func

SEED = 2020
np.random.seed(SEED)


# Use a decorator to test exception
@dygraph_to_static_func
def dyfunc_with_if(x_v):
    if fluid.layers.mean(x_v).numpy()[0] > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    return x_v


@dygraph_to_static_func
def nested_func(x_v):
    x_v = fluid.dygraph.to_variable(x_v)

    def fn1():
        return x_v

    res = fn1()
    return res


class TestRecursiveCall1(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random([10, 16]).astype('float32')
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.init_test_func()

    def init_test_func(self):
        self.dyfunc = nested_func

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            res = self.dyfunc(self.input).numpy()
            return res

    def get_static_output(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            static_out = self.dyfunc(self.input)
        exe = fluid.Executor(self.place)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]

    def test_transformed_static_result(self):
        static_res = self.get_static_output()
        dygraph_res = self.get_dygraph_output()
        self.assertTrue(
            np.allclose(dygraph_res, static_res),
            msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                             static_res))


lambda_fun = lambda x: x


class MyConvLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(MyConvLayer, self).__init__()
        self._conv = fluid.dygraph.Conv2D(
            num_channels=3,
            num_filters=2,
            filter_size=3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.99)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.5)))

    @dygraph_to_static_func
    def forward(self, inputs):
        y = dyfunc_with_if(inputs)
        y = lambda_fun(y)
        y = self.dymethod(y)
        return y

    @dygraph_to_static_func
    def dymethod(self, x_v):
        return x_v


class MyLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

        self.conv = MyConvLayer()
        self.fc = fluid.dygraph.Linear(
            input_dim=5,
            output_dim=1,
            act='relu',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.99)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.5)))

    @dygraph_to_static_func
    def forward(self, inputs):
        h = self.conv(inputs)
        out = self.fc(h)
        return out


class TestRecursiveCall2(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random((1, 3, 3, 5)).astype('float32')
        self.Layer = MyLayer
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            self.dygraph_func = self.Layer()
            fluid.default_startup_program.random_seed = SEED
            fluid.default_main_program.random_seed = SEED
            data = fluid.dygraph.to_variable(self.input)
            res = self.dygraph_func(data)

            return res.numpy()

    def get_static_output(self):
        startup_program = fluid.Program()
        startup_program.random_seed = SEED
        main_program = fluid.Program()
        main_program.random_seed = SEED

        with fluid.program_guard(main_program, startup_program):
            self.dygraph_func = self.Layer()
            data = fluid.layers.assign(self.input)
            static_out = self.dygraph_func(data)

        exe = fluid.Executor(self.place)
        exe.run(startup_program)
        static_res = exe.run(main_program, fetch_list=static_out)
        return static_res[0]

    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        self.assertTrue(
            np.allclose(dygraph_res, static_res),
            msg='dygraph is {}\n static_res is \n{}'.format(dygraph_res,
                                                            static_res))


if __name__ == '__main__':
    unittest.main()
