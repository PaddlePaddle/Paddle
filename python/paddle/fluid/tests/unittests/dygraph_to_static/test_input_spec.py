# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, declarative, ProgramTranslator, Layer, jit
from paddle.fluid.dygraph import VariableSpec
import decorator

import unittest

program_trans = ProgramTranslator()


class TT(object):
    def __init__(self, func):
        self._func = func

    # implement this for Descriptors to parse instance of class
    # see: https://docs.python.org/3/reference/datamodel.html#implementing-descriptors
    def __get__(self, instance, owner):
        return self._func.__get__(instance, owner)

    def __call__(self, *args, **kwargs):
        print('args in dd :', args)
        return self._func(*args, **kwargs)

    def code(self):
        print("code")


class SS(object):
    def __init__(self, func, info):
        self._func = func
        self._info = info

    def __get__(self, instance, owner):
        return self._func.__get__(instance, owner)

    def __call__(self, *args, **kwargs):
        print('args in ss:', args, kwargs)
        print("in SS.__call__")
        return self._func(*args, **kwargs)


def dd(func=None, info=None):
    def _decorate_(inner_func):
        d_name = "dd_name"

        tt = TT(inner_func)
        ss = SS(inner_func, info)

        ss._tt_val = tt
        ss.__name__ = inner_func.__name__
        ss._decorator_name = d_name
        ss.__wrapped__ = inner_func
        ss.__doc__ = inner_func.__doc__
        if hasattr(inner_func, "__module__"):
            ss.__module__ = inner_func.__module__
        print(ss.__dict__)

        return ss

    # for `dd(foo, ...)`
    if func is not None:
        print('bere')
        return _decorate_(func)

    # for `@dd(...)`
    return _decorate_


def bar(a, b, c=1, d=2):
    z = a + b
    return z


def test_d():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        d_bar = dd(bar)
        x = to_variable(np.ones([4, 10]).astype('float32'))

        y = d_bar(x, x)
        print("test_d func: ", d_bar)
        print(y.numpy())


class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = fluid.dygraph.Linear(10, 3)

    @declarative(
        input_signature=[VariableSpec(
            shape=[None, 10], dtype='float32')])
    # def forward(self, x, a=1, b=2, **kwargs):
    # @dd(info='info')
    def forward(self, x, a=1, b=2):
        y = self.linear(x)
        return y


def test():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        net = SimpleNet()
        x = to_variable(np.ones([4, 10]).astype('float32'))
        y = net(x)
        print("net.forward function: ", net.forward)
        print(y)


class TestInputSpec(unittest.TestCase):
    def setUp(self):
        self.program_trans = ProgramTranslator()

    def test_input_spec(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))

            # TODO: support to transform directly when specific input_signature
            net = SimpleNet()
            y = net(x, a=1)
            print(program_trans.get_program_cache().concrete_programs())


@declarative
def foo(a, b, c=1, d=2):
    z = a + b
    return z


class TestDifferentInputSpecCacheProgram(unittest.TestCase):
    def test_with_different_input(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x_data = np.ones([16, 10]).astype('float32')
            y_data = np.ones([10]).astype('float32') * 2
            z_data = np.ones([10]).astype('float32') * 2.2

            # [16, 10] + [10] (varbase)
            out_1 = foo(to_variable(x_data), to_variable(y_data))
            self.assertTrue(np.allclose(x_data + y_data, out_1.numpy()))
            concrete_programs_1 = program_trans.get_program_cache(
            ).concrete_programs()
            self.assertTrue(len(concrete_programs_1) == 1)

            # [16, 10] + [10] (numpy)
            out_2 = foo(to_variable(x_data), y_data)
            self.assertTrue(np.allclose(x_data + y_data, out_2.numpy()))
            concrete_programs_2 = program_trans.get_program_cache(
            ).concrete_programs()
            self.assertTrue(len(concrete_programs_2) == 2)

            # [16, 10] + [10] (numpy)
            out_3 = foo(to_variable(x_data), z_data)
            self.assertTrue(np.allclose(x_data + z_data, out_3.numpy()))
            concrete_programs_3 = program_trans.get_program_cache(
            ).concrete_programs()
            # hit cache with program in concrete_programs_2
            self.assertTrue(concrete_programs_3 == concrete_programs_2)

            # [16, 10] + [10] (numpy) with other different arguments (c=3)
            out_4 = foo(to_variable(x_data), z_data, 3)
            self.assertTrue(np.allclose(x_data + z_data, out_4.numpy()))
            concrete_programs_4 = program_trans.get_program_cache(
            ).concrete_programs()
            # create a new program
            self.assertTrue(len(concrete_programs_4) == 3)


if __name__ == '__main__':
    unittest.main()
    # test()
    # test_d()
