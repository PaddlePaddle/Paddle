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

import astor
from paddle.utils import gast
import inspect
import numpy as np
import textwrap
import unittest

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.dygraph_to_static.utils import func_to_source_code
import paddle.jit.dy2static as _jst

from ifelse_simple_func import dyfunc_with_if_else, dyfunc_with_if_else_early_return1, dyfunc_with_if_else_early_return2

np.random.seed(0)


# TODO(Aurelius): Currently, `declarative` don't support decorate the function
# that contains layers with initialized operation, like `fc = linear(10, 3)`.
# Because initialized ops will be added into program and be executed many times.
# The parameters are assumed to initialized outside of the function.
def simple_func(x, weight_numpy):
    x = fluid.dygraph.to_variable(x)
    w = fluid.dygraph.to_variable(weight_numpy)
    y = fluid.layers.matmul(x, w)
    z = paddle.mean(y)
    return z


@declarative
def decorated_simple_func(x, weight_numpy):
    x = fluid.dygraph.to_variable(x)
    w = fluid.dygraph.to_variable(weight_numpy)
    y = fluid.layers.matmul(x, w)
    z = paddle.mean(y)
    return z


def get_source_code(func):
    raw_code = inspect.getsource(func)
    code = textwrap.dedent(raw_code)
    root = gast.parse(code)
    source_code = astor.to_source(gast.gast_to_ast(root))
    return source_code


class StaticCode1():

    def dyfunc_with_if_else(x_v, label=None):
        loss = _jst.UndefinedVar('loss')
        __return_1 = _jst.UndefinedVar('__return_1')
        __return_0 = _jst.UndefinedVar('__return_0')
        __return_value_0 = None

        def get_args_0():
            nonlocal x_v
            return x_v,

        def set_args_0(__args):
            nonlocal x_v
            x_v, = __args

        def true_fn_0():
            nonlocal x_v
            x_v = x_v - 1
            return

        def false_fn_0():
            nonlocal x_v
            x_v = x_v + 1
            return

        _jst.IfElse(paddle.mean(x_v)[0] > 5,
                    true_fn_0,
                    false_fn_0,
                    get_args_0,
                    set_args_0, ('x_v', ),
                    push_pop_names=None)

        def get_args_1():
            nonlocal __return_0, __return_1, __return_value_0, loss
            return __return_0, __return_1, __return_value_0, loss

        def set_args_1(__args):
            nonlocal __return_0, __return_1, __return_value_0, loss
            __return_0, __return_1, __return_value_0, loss = __args

        def true_fn_1():
            nonlocal __return_0, __return_1, __return_value_0, loss
            loss = fluid.layers.cross_entropy(x_v, label)
            __return_0 = _jst.create_bool_as_type(label is not None, True)
            __return_value_0 = loss
            return

        def false_fn_1():
            nonlocal __return_0, __return_1, __return_value_0, loss
            __return_1 = _jst.create_bool_as_type(label is not None, True)
            __return_value_0 = x_v
            return

        _jst.IfElse(label is not None,
                    true_fn_1,
                    false_fn_1,
                    get_args_1,
                    set_args_1,
                    ('__return_0', '__return_1', '__return_value_0', 'loss'),
                    push_pop_names=None)
        return __return_value_0


class StaticCode2():
    # TODO: Transform return statement
    def dyfunc_with_if_else(x_v, label=None):
        loss = _jst.UndefinedVar('loss')
        __return_3 = _jst.UndefinedVar('__return_3')
        __return_2 = _jst.UndefinedVar('__return_2')
        __return_value_1 = None

        def get_args_2():
            nonlocal x_v
            return x_v,

        def set_args_2(__args):
            nonlocal x_v
            x_v, = __args

        def true_fn_2():
            nonlocal x_v
            x_v = x_v - 1
            return

        def false_fn_2():
            nonlocal x_v
            x_v = x_v + 1
            return

        _jst.IfElse(paddle.mean(x_v)[0] > 5,
                    true_fn_2,
                    false_fn_2,
                    get_args_2,
                    set_args_2, ('x_v', ),
                    push_pop_names=None)

        def get_args_3():
            nonlocal __return_2, __return_3, __return_value_1, loss
            return __return_2, __return_3, __return_value_1, loss

        def set_args_3(__args):
            nonlocal __return_2, __return_3, __return_value_1, loss
            __return_2, __return_3, __return_value_1, loss = __args

        def true_fn_3():
            nonlocal __return_2, __return_3, __return_value_1, loss
            loss = fluid.layers.cross_entropy(x_v, label)
            __return_2 = _jst.create_bool_as_type(label is not None, True)
            __return_value_1 = loss
            return

        def false_fn_3():
            nonlocal __return_2, __return_3, __return_value_1, loss
            __return_3 = _jst.create_bool_as_type(label is not None, True)
            __return_value_1 = x_v
            return

        _jst.IfElse(label is not None,
                    true_fn_3,
                    false_fn_3,
                    get_args_3,
                    set_args_3,
                    ('__return_2', '__return_3', '__return_value_1', 'loss'),
                    push_pop_names=None)
        return __return_value_1


class NetWithError(fluid.dygraph.layers.Layer):

    @declarative
    def forward(self, x):
        linear = fluid.dygraph.Linear(32, 64)
        y = linear(x)
        return y


class TestDygraphToStaticCode(unittest.TestCase):

    def setUp(self):
        # set to print all string diff when assertEqual fails
        self.maxDiff = None

    def test_decorator(self):
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        print(code)
        answer = get_source_code(StaticCode1.dyfunc_with_if_else)
        self.assertEqual(
            answer.replace('\n', '').replace(' ', ''),
            code.replace('\n', '').replace(' ', ''))

    def test_program_translator(self):
        answer = get_source_code(StaticCode2.dyfunc_with_if_else)
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        print(code)
        self.assertEqual(
            answer.replace('\n', '').replace(' ', ''),
            code.replace('\n', '').replace(' ', ''))


class TestEnableDeclarative(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(30, 10, 32).astype('float32')
        self.weight = np.random.randn(32, 64).astype('float32')
        self.program_translator = ProgramTranslator()

    def test_raise_error(self):
        with fluid.dygraph.guard():
            self.program_translator.enable(True)
            net = NetWithError()
            with self.assertRaises(ValueError):
                net(fluid.dygraph.to_variable(self.x))

    def test_enable_disable_get_output(self):
        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_output = self.program_translator.get_output(
                simple_func, self.x, self.weight)

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = self.program_translator.get_output(
                simple_func, self.x, self.weight)
            np.testing.assert_allclose(static_output.numpy(),
                                       dygraph_output.numpy(),
                                       rtol=1e-05,
                                       atol=1e-4)

    def test_enable_disable_get_func(self):

        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_func = self.program_translator.get_func(simple_func)
            self.assertTrue(callable(static_func))
            static_output = static_func(self.x, self.weight)
            self.assertTrue(isinstance(static_output, fluid.Variable))

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_func = self.program_translator.get_func(simple_func)
            self.assertTrue(callable(dygraph_func))
            dygraph_output = dygraph_func(self.x, self.weight)
            self.assertTrue(
                isinstance(dygraph_output,
                           (fluid.core.VarBase, fluid.core.eager.Tensor)))

    def test_enable_disable_get_program(self):

        self.program_translator.enable(True)
        static_output = self.program_translator.get_program(
            simple_func, self.x, self.weight)
        self.assertTrue(isinstance(static_output, tuple))
        self.assertEqual(len(static_output), 4)
        self.assertTrue(isinstance(static_output[0], fluid.Program))
        self.assertTrue(isinstance(static_output[1], fluid.Program))
        # Check all inputs and outputs are Variable
        for var in static_output[2]:
            self.assertTrue(isinstance(var, fluid.Variable))

        for var in static_output[3]:
            self.assertTrue(isinstance(var, fluid.Variable))

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = self.program_translator.get_program(
                simple_func, self.x, self.weight)
            self.assertTrue(
                isinstance(dygraph_output,
                           (fluid.core.VarBase, fluid.core.eager.Tensor)))

    def test_enable_disable_declarative(self):

        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_output = decorated_simple_func(self.x, self.weight)

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = decorated_simple_func(self.x, self.weight)
            np.testing.assert_allclose(static_output.numpy(),
                                       dygraph_output.numpy(),
                                       rtol=1e-05,
                                       atol=1e-4)


class Net(fluid.dygraph.layers.Layer):

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x + 1


class TestErrorWithInitFromStaticMode(unittest.TestCase):

    def setUp(self):
        self.program_translator = ProgramTranslator()
        self.x = np.random.randn(10, 32).astype('float32')

    def test_raise_error(self):
        # disable imperative
        paddle.enable_static()
        net = Net()

        self.program_translator.enable(True)
        with self.assertRaisesRegexp(RuntimeError,
                                     "only available in dynamic mode"):
            self.program_translator.get_output(net.forward, self.x)

        with self.assertRaisesRegexp(RuntimeError,
                                     "only available in dynamic mode"):
            self.program_translator.get_program(net.forward, self.x)


class SwitchModeNet(paddle.nn.Layer):

    def __init__(self):
        super(SwitchModeNet, self).__init__()

    @paddle.jit.to_static
    def forward(self, x):
        return x + 1

    @paddle.jit.to_static
    def foo(self):
        return True


@paddle.jit.to_static
def switch_mode_funciton():
    return True


class TestFunctionTrainEvalMode(unittest.TestCase):

    def test_switch_mode(self):
        paddle.disable_static()
        switch_mode_funciton.eval()
        switch_mode_funciton()
        self.assertEqual(switch_mode_funciton._training, False)
        _, partial_layer = switch_mode_funciton.program_cache.last()[-1]
        self.assertEqual(partial_layer.training, False)

        switch_mode_funciton.train()
        switch_mode_funciton()
        self.assertEqual(switch_mode_funciton._training, True)
        _, partial_layer = switch_mode_funciton.program_cache.last()[-1]
        self.assertEqual(partial_layer.training, True)

    def test_raise_error(self):
        paddle.disable_static()
        net = SwitchModeNet()

        self.assertEqual(net.training, True)
        with self.assertRaises(RuntimeError):
            net.forward.eval()

        net.eval()
        self.assertEqual(net.training, False)
        with self.assertRaises(RuntimeError):
            net.foo.train()


class TestIfElseEarlyReturn(unittest.TestCase):

    def test_ifelse_early_return1(self):
        answer = np.zeros([2, 2]) + 1
        static_func = paddle.jit.to_static(dyfunc_with_if_else_early_return1)
        out = static_func()
        np.testing.assert_allclose(answer, out[0].numpy(), rtol=1e-05)

    def test_ifelse_early_return2(self):
        answer = np.zeros([2, 2]) + 3
        static_func = paddle.jit.to_static(dyfunc_with_if_else_early_return2)
        out = static_func()
        np.testing.assert_allclose(answer, out[0].numpy(), rtol=1e-05)


class TestRemoveCommentInDy2St(unittest.TestCase):

    def func_with_comment(self):
        # Comment1
        x = paddle.to_tensor([1, 2, 3])
        # Comment2
        # Comment3
        y = paddle.to_tensor([4, 5, 6])

    def test_remove_comment(self):
        code_string = func_to_source_code(self.func_with_comment)
        self.assertEqual('#' not in code_string, True)


if __name__ == '__main__':
    with fluid.framework._test_eager_guard():
        unittest.main()
