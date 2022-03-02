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

import numpy as np
import unittest

import paddle
import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.fluid.dygraph import to_variable, declarative, ProgramTranslator, Layer, jit
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ConcreteProgram, StaticFunction

from test_basic_api_transformation import dyfunc_to_variable

program_trans = ProgramTranslator()


class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = fluid.dygraph.Linear(10, 3)

    @declarative(input_spec=[InputSpec(shape=[None, 10], dtype='float32')])
    def forward(self, x, a=1, b=2):
        y = self.inner_function(x)
        return y

    # `declarative` is not essential, add it to test for robustness.
    @declarative
    def inner_function(self, x):
        y = self.linear(x)
        return y

    def add_func(self, x, y):
        z = x + y
        return z

    @declarative(input_spec=[[InputSpec([None, 10]), InputSpec([None, 10])]])
    def func_with_list(self, l, int_val=1):
        x, y = l
        z = x + y
        z = z + int_val
        return z

    @declarative(input_spec=[{
        'x': InputSpec([None, 10]),
        'y': InputSpec([None, 10])
    }])
    def func_with_dict(self, d):
        x = d['x']
        y = d['y']
        z = x + y

        return z

    @declarative(input_spec=[[
        InputSpec([None]), {
            'x': InputSpec([None, 10]),
            'y': InputSpec([None, 10])
        }
    ]])
    def func_with_list_dict(self, dl):
        bias = dl[0]
        x = dl[1]['x']
        y = dl[1]['y']

        z = x + y
        z = z + bias

        return z


class TestStaticFunctionInstance(unittest.TestCase):
    def test_instance_same_class(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            net_1 = SimpleNet()
            net_2 = SimpleNet()

            self.assertTrue(isinstance(net_1.forward, StaticFunction))
            self.assertTrue(isinstance(net_2.forward, StaticFunction))
            self.assertNotEqual(net_1.forward, net_2.forward)

            # convert layer into static progam of net_1
            net_1.forward.concrete_program
            self.assertTrue(len(net_1.forward.program_cache) == 1)
            # check no conversion applid with net_2
            self.assertTrue(len(net_2.forward.program_cache) == 0)


class TestInputSpec(unittest.TestCase):
    def setUp(self):
        pass

    def test_with_input_spec(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))
            y = to_variable(np.ones([4, 10]).astype('float32') * 2)
            int_val = 4.

            net = SimpleNet()

            # 1. each method holds independent program cache
            out = net(x)
            self.assertTrue(len(net.forward.program_cache) == 1)

            # 2. test save load
            net.inner_function(x)
            jit.save(net, './simple_net')
            infer_net = fluid.dygraph.jit.load('./simple_net')
            pred = infer_net(x)
            self.assertTrue(np.allclose(out.numpy(), pred.numpy()))

            # 3. we can decorate any method
            x_2 = to_variable(np.ones([4, 20]).astype('float32'))
            # uses `declarative(func)` instead of `@declarative`
            net.add_func = declarative(net.add_func)
            out = net.add_func(x_2, np.ones([20]).astype('float32'))
            self.assertTrue(len(net.add_func.program_cache) == 1)

            # 5. test input with list
            out = net.func_with_list([x, y], int_val)

            # 6. test input with dict
            out = net.func_with_dict({'x': x, 'y': y})

            # 7. test input with lits contains dict
            int_np = np.ones([1]).astype('float32')
            out = net.func_with_list_dict([int_np, {'x': x, 'y': y}])

    def test_with_error(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))
            y = to_variable(np.ones([4, 10]).astype('float32') * 2)
            int_val = 4.

            net = SimpleNet()

            # 1. kwargs and input_spec should not be specificed in same time
            with self.assertRaises(ValueError):
                net(x, a=1, other_kwarg=2)

            # 2. requires len(input_spec) <= len(args)
            with self.assertRaises(ValueError):
                net.add_func = declarative(
                    net.add_func,
                    input_spec=[
                        InputSpec([-1, 10]), InputSpec([-1, 10]),
                        InputSpec([10])
                    ])
                net.add_func(x, y)

    def test_concrete_program(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))
            y = to_variable(np.ones([4, 10]).astype('float32') * 2)
            int_val = 4.

            net = SimpleNet()
            # We can get concrete_program by specificing InputSpec information. Faking input is no need.
            net.add_func = declarative(
                net.add_func,
                input_spec=[
                    InputSpec([-1, 10]), InputSpec(
                        [-1, 10], name='y')
                ])
            cp1 = net.add_func.concrete_program
            self.assertTrue(cp1.inputs[-1].shape == (-1, 10))
            self.assertTrue(cp1.inputs[-1].name == 'y')

            # generate another program
            net.add_func = declarative(
                net.add_func,
                input_spec=[InputSpec([10]), InputSpec(
                    [10], name='label')])
            cp2 = net.add_func.concrete_program
            self.assertTrue(cp2.inputs[-1].shape == (10, ))
            self.assertTrue(cp2.inputs[-1].name == 'label')
            # Note(Aurelius84): New instance will be returned if we use `declarative(foo)` every time.
            # So number of cache program is 1.
            self.assertTrue(len(net.add_func.program_cache) == 1)
            self.assertTrue(cp1 != cp2)


def foo_func(a, b, c=1, d=2):
    z = a + b
    return z


class TestDifferentInputSpecCacheProgram(unittest.TestCase):
    def setUp(self):
        program_trans.enable(True)

    def test_with_different_input(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x_data = np.ones([16, 10]).astype('float32')
            y_data = np.ones([10]).astype('float32') * 2
            z_data = np.ones([10]).astype('float32') * 2.2

            foo = declarative(foo_func)

            # [16, 10] + [10] (varbase)
            out_1 = foo(to_variable(x_data), to_variable(y_data))
            self.assertTrue(np.allclose(x_data + y_data, out_1.numpy()))
            self.assertTrue(len(foo.program_cache) == 1)
            self.assertTrue(len(foo.program_cache.concrete_programs()) == 1)
            first_program = foo.program_cache.last()

            # [16, 10] + [10] (numpy)
            out_2 = foo(to_variable(x_data), y_data)
            self.assertTrue(np.allclose(x_data + y_data, out_2.numpy()))
            self.assertTrue(len(foo.program_cache) == 1)

            # [16, 10] + [10] (numpy)
            out_3 = foo(to_variable(x_data), z_data)
            self.assertTrue(np.allclose(x_data + z_data, out_3.numpy()))
            # hit cache program
            self.assertTrue(len(foo.program_cache) == 1)

            # [16, 10] + [10] (numpy) with other different arguments (c=3)
            out_4 = foo(to_variable(x_data), z_data, 3)
            self.assertTrue(np.allclose(x_data + z_data, out_4.numpy()))
            # create a new program
            self.assertTrue(len(foo.program_cache) == 2)

            # test for recent program
            foo(to_variable(x_data), y_data)
            recent_program = foo.program_cache.last()
            self.assertTrue(first_program == recent_program)

    def test_get_concrete_program(self):

        foo = declarative(foo_func)

        # 1. specific InputSpec for `x`/`y`
        concrete_program_1 = foo.get_concrete_program(
            InputSpec([None, 10]), InputSpec([10]))
        self.assertTrue(len(foo.program_cache) == 1)

        # 2. specific `c`/`d` explicitly with same default value
        concrete_program_2 = foo.get_concrete_program(
            InputSpec([None, 10]), InputSpec([10]), 1, 2)
        self.assertTrue(concrete_program_2 == concrete_program_1)
        self.assertTrue(len(foo.program_cache) == 1)

        # 3. specific `c` = 2
        concrete_program_3 = foo.get_concrete_program(
            InputSpec([None, 10]), InputSpec([10]), c=2)
        self.assertTrue(concrete_program_3 != concrete_program_1)
        self.assertTrue(len(foo.program_cache) == 2)

        # 4. specific x.shape = [10]
        concrete_program_4 = foo.get_concrete_program(
            InputSpec([10]), InputSpec([10]))
        self.assertTrue(concrete_program_4 != concrete_program_1)
        self.assertTrue(len(foo.program_cache) == 3)

        # 5. only specific InputSpec of x
        with self.assertRaises(ValueError):
            concrete_program_5 = foo.get_concrete_program(InputSpec([10]))

        # 6. specific unknown kwargs `e`=4
        with self.assertRaises(TypeError):
            concrete_program_5 = foo.get_concrete_program(
                InputSpec([10]), InputSpec([10]), e=4)

    def test_concrete_program(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):

            # usage 1
            foo_1 = paddle.jit.to_static(
                foo_func,
                input_spec=[
                    InputSpec(
                        [10], name='x'), InputSpec(
                            [10], name='y')
                ])
            self.assertTrue(isinstance(foo_1.concrete_program, ConcreteProgram))

            # usage 2
            foo_2 = paddle.jit.to_static(foo_func)
            out = foo_2(paddle.rand([10]), paddle.rand([10]))
            self.assertTrue(isinstance(foo_2.concrete_program, ConcreteProgram))

            # raise error
            foo_3 = paddle.jit.to_static(foo_func)
            with self.assertRaises(ValueError):
                foo_3.concrete_program


class TestInputDefaultName(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.net = SimpleNet()

    def assert_default_name(self, func_name, input_names):
        decorated_func = getattr(self.net, func_name)

        spec_names = [x.name for x in decorated_func.inputs]
        self.assertListEqual(spec_names, input_names)

    def test_common_input(self):
        self.assert_default_name('forward', ['x'])

    def test_list_input(self):
        self.assert_default_name('func_with_list', ['l_0', 'l_1'])

    def test_dict_input(self):
        self.assert_default_name('func_with_dict', ['x', 'y'])

    def test_nest_input(self):
        self.assert_default_name('func_with_list_dict', ['dl_0', 'x', 'y'])


class TestDeclarativeAPI(unittest.TestCase):
    def test_error(self):
        func = declarative(dyfunc_to_variable)

        paddle.enable_static()

        # Failed to run the callable object decorated by '@paddle.jit.to_static'
        # if it does NOT in dynamic mode.
        with self.assertRaises(RuntimeError):
            func(np.ones(5).astype("int32"))

        program_trans.enable(False)
        with self.assertRaises(AssertionError):
            # AssertionError: We Only support to_variable in imperative mode,
            #  please use fluid.dygraph.guard() as context to run it in imperative Mode
            func(np.ones(5).astype("int32"))


class TestDecorateModelDirectly(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        program_trans.enable(True)
        self.x = to_variable(np.ones([4, 10]).astype('float32'))

    def test_fake_input(self):
        net = SimpleNet()
        net = declarative(net)
        y = net(self.x)
        self.assertTrue(len(net.forward.program_cache) == 1)

    def test_input_spec(self):
        net = SimpleNet()
        net = declarative(net, input_spec=[InputSpec([None, 8, 10])])
        self.assertTrue(len(net.forward.inputs) == 1)
        self.assertTrue(len(net.forward.program_cache) == 1)
        input_shape = net.forward.inputs[0].shape
        self.assertListEqual(list(input_shape), [-1, 8, 10])

        # redecorate
        net = declarative(net, input_spec=[InputSpec([None, 16, 10])])
        input_shape = net.forward.inputs[0].shape
        self.assertListEqual(list(input_shape), [-1, 16, 10])


class TestErrorWithInitFromStaticMode(unittest.TestCase):
    def test_raise_error(self):
        # disable imperative
        paddle.enable_static()

        net = SimpleNet()
        with self.assertRaisesRegexp(RuntimeError,
                                     "only available in dynamic mode"):
            net.forward.concrete_program

        with self.assertRaisesRegexp(RuntimeError,
                                     "only available in dynamic mode"):
            net.forward.inputs

        with self.assertRaisesRegexp(RuntimeError,
                                     "only available in dynamic mode"):
            net.forward.outputs


class CallNonForwardFuncNet(paddle.nn.Layer):
    def __init__(self):
        super(CallNonForwardFuncNet, self).__init__()
        self.sub = CallNonForwardFuncSubNet()

    @paddle.jit.to_static
    def forward(self):
        return self.sub.func()


class CallNonForwardFuncSubNet(paddle.nn.Layer):
    def __init__(self):
        super(CallNonForwardFuncSubNet, self).__init__()
        self.a = paddle.to_tensor([1, 2])

    def func(self):
        x = self.a * 2
        return x


class TestCallNonForwardFunc(unittest.TestCase):
    def test_call_non_forward(self):
        paddle.disable_static()
        net = CallNonForwardFuncNet()
        out = net()
        self.assertEqual(out.numpy().tolist(), [2, 4])
        paddle.enable_static()


class SetBuffersNet1(paddle.nn.Layer):
    def __init__(self):
        super(SetBuffersNet1, self).__init__()
        self.a = paddle.to_tensor([1])

    @paddle.jit.to_static
    def forward(self):
        self.a = self.a + 1
        return self.a


class SetBuffersNet2(paddle.nn.Layer):
    def __init__(self):
        super(SetBuffersNet2, self).__init__()
        self.b = paddle.to_tensor([2])

    @paddle.jit.to_static
    def forward(self):
        self.b = None
        self.b = paddle.to_tensor([3])
        return self.b


class TestSetBuffers(unittest.TestCase):
    def test_set_buffers1(self):
        paddle.disable_static()
        net = SetBuffersNet1()
        out = net()
        self.assertEqual(out.numpy().tolist(), [2])
        paddle.jit.save(net, './SetBuffersNet1')
        paddle.enable_static()

    def test_set_buffers2(self):
        paddle.disable_static()
        net = SetBuffersNet2()
        with self.assertRaises(RuntimeError):
            out = net()
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
