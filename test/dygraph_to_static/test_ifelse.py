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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    IrMode,
    ToStaticMode,
    disable_test_case,
    enable_to_static_guard,
    test_ast_only,
    test_legacy_and_pir,
    test_pir_only,
)
from ifelse_simple_func import (
    NetWithControlFlowIf,
    add_fn,
    dyfunc_empty_nonlocal,
    dyfunc_ifelse_ret_int1,
    dyfunc_ifelse_ret_int2,
    dyfunc_ifelse_ret_int3,
    dyfunc_ifelse_ret_int4,
    dyfunc_with_if_else,
    dyfunc_with_if_else2,
    dyfunc_with_if_else3,
    dyfunc_with_if_else_with_list_generator,
    if_tensor_case,
    if_with_and_or,
    if_with_and_or_1,
    if_with_and_or_2,
    if_with_and_or_3,
    if_with_and_or_4,
    if_with_class_var,
    loss_fn,
    nested_if_else,
    nested_if_else_2,
    nested_if_else_3,
)

import paddle
import paddle.nn.functional as F
from paddle.jit.dy2static.utils import Dygraph2StaticException

np.random.seed(1)


class TestDy2staticException(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = None
        self.error = "Your if/else have different number of return value."

    @test_ast_only
    def test_error(self):
        if self.dyfunc:
            with self.assertRaisesRegex(Dygraph2StaticException, self.error):
                with enable_to_static_guard(True):
                    self.assertTrue(paddle.jit.to_static(self.dyfunc)(self.x))


class TestDy2StIfElseRetInt2(TestDy2staticException):
    def setUp(self):
        self.x = np.random.random([5]).astype('float32')
        self.error = "Your if/else have different number of return value."
        self.dyfunc = dyfunc_ifelse_ret_int2


class TestDygraphIfElse(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        x_v = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(x_v)
        else:
            ret = self.dyfunc(x_v)
        return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElse2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else2

    # TODO(dev): fix AST mode
    @disable_test_case((ToStaticMode.AST, IrMode.PT))
    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElse3(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else3

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        x_v = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(x_v)
        else:
            ret = self.dyfunc(x_v)
        return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElse4(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_empty_nonlocal


class TestDygraphIfElseWithListGenerator(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else_with_list_generator

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphNestedIfElse(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        x_v = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(x_v)
        else:
            ret = self.dyfunc(x_v)
        return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphNestedIfElse2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_2

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphNestedIfElse3(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_3

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        x_v = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(x_v)
        else:
            ret = self.dyfunc(x_v)
        return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


def dyfunc_ifExp_with_while(x):
    y = [x]

    def add_fn(x):
        x = x + 1
        return x

    def cond(i, ten, y):
        return i < ten

    def map_func(func, tensor_list):
        return [func(x) for x in tensor_list]

    def body(i, ten, y):
        # It will be converted into `layers.cond` as followed.
        # map_func(lambda x: paddle.static.nn.cond(i==0, lambda: x, lambda: add_fn(x), y)
        y = map_func(lambda x: x if (i == 0) is not None else add_fn(x), y)
        i += 1
        return [i, ten, y]

    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    ten = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=10)
    i, ten, y = paddle.static.nn.while_loop(cond, body, [i, ten, y])
    return y[0]


# class TestDygraphIfElse6(TestDygraphIfElse):
#     def setUp(self):
#         self.x = np.random.random([10, 16]).astype('float32')
#         self.dyfunc = dyfunc_ifExp_with_while


def dyfunc_ifExp(x):
    y = [x]

    def add_fn(x):
        x = x + 1
        return x

    def map_func(func, tensor_list):
        return [func(x) for x in tensor_list]

    i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    # It will be converted into `layers.cond` as followed.
    # map_func(lambda x: paddle.static.nn.cond(i==1, lambda: x, lambda: add_fn(x), y)
    # `if (Tensor) == 1` is supported in dygraph.
    y = map_func(lambda x: x if i == 1 else add_fn(x), y)
    return y[0]


class TestDygraphIfElse7(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_ifExp


class TestDygraphIfElseWithAndOr(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or


class TestDygraphIfElseWithAndOr1(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_1


class TestDygraphIfElseWithAndOr2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_2


class TestDygraphIfElseWithAndOr3(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_3


class TestDygraphIfElseWithAndOr4(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_4


class TestDygraphIfElseWithClassVar(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_class_var


class TestDygraphIfTensor(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_tensor_case

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        x_v = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(x_v)
        else:
            ret = self.dyfunc(x_v)
        return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElseNet(Dy2StTestBase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `paddle.static.nn.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithControlFlowIf

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static=False):
        with enable_to_static_guard(to_static):
            net = paddle.jit.to_static(self.Net())
            x_v = paddle.to_tensor(self.x)
            ret = net(x_v)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


# Test to call function ahead caller.
def relu(x):
    return F.relu(x)


def call_external_func(x, label=None):
    if paddle.mean(x) < 0:
        x_v = x - 1
    else:
        x_v = add_fn(x)

    x_v = relu(x_v)
    if label is not None:
        loss = loss_fn(x_v, label)
        return loss
    return x_v


class TestAst2FuncWithExternalFunc(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = call_external_func


class NetWithExternalFunc(paddle.nn.Layer):
    def forward(self, x, label=None):
        if paddle.mean(x) < 0:
            x_v = x - 1
        else:
            x_v = add_fn(x)

        x_v = softmax(x_v)
        if label is not None:
            loss = loss_fn(x_v, label)
            return loss
        return x_v


# Test to call function behind caller.
def softmax(x):
    return paddle.nn.functional.softmax(x)


class TestNetWithExternalFunc(TestDygraphIfElseNet):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithExternalFunc

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class DiffModeNet1(paddle.nn.Layer):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, y):
        if self.mode == 'train':
            out = x + y
        elif self.mode == 'infer':
            out = x - y
        else:
            raise ValueError('Illegal mode')
        return out


class DiffModeNet2(paddle.nn.Layer):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, y):
        if self.mode == 'train':
            out = x + y
            return out
        elif self.mode == 'infer':
            out = x - y
            return out
        else:
            raise ValueError('Illegal mode')


class TestDiffModeNet(Dy2StTestBase):
    """
    TestCase for the net with different modes
    """

    def setUp(self):
        self.x = paddle.randn([10, 16], 'float32')
        self.y = paddle.randn([10, 16], 'float32')
        self.init_net()

    def init_net(self):
        self.Net = DiffModeNet1

    def _run(self, mode, to_static):
        with enable_to_static_guard(to_static):
            if to_static:
                net = paddle.jit.to_static(self.Net(mode))
            else:
                net = self.Net(mode)
            ret = net(self.x, self.y)
            return ret.numpy()

    def test_train_mode(self):
        self.assertTrue(
            (
                self._run(mode='train', to_static=True)
                == self._run(mode='train', to_static=False)
            ).all()
        )

    def test_infer_mode(self):
        self.assertTrue(
            (
                self._run(mode='infer', to_static=True)
                == self._run(mode='infer', to_static=False)
            ).all()
        )


class TestDiffModeNet2(TestDiffModeNet):
    def init_net(self):
        self.Net = DiffModeNet2


class TestNewVarCreateInOneBranch(Dy2StTestBase):
    def test_var_used_in_another_for(self):
        def case_func(training):
            # targets and targets_list is dynamically defined by training
            if training:
                targets = [1, 2, 3]
                targets_list = [targets]

            num_step = 3
            for i in range(num_step):
                if i > 0:
                    rois, rosi_num = 1, 2
                    # targets is in loop_vars.
                    if training:
                        ros, rosi_num, targets = -1, -2, [-1, -2, -3]
                        targets_list.append(targets)

            return rosi_num

        self.assertEqual(paddle.jit.to_static(case_func)(False), 2)
        self.assertEqual(paddle.jit.to_static(case_func)(True), -2)


class TestDy2StIfElseRetInt1(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([5]).astype('float32')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int1)
        self.out = self.get_dy2stat_out()

    def get_dy2stat_out(self):
        with enable_to_static_guard(True):
            static_func = paddle.jit.to_static(self.dyfunc)
            out = static_func(self.x)
        return out

    @test_ast_only
    def test_ast_to_func(self):
        self.setUp()
        self.assertIsInstance(self.out[0], paddle.Tensor)
        self.assertIsInstance(self.out[1], int)


class TestDy2StIfElseRetInt3(TestDy2StIfElseRetInt1):
    def setUp(self):
        self.x = np.random.random([5]).astype('int64')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int3)
        self.out = self.get_dy2stat_out()

    @test_ast_only
    def test_ast_to_func(self):
        self.setUp()
        self.assertIsInstance(self.out, paddle.Tensor)


class TestDy2StIfElseRetInt4(TestDy2StIfElseRetInt1):
    def setUp(self):
        self.x = np.random.random([5]).astype('float32')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int4)

    @test_ast_only
    def test_ast_to_func(self):
        with enable_to_static_guard(True):
            with self.assertRaises(Dygraph2StaticException):
                static_func = paddle.jit.to_static(self.dyfunc)
                out = static_func(self.x)


class IfElseNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.param = self.create_parameter(
            shape=[3, 2], dtype='float32', is_bias=False
        )

    def forward(self, a, b, c):
        a = paddle.matmul(a, self.param)
        a = paddle.reshape(a, (2, 4))
        cond = paddle.to_tensor([10])
        b = b.broadcast_to(self.param.shape)
        if paddle.equal(cond, 10):
            a_argmax = a.argmax(axis=-1)
            b = b + self.param
        else:
            print(c)
        return b


class TestDy2StIfElseBackward(Dy2StTestBase):
    @test_legacy_and_pir
    def test_run_backward(self):
        a = paddle.randn((4, 3), dtype='float32')
        a.stop_gradient = False
        b = paddle.to_tensor([10]).astype('float32')
        b.stop_gradient = False
        c = paddle.to_tensor([2])
        c.stop_gradient = False

        net = paddle.jit.to_static(IfElseNet())
        net.train()
        out = net(a, b, c)
        out.backward()
        np.testing.assert_allclose(
            (b + net.param).numpy(), out.numpy(), rtol=1e-05
        )


def ifelse_temp_local_var(x):
    if x:
        y = x + 1
    else:
        tmp = x + 2
        y = tmp * 2
    return y


def ifelse_use_undefined_var(x):
    if x:
        y = x + 1
    else:
        tmp = x + 2
        y = tmp * 2
    return tmp + 1


class TestIfElseMaybeUnbound(Dy2StTestBase):
    def test_maybe_unbound(self):
        truethy = paddle.to_tensor(1)
        falsy = paddle.to_tensor(0)

        dygraph_out = ifelse_temp_local_var(truethy)
        static_fn = paddle.jit.to_static(ifelse_temp_local_var)
        static_out = static_fn(truethy)
        np.testing.assert_allclose(dygraph_out.numpy(), static_out.numpy())

        dygraph_out = ifelse_temp_local_var(falsy)
        static_fn = paddle.jit.to_static(ifelse_temp_local_var)
        static_out = static_fn(falsy)
        np.testing.assert_allclose(dygraph_out.numpy(), static_out.numpy())

    @test_ast_only
    @test_pir_only
    def test_use_undefined_var(self):
        truethy = paddle.to_tensor(1)
        falsy = paddle.to_tensor(0)

        static_fn = paddle.jit.to_static(ifelse_use_undefined_var)
        with self.assertRaises(TypeError):
            static_fn(truethy)

        static_fn = paddle.jit.to_static(ifelse_use_undefined_var)
        with self.assertRaises(TypeError):
            static_fn(falsy)


def dynamic_shape_with_constant_promotion(x):
    x_shape0 = x.shape[0]
    if x_shape0 < 10:
        x_shape0 = x.shape[-1]
    return x_shape0


class TestDynamicShapeWithConstantPromotion(Dy2StTestBase):
    @test_ast_only
    @test_pir_only
    def test_dynamic_shape_with_constant_promotion(self):
        x = paddle.randn([5, 3])
        static_fn = paddle.jit.to_static(
            dynamic_shape_with_constant_promotion,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 3],
                    dtype='float32',
                )
            ],
        )
        out = static_fn(x)
        self.assertEqual(out, 3)


if __name__ == '__main__':
    unittest.main()
