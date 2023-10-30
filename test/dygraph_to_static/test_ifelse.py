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
from dygraph_to_static_utils_new import (
    Dy2StTestBase,
    test_ast_only,
    test_legacy_and_pir,
)
from ifelse_simple_func import (
    NetWithControlFlowIf,
    add_fn,
    base,
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
from paddle.base import core
from paddle.jit.dy2static.utils import Dygraph2StaticException

np.random.seed(1)

if base.is_compiled_with_cuda():
    place = base.CUDAPlace(0)
else:
    place = base.CPUPlace()


class TestDy2staticException(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = None
        self.error = "Your if/else have different number of return value."

    @test_ast_only
    @test_legacy_and_pir
    def test_error(self):
        if self.dyfunc:
            with self.assertRaisesRegex(Dygraph2StaticException, self.error):
                paddle.jit.enable_to_static(True)
                self.assertTrue(paddle.jit.to_static(self.dyfunc)(self.x))
        paddle.base.dygraph.base.global_var._in_to_static_mode_ = False
        paddle.jit.enable_to_static(False)


class TestDygraphIfElse(Dy2StTestBase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `base.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        with base.dygraph.guard(place):
            x_v = base.dygraph.to_variable(self.x)
            if to_static:
                ret = paddle.jit.to_static(self.dyfunc)(x_v)
            else:
                ret = self.dyfunc(x_v)
            return ret.numpy()

    @test_legacy_and_pir
    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElse2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else2


class TestDygraphIfElse3(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else3


class TestDygraphIfElse4(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_empty_nonlocal


class TestDygraphIfElseWithListGenerator(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else_with_list_generator


class TestDygraphNestedIfElse(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else

    def _run_static(self):
        return self._run_dygraph(to_static=True)

    def _run_dygraph(self, to_static=False):
        with base.dygraph.guard(place):
            x_v = paddle.to_tensor(self.x)
            if to_static:
                ret = paddle.jit.to_static(self.dyfunc)(x_v)
            else:
                ret = self.dyfunc(x_v)
            return ret.numpy()

    @test_legacy_and_pir
    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphNestedIfElse2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_2


class TestDygraphNestedIfElse3(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else_3


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


class TestDygraphIfElse6(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_ifExp_with_while


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
        with base.dygraph.guard(place):
            x_v = paddle.to_tensor(self.x)
            if to_static:
                ret = paddle.jit.to_static(self.dyfunc)(x_v)
            else:
                ret = self.dyfunc(x_v)
            return ret.numpy()

    @test_legacy_and_pir
    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElseNet(Dy2StTestBase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `base.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithControlFlowIf

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static=False):
        paddle.jit.enable_to_static(to_static)

        with base.dygraph.guard(place):
            net = self.Net()
            x_v = base.dygraph.to_variable(self.x)
            ret = net(x_v)
            return ret.numpy()

    @test_legacy_and_pir
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
    @paddle.jit.to_static
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

    @test_legacy_and_pir
    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class DiffModeNet1(paddle.nn.Layer):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    @paddle.jit.to_static
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

    @paddle.jit.to_static
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
        paddle.jit.enable_to_static(to_static)

        net = self.Net(mode)
        ret = net(self.x, self.y)
        return ret.numpy()

    @test_legacy_and_pir
    def test_train_mode(self):
        self.assertTrue(
            (
                self._run(mode='train', to_static=True)
                == self._run(mode='train', to_static=False)
            ).all()
        )

    @test_legacy_and_pir
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
    @test_legacy_and_pir
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
        paddle.jit.enable_to_static(True)
        static_func = paddle.jit.to_static(self.dyfunc)
        out = static_func(self.x)
        paddle.jit.enable_to_static(False)
        return out

    @test_ast_only
    @test_legacy_and_pir
    def test_ast_to_func(self):
        self.setUp()
        self.assertIsInstance(self.out[0], (paddle.Tensor, core.eager.Tensor))
        self.assertIsInstance(self.out[1], int)


class TestDy2StIfElseRetInt2(TestDy2staticException):
    def setUp(self):
        self.x = np.random.random([5]).astype('float32')
        self.error = "Your if/else have different number of return value."
        self.dyfunc = dyfunc_ifelse_ret_int2


class TestDy2StIfElseRetInt3(TestDy2StIfElseRetInt1):
    def setUp(self):
        self.x = np.random.random([5]).astype('float32')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int3)
        self.out = self.get_dy2stat_out()

    @test_ast_only
    @test_legacy_and_pir
    def test_ast_to_func(self):
        self.setUp()
        self.assertIsInstance(self.out, (paddle.Tensor, core.eager.Tensor))


class TestDy2StIfElseRetInt4(TestDy2StIfElseRetInt1):
    def setUp(self):
        self.x = np.random.random([5]).astype('float32')
        self.dyfunc = paddle.jit.to_static(dyfunc_ifelse_ret_int4)

    @test_ast_only
    @test_legacy_and_pir
    def test_ast_to_func(self):
        paddle.jit.enable_to_static(True)
        with self.assertRaises(Dygraph2StaticException):
            static_func = paddle.jit.to_static(self.dyfunc)
            out = static_func(self.x)
        # Why need set `_in_to_static_mode_` here?
        # In Dy2St we use `with _to_static_mode_guard_()` to indicate
        # that the code block is under @to_static, but in this UT
        # an exception is thrown during Dy2St, making the `_in_to_static_mode_`
        # a wrong value. So We need set `_in_to_static_mode_` to False manually.
        paddle.base.dygraph.base.global_var._in_to_static_mode_ = False
        paddle.jit.enable_to_static(False)


class IfElseNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.param = self.create_parameter(
            shape=[3, 2], dtype='float32', is_bias=False
        )

    @paddle.jit.to_static
    def forward(self, a, b, c):
        a = paddle.matmul(a, self.param)
        a = paddle.reshape(a, (2, 4))
        cond = paddle.to_tensor([10])
        if cond == 10:
            a_argmax = a.argmax(axis=-1)
            b = b + self.param
        else:
            print(c)
        return b


class TestDy2StIfElseBackward(Dy2StTestBase):
    # TODO(zhangbo): open pir test (IfOp grad execution not yet supported)
    def test_run_backward(self):
        a = paddle.randn((4, 3), dtype='float32')
        a.stop_gradient = False
        b = paddle.to_tensor([10]).astype('float32')
        b.stop_gradient = False
        c = paddle.to_tensor([2])
        c.stop_gradient = False

        net = IfElseNet()
        net.train()
        out = net(a, b, c)
        out.backward()
        np.testing.assert_allclose(
            (b + net.param).numpy(), out.numpy(), rtol=1e-05
        )


if __name__ == '__main__':
    unittest.main()
