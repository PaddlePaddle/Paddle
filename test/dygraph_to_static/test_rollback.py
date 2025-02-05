# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
    test_ast_only,
)

import paddle
from paddle.jit.dy2static.program_translator import StaticFunction
from paddle.jit.dy2static.utils import func_to_source_code


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.sub = SubNet()

    def forward(self, x):
        x = self.sub(x)
        x = foo(x)
        out = self.sub.bar(x)
        return out

    def infer(self, x):
        x = self.sub.bar(x)
        out = foo(x)
        return out


class SubNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, flag=True):
        if flag:
            out = x + 1
        else:
            out = x - 1
        return out

    def bar(self, x, flag=True):
        if flag:
            out = x + 2
        else:
            out = x - 2
        return out


def foo(x, flag=False):
    if flag:
        out = x * 2.0
    else:
        out = x / 2.0

    return out


class TestRollBackPlainFunction(Dy2StTestBase):
    def test_plain_func(self):
        paddle.set_device("cpu")
        st_foo = paddle.jit.to_static(foo)
        x = paddle.randn([3, 4])
        st_out = st_foo(x)

        self.assertTrue(isinstance(st_foo, StaticFunction))

        st_foo = st_foo.rollback()
        dy_out = st_foo(x)

        self.assertTrue(func_to_source_code(foo) == func_to_source_code(st_foo))
        np.testing.assert_array_equal(st_out.numpy(), dy_out.numpy())


class TestRollBackNet(Dy2StTestBase):
    @test_ast_only
    def test_net(self):
        paddle.set_device("cpu")
        net = paddle.jit.to_static(Net())
        x = paddle.randn([3, 4])
        st_fwd_out = net(x)

        # forward function is inplacly converted.
        self.assertTrue(isinstance(net.forward, StaticFunction))
        self.assertTrue("true_fn" in func_to_source_code(net.sub.forward))
        # other non-forward function is not inplacly converted.
        self.assertFalse("true_fn" in func_to_source_code(net.sub.bar))

        net.infer = paddle.jit.to_static(net.infer)
        st_infer_out = net.infer(x)
        self.assertTrue(isinstance(net.infer, StaticFunction))
        self.assertFalse("true_fn" in func_to_source_code(net.sub.bar))

        # rollback forward into original dygraph method
        net.forward = net.forward.rollback()
        self.assertFalse(isinstance(net.forward, StaticFunction))
        self.assertFalse("true_fn" in func_to_source_code(net.sub.forward))
        dy_fwd_out = net(x)
        np.testing.assert_array_equal(st_fwd_out.numpy(), dy_fwd_out.numpy())

        # rollback infer into original dygraph method
        net.infer.rollback()
        self.assertFalse(isinstance(net.infer, StaticFunction))
        self.assertFalse("true_fn" in func_to_source_code(net.sub.forward))
        dy_infer_out = net.infer(x)
        np.testing.assert_array_equal(
            st_infer_out.numpy(), dy_infer_out.numpy()
        )


class FuncRollback(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x + 1

    @paddle.jit.to_static(full_graph=True)
    def func(self, x):
        return x + 2


class TestRollBackNotForward(Dy2StTestBase):
    @test_ast_only
    def test_rollback(self):
        x = paddle.zeros([2, 2])
        net = FuncRollback()
        out = net.func(x)
        net.func.rollback()
        self.assertTrue(not isinstance(net.func, StaticFunction))


class FuncRollbackWithPatchedFunction(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x + 1


def patched_fn(self, x):
    return x + 2


FuncRollbackWithPatchedFunction.forward = patched_fn


class TestRollBackWithPatchedFunction(Dy2StTestBase):
    @test_ast_only
    def test_rollback(self):
        x = paddle.zeros([2, 2])
        net = FuncRollbackWithPatchedFunction()
        dy_out = net(x)
        static_net = paddle.jit.to_static(net, full_graph=True)
        st_out = static_net(x)
        static_net.forward.rollback()
        dy_out_rollback = net(x)

        self.assertTrue(not isinstance(net.forward, StaticFunction))

        np.testing.assert_array_equal(dy_out.numpy(), st_out.numpy())
        np.testing.assert_array_equal(dy_out.numpy(), dy_out_rollback.numpy())


if __name__ == "__main__":
    unittest.main()
