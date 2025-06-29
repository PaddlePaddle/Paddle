# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
import types
import unittest

import numpy as np
from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils.exceptions import InnerError

sot_test_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(f'{sot_test_dir}/../dygraph_to_static'))

from dygraph_to_static_utils import Dy2StTestBase, test_default_mode_only


# ---------------------- test single inheritance case ----------------------
class A:
    @check_no_breakgraph
    def add2(self, x):
        return x + 2

    @check_no_breakgraph
    def add3(self, x):
        return x + 3


class B(A):
    @check_no_breakgraph
    def test_super_no_args_add2(self, x):
        y = super().add2(x)
        return y

    @check_no_breakgraph
    def test_super_with_args_add3(self, x):
        y = super(B, self).add3(x)  # noqa: UP008
        return y

    @check_no_breakgraph
    def test_super_both_add5(self, x):
        return super().add2(x) + super(B, self).add3(x)  # noqa: UP008

    @check_no_breakgraph
    def test_self_name_me(me, x):
        # Test case where the instance is referred to as 'me' instead of 'self'
        return super(B, me).add2(x)  # noqa: UP008

    @check_no_breakgraph
    def test_self_name_this(this, x):
        # Test case where the instance is referred to as 'this' instead of 'self'
        return super(B, this).add2(x)  # noqa: UP008


class TestSingleInheritance(TestCaseBase):
    def test_super_no_args(self):
        self.assert_results(B().test_super_no_args_add2, paddle.to_tensor(33))

    def test_super_with_args(self):
        self.assert_results(B().test_super_with_args_add3, paddle.to_tensor(33))

    def test_super_both(self):
        self.assert_results(B().test_super_both_add5, paddle.to_tensor(33))

    def test_super_self_name(self):
        self.assert_results(B().test_self_name_me, paddle.to_tensor(33))
        self.assert_results(B().test_self_name_this, paddle.to_tensor(33))

    def test_guard_run(self):  # test guard
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(
                B().test_super_no_args_add2, paddle.to_tensor(1)
            )
            self.assert_results(
                B().test_super_no_args_add2, paddle.to_tensor(2)
            )
            self.assertEqual(ctx.translate_count, 1)

        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(
                B().test_super_no_args_add2, paddle.to_tensor(1)
            )
            self.assert_results(
                B().test_super_no_args_add2, paddle.to_tensor(2)
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                B().test_super_with_args_add3, paddle.to_tensor(3)
            )
            self.assert_results(
                B().test_super_with_args_add3, paddle.to_tensor(4)
            )
            self.assertEqual(ctx.translate_count, 2)

        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(B().test_super_both_add5, paddle.to_tensor(5))
            self.assert_results(B().test_super_both_add5, paddle.to_tensor(6))
            self.assertEqual(ctx.translate_count, 1)


# ---------------------- test multiple inheritance case ----------------------
class X:
    @check_no_breakgraph
    def addx(self, x):
        return 1 + x


class Y:
    @check_no_breakgraph
    def addx(self, x):
        return 2 + x


class Z(X, Y):
    @check_no_breakgraph
    def addx(self, x):
        return super().addx(x) + 3 + x


class P(Y):
    @check_no_breakgraph
    def addx(self, x):
        return super(P, self).addx(x) + 4 + x  # noqa: UP008


class Q(Z, P):
    @check_no_breakgraph
    def addx(self, x):
        return super(Q, self).addx(x) + 5 + x  # noqa: UP008

    @check_no_breakgraph
    def addxP(self, x):
        return super(P, self).addx(x) + 5 + x

    @check_no_breakgraph
    def addxZ(self, x):
        return super(Z, self).addx(x) + 5 + x


# Inheritance diagram
# X     Y
#  \   / \
#   \ /   \
#    Z     P
#     \   /
#      \ /
#       Q
class TestMultipleInheritance(TestCaseBase):
    def test_with_args(self):
        x = paddle.to_tensor([1.0])
        self.assert_results(Q().addx, x)
        self.assert_results(Q().addxP, x)
        self.assert_results(Q().addxZ, x)

    def test_guard_run(self):  # test guard
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(Q().addx, paddle.to_tensor(1))
            self.assert_results(Q().addx, paddle.to_tensor(2))
            self.assert_results(Q().addx, paddle.to_tensor(3))
            self.assertEqual(ctx.translate_count, 1)

        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(Q().addxP, paddle.to_tensor(4))
            self.assert_results(Q().addxP, paddle.to_tensor(5))
            self.assert_results(Q().addxP, paddle.to_tensor(6))
            self.assertEqual(ctx.translate_count, 1)

        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(Q().addxZ, paddle.to_tensor(7))
            self.assert_results(Q().addxZ, paddle.to_tensor(8))
            self.assert_results(Q().addxZ, paddle.to_tensor(9))
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(Q().addxP, paddle.to_tensor(4))
            self.assert_results(Q().addxP, paddle.to_tensor(5))
            self.assert_results(Q().addxP, paddle.to_tensor(6))
            self.assertEqual(ctx.translate_count, 2)


# ---------------------- test `super()` as input ----------------------
class ClassSuperAsInput:
    @check_no_breakgraph
    def fn(self, x):
        return x + 1


@check_no_breakgraph
def super_as_input(spr):
    return spr.fn(2)


class TestSuperAsInput1(TestCaseBase, ClassSuperAsInput):
    def test_super_as_input(self):
        self.assert_results(super_as_input, super())

    def test_guard_run(self):  # test guard
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(super_as_input, super())
            self.assert_results(super_as_input, super())
            self.assert_results(super_as_input, super())
            self.assertEqual(ctx.translate_count, 1)


class ClassSuperAsInputA:
    @check_no_breakgraph
    def test_fn(self, x):
        return x + 1


class ClassSuperAsInputB(ClassSuperAsInputA):
    @check_no_breakgraph
    def test_fn(self, x):
        return x + 4


class ClassSuperAsInputC(ClassSuperAsInputB):
    @check_no_breakgraph
    def super_as_input(self, spr, x):
        return spr.test_fn(x)

    @check_no_breakgraph
    def test_super_as_input(self, x, cls):
        return self.super_as_input(super(cls, self), x)


class TestSuperAsInput2(TestCaseBase):
    def test_super_as_input(self):
        x = paddle.to_tensor(3)
        self.assert_results(
            ClassSuperAsInputC().test_super_as_input, x, ClassSuperAsInputC
        )
        self.assert_results(
            ClassSuperAsInputC().test_super_as_input, x, ClassSuperAsInputB
        )

    def test_guard_run(self):  # test guard
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x = paddle.to_tensor(3)
            self.assert_results(
                ClassSuperAsInputC().test_super_as_input, x, ClassSuperAsInputC
            )
            self.assert_results(
                ClassSuperAsInputC().test_super_as_input, x, ClassSuperAsInputC
            )
            self.assertEqual(ctx.translate_count, 1)
            x = paddle.to_tensor(4)
            self.assert_results(
                ClassSuperAsInputC().test_super_as_input, x, ClassSuperAsInputB
            )
            self.assert_results(
                ClassSuperAsInputC().test_super_as_input, x, ClassSuperAsInputB
            )
            self.assertEqual(ctx.translate_count, 2)


# ---------------------- test case which has no functions ----------------------
class ClassWithAttributionA:
    a = 1


class ClassWithAttributionB(ClassWithAttributionA):
    a = 111


class ClassWithAttributionC(ClassWithAttributionB):
    @check_no_breakgraph
    def foo(self, x):
        return (
            super().a + x,
            super(ClassWithAttributionC, self).a + x,  # noqa: UP008
            super(ClassWithAttributionB, self).a + x,
        )


class TestSuperAttr(TestCaseBase):
    def test_attr_equal(self):
        x = paddle.to_tensor([4.0])
        self.assert_results(ClassWithAttributionC().foo, x)

    def test_guard_run(self):  # test guard
        x = paddle.to_tensor([4.0])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(ClassWithAttributionC().foo, x)
            self.assert_results(ClassWithAttributionC().foo, x)
            self.assert_results(ClassWithAttributionC().foo, x)
            self.assertEqual(ctx.translate_count, 1)


# ---------------------- test case which has fake super ----------------------
class Toy:
    @check_no_breakgraph
    def get(self, x):
        return x + 1


class FakeSuperBase:
    @check_no_breakgraph
    def put(self, x):
        return x + 1


class FakeSuperClass(FakeSuperBase):
    @check_no_breakgraph
    def fake_super_function(self, x):
        return super(1, 2).get(x)

    def super_function_as_input(self, fn, x):
        return fn().put(x)


# We create a fake `super` and inject it to `__globals__` of the function
new_globals = FakeSuperClass.fake_super_function.__globals__.copy()
new_globals["super"] = lambda x, y: Toy()

FakeSuperClass.fake_super_function = types.FunctionType(
    FakeSuperClass.fake_super_function.__code__,
    new_globals,
    name=FakeSuperClass.fake_super_function.__name__,
    argdefs=FakeSuperClass.fake_super_function.__defaults__,
    closure=FakeSuperClass.fake_super_function.__closure__,
)


class TestCustomSuper(TestCaseBase):
    def test_fake_super(self):
        self.assert_results(
            FakeSuperClass().fake_super_function, paddle.to_tensor(3.0)
        )

    def test_super_function_as_input(self):
        if sys.version_info >= (3, 13):
            self.assert_exceptions(
                RuntimeError,
                r"super\(\): __class__ cell not found",
                FakeSuperClass().super_function_as_input,
                super,
                paddle.to_tensor(3.0),
            )
        if sys.version_info < (3, 13):
            self.assert_exceptions(
                InnerError,
                "KeyError: '__class__'",
                FakeSuperClass().super_function_as_input,
                super,
                paddle.to_tensor(3.0),
            )


# ------ test SuperVariable setattr + getattr ------
class LrClassBase:
    def __init__(self, last_lr, last_epoch):
        self.last_lr = last_lr
        self.last_epoch = last_epoch
        self.step()

    def step(self):
        self.last_epoch += 1
        self.last_lr = self.get_lr()

    def get_lr(self):
        return self.last_lr + 0.0001

    def __call__(self) -> float:
        return self.last_lr


class LrClassSub(LrClassBase):
    def __init__(self, **kwargs):
        super().__init__(
            kwargs.get("last_lr", 0.01), kwargs.get("last_epoch", 0)
        )


class TestSuperSetattrGetattr(Dy2StTestBase):
    def setUp(self):
        def dyfunc(lr_decay):
            lr = lr_decay()
            return paddle.to_tensor(lr)

        lr_decay = LrClassSub(
            base_lr=0.1, verbose=True, last_lr=0.3, last_epoch=3
        )
        self.dygraph_func = lambda: dyfunc(lr_decay)

    def get_dygraph_output(self):
        res = self.dygraph_func()
        return res

    def get_static_output(self):
        static_res = paddle.jit.to_static(self.dygraph_func)()
        return static_res

    @test_default_mode_only
    def test_transformed_static_result(self):
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
