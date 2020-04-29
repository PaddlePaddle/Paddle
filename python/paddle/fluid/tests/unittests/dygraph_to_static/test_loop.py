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

import gast
import inspect
import numpy as np
import paddle.fluid as fluid
import unittest

from paddle.fluid.dygraph.dygraph_to_static.loop_transformer import NameVisitor
from paddle.fluid.dygraph.jit import declarative

SEED = 2020
np.random.seed(SEED)


def while_loop_dyfunc(x):
    i = fluid.dygraph.to_variable(x)
    # Use `to_variable` so that static analysis can analyze the type of X is Tensor
    x = fluid.dygraph.to_variable(
        x)  # TODO(liym27): Delete it if the type of parameter x can be resolved
    while x < 10:
        i = i + x
        x = x + 1
    return i


def while_loop_dyfun_with_conflict_var(x):
    i = fluid.dygraph.to_variable(x)
    # Use `to_variable` so that static analysis can analyze the type of X is Tensor
    x = fluid.dygraph.to_variable(
        x)  # TODO(liym27): Delete it if the type of parameter x can be resolved

    def relu(y):
        # 'y' is not visible outside the scope.
        return fluid.layers.relu(y)

    while x < 10:
        # If a tmp variable is created which has same name
        # with a argument in function, it should not be
        # included in the loop_vars.
        add_fn = lambda x, y: x + y
        i = add_fn(i, x)
        x = x + 1
    return i


def while_loop_dyfunc_with_none(x):
    i = fluid.dygraph.to_variable(x)\
        if x is not None \
        else fluid.dygraph.to_variable(x+1)
    # Use `to_variable` so that static analysis can analyze the type of X is Tensor
    x = fluid.dygraph.to_variable(
        x)  # TODO(liym27): Delete it if the type of parameter x can be resolved
    flag = 1
    while x < 10:
        i = i + x if flag is not None else x + i
        x = x + 1
    return i


def for_loop_dyfunc(max_len):
    for i in range(max_len):
        ret = fluid.layers.zeros(shape=[1], dtype='float32')
        fluid.layers.increment(ret, value=2.0, in_place=True)
    return ret


def while_loop_bool_op(x):
    i = fluid.dygraph.to_variable(x)

    # Use `to_variable` so that static analysis can analyze the type of X is Tensor
    x = fluid.dygraph.to_variable(
        x)  # TODO(liym27): Delete it if the type of parameter x can be resolved
    while (x >= 0 and x < 10) or x <= -1 or x < -3 or (x < -7 or x < -5):
        i = i + x
        x = x + 1
    return i


def while_loop_class_var(x):
    class Foo(object):
        def __init__(self):
            self.a = 3
            self.b = 4
            self.c = 5

    foo = Foo()
    i = fluid.dygraph.to_variable(x)
    while i < 10:
        foo.b = fluid.layers.zeros(shape=[1], dtype='float32')
        foo.c = foo.b + foo.a
        i += 1
    return foo.c


def for_loop_class_var(max_len):
    class Foo(object):
        def __init__(self):
            self.a = 3
            self.b = 4
            self.c = 5

    foo = Foo()

    # Use `to_variable` so that static analysis can analyze the type of X is Tensor
    # TODO(liym27): Delete it if the type of parameter x can be resolved
    max_len = fluid.layers.fill_constant(
        shape=[1], value=max_len, dtype="int32")
    for i in range(max_len):
        foo.b = fluid.layers.zeros(shape=[1], dtype='float32')
        foo.c = foo.b + foo.a
    return foo.c


def var_create_in_for_loop(max_len):
    for i in range(max_len):
        ret = fluid.layers.zeros(shape=[3, 4, 5], dtype='float64')
    return ret


class TestNameVisitor(unittest.TestCase):
    def setUp(self):
        self.loop_funcs = [
            while_loop_dyfunc, for_loop_dyfunc, while_loop_dyfunc_with_none
        ]
        self.loop_var_names = [
            set(["i", "x"]), set(["i", "ret", "max_len"]), set(["i", "x"])
        ]
        self.create_var_names = [set(), set(["ret"]), set()]

    def test_loop_vars(self):
        for i in range(len(self.loop_funcs)):
            func = self.loop_funcs[i]
            test_func = inspect.getsource(func)
            gast_root = gast.parse(test_func)
            name_visitor = NameVisitor(gast_root)
            for node in gast.walk(gast_root):
                if isinstance(node, (gast.While, gast.For)):
                    loop_var_names, create_var_names = name_visitor.get_loop_var_names(
                        node)
                    self.assertEqual(loop_var_names, self.loop_var_names[i])
                    self.assertEqual(create_var_names, self.create_var_names[i])


class TestTransformWhileLoop(unittest.TestCase):
    def setUp(self):
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.x = np.zeros(shape=(1), dtype=np.int32)
        self._init_dyfunc()

    def _init_dyfunc(self):
        self.dyfunc = while_loop_dyfunc

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        with fluid.dygraph.guard(self.place):
            if to_static:
                ret = declarative(self.dyfunc)(self.x)
            else:
                ret = self.dyfunc(self.x)
            return ret.numpy()

    def test_ast_to_func(self):
        static_numpy = self._run_static()
        dygraph_numpy = self._run_dygraph()
        self.assertTrue(np.allclose(dygraph_numpy, static_numpy))


class TestTransformWhileLoopWithConflicVar(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_dyfun_with_conflict_var


class TestTransformWhileLoopWithNone(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_dyfunc_with_none


class TestWhileLoopBoolOp(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_bool_op


class TestWhileLoopClassVar(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_class_var


class TestTransformForLoop(unittest.TestCase):
    def setUp(self):
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.len = 100
        self._init_dyfunc()

    def _init_dyfunc(self):
        self.dyfunc = for_loop_dyfunc

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        with fluid.dygraph.guard(self.place):
            if to_static:
                ret = declarative(self.dyfunc)(self.len)
            else:
                ret = self.dyfunc(self.len)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue(np.allclose(self._run_dygraph(), self._run_static()))


class TestClassVarInForLoop(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = for_loop_class_var


class TestVarCreateInForLoop(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = var_create_in_for_loop


if __name__ == '__main__':
    unittest.main()
