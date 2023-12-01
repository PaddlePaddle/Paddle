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

import inspect
import tempfile
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_default_and_pir,
    test_default_mode_only,
)

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.jit.dy2static.loop_transformer import NameVisitor
from paddle.utils import gast

SEED = 2020
np.random.seed(SEED)


def while_loop_dyfunc(x):
    i = base.dygraph.to_variable(x)
    while x < 10:
        i = i + x
        x = x + 1
    return i


def while_loop_dyfunc_without_tensor(x):
    a = 1
    # There are no tensors in the while condition, which means it's a plain while in python,
    # so it wont't be transformed to `while_loop` op.
    while not a > 4 and a > 0:
        x = x + 1
        a = a + 1

    return x


def while_loop_dyfun_with_conflict_var(x):
    i = base.dygraph.to_variable(x)

    def relu(y):
        # 'y' is not visible outside the scope.
        return F.relu(y)

    while x < 10:
        # If a tmp variable is created which has same name
        # with a argument in function, it should not be
        # included in the loop_vars.
        add_fn = lambda x, y: x + y
        i = add_fn(i, x)
        x = x + 1
    return i


def while_loop_dyfunc_with_none(x):
    i = (
        base.dygraph.to_variable(x)
        if x is not None
        else base.dygraph.to_variable(x + 1)
    )
    # Use `to_variable` so that static analysis can analyze the type of X is Tensor
    x = base.dygraph.to_variable(
        x
    )  # TODO(liym27): Delete it if the type of parameter x can be resolved
    flag = 1
    while x < 10:
        i = i + x if flag is not None else x + i
        x = x + 1
    return i


def for_loop_dyfunc(max_len):
    for i in range(max_len):
        ret = paddle.zeros(shape=[1], dtype='float32')
        paddle.increment(ret, value=2.0)
    return ret


def for_loop_dyfunc2(max_len):
    # Test case: a variable is used and created in loop, but used before created
    x = paddle.tensor.fill_constant(shape=[1, 2], dtype="int32", value=1)

    for i in range(max_len):
        if i > 1:
            s = a
        a = 1
        q, _ = x.shape  # test var x.shape only used but not created in loop

    ret = paddle.tensor.fill_constant(shape=[1], dtype="int32", value=s + q)
    return ret


def for_loop_dyfunc3(max_len):
    ret = paddle.zeros(shape=[1], dtype='float32')
    for i in range(1, 10, 2):
        paddle.increment(ret, value=2.0)
    return ret


def for_loop_dyfunc4(max_len):
    ret = paddle.zeros(shape=[1], dtype='float32')
    for i in range(10, 1, -2):
        paddle.increment(ret, value=2.0)
    return ret


def for_loop_dyfunc_not_support(max_len):
    ret = paddle.zeros(shape=[1], dtype='float32')
    a = -2
    for i in range(10, 1, a):
        paddle.increment(ret, value=2.0)
    return ret


def for_break_single_return(max_len):
    x = 0
    for i in range(3):
        if i == 2:
            break
        x += 1
    return x


def while_loop_bool_op(x):
    i = base.dygraph.to_variable(x)

    while x <= -1 or x < -3 or (x < -7 or x < -5) or (x >= 0 and x < 10):
        i = i + x
        x = x + 1
    return i


def while_loop_bool_op2(x):
    i = base.dygraph.to_variable(x)
    a = 1

    # In the while condition, there are both Paddle Variable and non-Variable.
    while x < 10 and (a < 4 or a > 0) or a < -1 or not x > -1:
        i = i + x
        x = x + 1
        a = a + 1
    return i


def while_loop_class_var(x):
    class Foo:
        def __init__(self):
            self.a = 3
            self.b = 4
            self.c = 5

    foo = Foo()
    i = base.dygraph.to_variable(x)
    while i < 10:
        foo.b = paddle.zeros(shape=[1], dtype='float32')
        foo.c = foo.b + foo.a
        i += 1
    return foo.c


def loop_var_contains_property(x):
    a = paddle.zeros(shape=[1], dtype='float32')
    i = paddle.to_tensor(x)
    s = i.shape
    while i < 10 and s[0] >= 1:
        a += i.shape[0]
        i += 1
    return a


def for_loop_class_var(max_len):
    class Foo:
        def __init__(self):
            self.a = 3
            self.b = 4
            self.c = 5

    foo = Foo()

    # Use `to_variable` so that static analysis can analyze the type of X is Tensor
    max_len = paddle.tensor.fill_constant(
        shape=[1], value=max_len, dtype="int32"
    )

    for i in range(max_len):
        foo.b = paddle.zeros(shape=[1], dtype='float32')
        foo.c = foo.b + foo.a
    return foo.c


def var_create_in_for_loop(max_len):
    for i in range(max_len):
        ret = paddle.zeros(shape=[3, 4, 5], dtype='float64')
    return ret


def nested_for_loop_dyfunc():
    two = paddle.tensor.fill_constant(shape=[1], value=2, dtype="int32")
    three = paddle.tensor.fill_constant(shape=[1], value=3, dtype="int32")
    for j in range(two):
        for i in range(10):
            a = 2 + j

    for i in range(three):
        b = paddle.zeros(shape=[1], dtype='float32')

    return b


def for_loop_dufunc_with_listcomp(array):
    a = 1
    for j in range(array):
        res = [x + a for x in array]
        res = [i for i in array]
        x = 1
    b = [i for i in array]
    print(x)
    return res


class TestNameVisitor(Dy2StTestBase):
    def setUp(self):
        self.loop_funcs = [
            while_loop_dyfunc,
            for_loop_dyfunc,
            while_loop_dyfunc_with_none,
            for_loop_dufunc_with_listcomp,
        ]
        self.loop_var_names = [
            {"i", "x"},
            {"i", "ret", "max_len"},
            {"i", "x"},
            {"j", "array", "res", "x"},
        ]
        self.create_var_names = [set(), {"ret"}, set(), {"res", "x"}]

        self.nested_for_loop_func = nested_for_loop_dyfunc

    @test_default_and_pir
    def test_loop_vars(self):
        for i in range(len(self.loop_funcs)):
            func = self.loop_funcs[i]
            test_func = inspect.getsource(func)
            gast_root = gast.parse(test_func)
            name_visitor = NameVisitor(gast_root)
            for node in gast.walk(gast_root):
                if isinstance(node, (gast.While, gast.For)):
                    (
                        loop_var_names,
                        create_var_names,
                    ) = name_visitor.get_loop_var_names(node)
                    self.assertEqual(loop_var_names, self.loop_var_names[i])
                    self.assertEqual(create_var_names, self.create_var_names[i])

    @test_default_and_pir
    def test_nested_loop_vars(self):
        func = self.nested_for_loop_func
        test_func = inspect.getsource(func)
        gast_root = gast.parse(test_func)
        name_visitor = NameVisitor(gast_root)

        self.loop_var_names = [
            {"j", "two"},
            {"i", "three", "b"},
            {"i"},
        ]
        self.create_var_names = [set(), {"b"}, set()]

        i = 0
        for node in gast.walk(gast_root):
            if isinstance(node, (gast.While, gast.For)):
                (
                    loop_var_names,
                    create_var_names,
                ) = name_visitor.get_loop_var_names(node)
                self.assertEqual(
                    loop_var_names,
                    self.loop_var_names[i],
                    msg="loop_var_names : {}, \nexpected loop_var_names : {}".format(
                        loop_var_names, self.loop_var_names[i]
                    ),
                )
                self.assertEqual(
                    create_var_names,
                    self.create_var_names[i],
                    msg="i = {}\ncreate_var_names : {}, \nexpected create_var_names : {}".format(
                        i, create_var_names, self.create_var_names[i]
                    ),
                )
                i += 1


class TestTransformWhileLoop(Dy2StTestBase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.x = np.zeros(shape=(1), dtype=np.int32)
        self._init_dyfunc()

    def _init_dyfunc(self):
        self.dyfunc = while_loop_dyfunc

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        # Set the input of dyfunc to Tensor
        tensor_x = base.dygraph.to_variable(self.x, zero_copy=False)
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(tensor_x)
        else:
            ret = self.dyfunc(tensor_x)
        if hasattr(ret, "numpy"):
            return ret.numpy()
        else:
            return ret

    @test_default_mode_only
    def test_ast_to_func(self):
        static_numpy = self._run_static()
        dygraph_numpy = self._run_dygraph()
        print(static_numpy, dygraph_numpy)
        np.testing.assert_allclose(dygraph_numpy, static_numpy, rtol=1e-05)


class TestTransformWhileLoopWithoutTensor(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_dyfunc_without_tensor


class TestTransformWhileLoopWithConflicVar(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_dyfun_with_conflict_var


class TestTransformWhileLoopWithNone(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_dyfunc_with_none


class TestForBreakSingleReturn(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = for_break_single_return


class TestWhileLoopBoolOp(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_bool_op


class TestWhileLoopBoolOp2(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_bool_op2


class TestWhileLoopClassVar(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = while_loop_class_var


class TestLoopVarContainsProperty(TestTransformWhileLoop):
    def _init_dyfunc(self):
        self.dyfunc = loop_var_contains_property


class TestTransformForLoop(Dy2StTestBase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.len = 100
        self._init_dyfunc()

    def _init_dyfunc(self):
        self.dyfunc = for_loop_dyfunc

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        if to_static:
            ret = paddle.jit.to_static(self.dyfunc)(self.len)
        else:
            ret = self.dyfunc(self.len)
        return ret.numpy()

    @test_default_mode_only
    def test_ast_to_func(self):
        np.testing.assert_allclose(
            self._run_dygraph(), self._run_static(), rtol=1e-05
        )


class TestTransformForLoop2(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = for_loop_dyfunc2


class TestTransformForLoop3(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = for_loop_dyfunc3


class TestTransformForLoop4(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = for_loop_dyfunc4


class TestClassVarInForLoop(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = for_loop_class_var


class TestVarCreateInForLoop(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = var_create_in_for_loop


class TestErrorInForLoop(TestTransformForLoop):
    def _init_dyfunc(self):
        self.dyfunc = for_loop_dyfunc_not_support


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        self.layer_dict = paddle.nn.LayerDict(
            {
                "conv1": paddle.nn.Conv2D(3, 3, 1),
                "conv2": paddle.nn.Conv2D(3, 3, 1),
                "conv3": paddle.nn.Conv2D(3, 3, 1),
            }
        )

    def forward(self, x):
        out = 0
        for layer_name in self.layer_dict:
            out += self.layer_dict[layer_name](x)
        return out


class TestForLoopMeetDict(Dy2StTestBase):
    @test_default_mode_only
    def test_start(self):
        net = Net()
        model = paddle.jit.to_static(
            net,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 3, 224, 224], dtype='float32'
                )
            ],
        )
        temp_dir = tempfile.TemporaryDirectory()
        paddle.jit.save(model, temp_dir.name)
        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
