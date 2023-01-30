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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.fluid as fluid


def add_fn(x):
    x = x + 1
    return x


def loss_fn(x, lable):
<<<<<<< HEAD
    loss = paddle.nn.functional.cross_entropy(
        x, lable, reduction='none', use_softmax=False
    )
=======
    loss = fluid.layers.cross_entropy(x, lable)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return loss


def dyfunc_empty_nonlocal(x):
    flag = True
    if flag:
        print("It's a test for empty nonlocal stmt")

    if paddle.mean(x) < 0:
        x + 1

    out = x * 2
    return out


def dyfunc_with_if_else(x_v, label=None):
    if paddle.mean(x_v).numpy()[0] > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    # plain if in python
    if label is not None:
<<<<<<< HEAD
        loss = paddle.nn.functional.cross_entropy(
            x_v, label, reduction='none', use_softmax=False
        )
=======
        loss = fluid.layers.cross_entropy(x_v, label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return loss
    return x_v


def dyfunc_with_if_else2(x, col=100):
    row = 0
    if abs(col) > x.shape[-1]:
        # TODO: Don't support return non-Tensor in Tensor-dependent `if` stament currently.
        #  `x` is Tensor, `col` is not Tensor, and `col` is the return value of `true_fn` after transformed.
        # col = -1
        col = fluid.layers.fill_constant(shape=[1], value=-1, dtype="int64")
<<<<<<< HEAD
    if paddle.mean(x).numpy()[0] > x.numpy()[row][col]:
        y = paddle.nn.functional.relu(x)
    else:
        x_pow = paddle.pow(x, 2)
        y = paddle.tanh(x_pow)
=======
    if fluid.layers.reduce_mean(x).numpy()[0] > x.numpy()[row][col]:
        y = fluid.layers.relu(x)
    else:
        x_pow = fluid.layers.pow(x, 2)
        y = fluid.layers.tanh(x_pow)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return y


def dyfunc_with_if_else3(x):
    # Create new var in parent scope, return it in true_fn and false_fn.
    # The var is created only in one of If.body or If.orelse node, and it used as gast.Load firstly after gast.If node.
    # The transformed code:
    """
    q = paddle.jit.dy2static.UndefinedVar('q')
    z = paddle.jit.dy2static.UndefinedVar('z')

    def true_fn_0(q, x, y):
        x = x + 1
        z = x + 2
        q = x + 3
        return q, x, y, z

    def false_fn_0(q, x, y):
        y = y + 1
        z = x - 2
        m = x + 2
        n = x + 3
        return q, x, y, z
<<<<<<< HEAD
    q, x, y, z = paddle.static.nn.cond(paddle.mean(x)[0] < 5, lambda :
=======
    q, x, y, z = fluid.layers.cond(paddle.mean(x)[0] < 5, lambda :
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.jit.dy2static.convert_call(true_fn_0)(q, x, y),
        lambda : paddle.jit.dy2static.convert_call(false_fn_0)(q,
        x, y))
    """
    y = x + 1
    # NOTE: x_v[0] < 5 is True
    if paddle.mean(x).numpy()[0] < 5:
        x = x + 1
        z = x + 2
        q = x + 3
    else:
        y = y + 1
        z = x - 2
        m = x + 2
        n = x + 3

    q = q + 1
    n = q + 2
    x = n
    return x


def dyfunc_with_if_else_early_return1():
    x = paddle.to_tensor([10])
    if x == 0:
        a = paddle.zeros([2, 2])
        b = paddle.zeros([3, 3])
        return a, b
    a = paddle.zeros([2, 2]) + 1
    return a, None


def dyfunc_with_if_else_early_return2():
    x = paddle.to_tensor([10])
    if x == 0:
        a = paddle.zeros([2, 2])
        b = paddle.zeros([3, 3])
        return a, b
    elif x == 1:
        c = paddle.zeros([2, 2]) + 1
        d = paddle.zeros([3, 3]) + 1
        return c, d
    e = paddle.zeros([2, 2]) + 3
    return e, None


def dyfunc_with_if_else_with_list_geneator(x):
    if 10 > 5:
        y = paddle.add_n(
<<<<<<< HEAD
            [paddle.full(shape=[2], fill_value=v) for v in range(5)]
        )
=======
            [paddle.full(shape=[2], fill_value=v) for v in range(5)])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    else:
        y = x
    return y


def nested_if_else(x_v):
    batch_size = 16
    feat_size = x_v.shape[-1]
    bias = fluid.layers.fill_constant([feat_size], dtype='float32', value=1)
    if x_v.shape[0] != batch_size:
        # TODO: Don't support return non-Tensor in Tensor-dependent `if` stament currently.
        #  `x_v.shape[0]` is not Tensor, and `batch_size` is the return value of `true_fn` after transformed.
        # col = -1
        # batch_size = x_v.shape[0]
<<<<<<< HEAD
        batch_size = paddle.shape(x_v)[0]
=======
        batch_size = fluid.layers.shape(x_v)[0]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # if tensor.shape is [1], now support to compare with numpy.
    if paddle.mean(x_v).numpy() < 0:
        y = x_v + bias
        w = fluid.layers.fill_constant([feat_size], dtype='float32', value=10)
        if y.numpy()[0] < 10:
            tmp = y * w
<<<<<<< HEAD
            y = paddle.nn.functional.relu(tmp)
            if paddle.mean(y).numpy()[0] < batch_size:
                y = paddle.abs(y)
            else:
                tmp = fluid.layers.fill_constant(
                    y.shape, dtype='float32', value=-1
                )
=======
            y = fluid.layers.relu(tmp)
            if paddle.mean(y).numpy()[0] < batch_size:
                y = fluid.layers.abs(y)
            else:
                tmp = fluid.layers.fill_constant(y.shape,
                                                 dtype='float32',
                                                 value=-1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                y = y - tmp
    else:
        y = x_v - bias
    return y


def nested_if_else_2(x):
<<<<<<< HEAD
    y = paddle.reshape(x, [-1, 1])
=======
    y = fluid.layers.reshape(x, [-1, 1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    b = 2
    if b < 1:
        # var `z` is not visible for outer scope
        z = y
    x_shape_0 = x.shape[0]
    if x_shape_0 < 1:
<<<<<<< HEAD
        if paddle.shape(y).numpy()[0] < 1:
            res = fluid.layers.fill_constant(
                value=2, shape=x.shape, dtype="int32"
            )
            # `z` is a new var here.
            z = y + 1
        else:
            res = fluid.layers.fill_constant(
                value=3, shape=x.shape, dtype="int32"
            )
=======
        if fluid.layers.shape(y).numpy()[0] < 1:
            res = fluid.layers.fill_constant(value=2,
                                             shape=x.shape,
                                             dtype="int32")
            # `z` is a new var here.
            z = y + 1
        else:
            res = fluid.layers.fill_constant(value=3,
                                             shape=x.shape,
                                             dtype="int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    else:
        res = x
    return res


def nested_if_else_3(x):
<<<<<<< HEAD
    y = paddle.reshape(x, [-1, 1])
=======
    y = fluid.layers.reshape(x, [-1, 1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    b = 2
    # var `z` is visible for func.body
    if b < 1:
        z = y
    else:
        z = x

    if b < 1:
        res = x
        # var `out` is only visible for current `if`
        if b > 1:
            out = x + 1
        else:
            out = x - 1
    else:
<<<<<<< HEAD
        y_shape = paddle.shape(y)
        if y_shape.numpy()[0] < 1:
            res = fluid.layers.fill_constant(
                value=2, shape=x.shape, dtype="int32"
            )
            # `z` is created in above code block.
            z = y + 1
        else:
            res = fluid.layers.fill_constant(
                value=3, shape=x.shape, dtype="int32"
            )
=======
        y_shape = fluid.layers.shape(y)
        if y_shape.numpy()[0] < 1:
            res = fluid.layers.fill_constant(value=2,
                                             shape=x.shape,
                                             dtype="int32")
            # `z` is created in above code block.
            z = y + 1
        else:
            res = fluid.layers.fill_constant(value=3,
                                             shape=x.shape,
                                             dtype="int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            # `out` is a new var.
            out = x + 1
    return res


class NetWithControlFlowIf(fluid.dygraph.Layer):
<<<<<<< HEAD
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = paddle.nn.Linear(
            in_features=hidden_dim,
            out_features=5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )
        self.alpha = 10.0
=======

    def __init__(self, hidden_dim=16):
        super(NetWithControlFlowIf, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = fluid.dygraph.Linear(
            input_dim=hidden_dim,
            output_dim=5,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.99)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5)))
        self.alpha = 10.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.constant_vars = {}

    @paddle.jit.to_static
    def forward(self, input):
        hidden_dim = input.shape[-1]
        if hidden_dim != self.hidden_dim:
            raise ValueError(
<<<<<<< HEAD
                "hidden_dim {} of input is not equal to FC.weight[0]: {}".format(
                    hidden_dim, self.hidden_dim
                )
            )

        self.constant_vars['bias'] = fluid.layers.fill_constant(
            [5], dtype='float32', value=1
        )
=======
                "hidden_dim {} of input is not equal to FC.weight[0]: {}".
                format(hidden_dim, self.hidden_dim))

        self.constant_vars['bias'] = fluid.layers.fill_constant([5],
                                                                dtype='float32',
                                                                value=1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # Control flow `if` statement
        fc_out = self.fc(input)
        if paddle.mean(fc_out).numpy()[0] < 0:
            y = fc_out + self.constant_vars['bias']
            self.constant_vars['w'] = fluid.layers.fill_constant(
<<<<<<< HEAD
                [5], dtype='float32', value=10
            )
=======
                [5], dtype='float32', value=10)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            if y.numpy()[0] < self.alpha:
                # Create new var, but is not used.
                x = 10
                tmp = y * self.constant_vars['w']
<<<<<<< HEAD
                y = paddle.nn.functional.relu(tmp)
=======
                y = fluid.layers.relu(tmp)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                # Nested `if/else`
                if y.numpy()[-1] < self.alpha:
                    # Modify variable of class
                    self.constant_vars['w'] = fluid.layers.fill_constant(
<<<<<<< HEAD
                        [hidden_dim], dtype='float32', value=9
                    )
                    y = paddle.abs(y)
                else:
                    tmp = fluid.layers.fill_constant(
                        y.shape, dtype='float32', value=-1
                    )
=======
                        [hidden_dim], dtype='float32', value=9)
                    y = fluid.layers.abs(y)
                else:
                    tmp = fluid.layers.fill_constant(y.shape,
                                                     dtype='float32',
                                                     value=-1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    y = y - tmp
        else:
            y = fc_out - self.constant_vars['bias']

        loss = paddle.mean(y)
        return loss


def if_with_and_or(x_v, label=None):
<<<<<<< HEAD
    batch_size = paddle.shape(x_v)
    if (
        x_v is not None
        and (paddle.mean(x_v).numpy()[0] > 0 or label is not None)
        and batch_size[0] > 1
        and True
    ):
=======
    batch_size = fluid.layers.shape(x_v)
    if x_v is not None and (paddle.mean(x_v).numpy()[0] > 0 or label
                            is not None) and batch_size[0] > 1 and True:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_v = x_v - 1
    else:
        x_v = x_v + 1

    if label is not None:
<<<<<<< HEAD
        loss = paddle.nn.functional.cross_entropy(
            x_v, label, reduction='none', use_softmax=False
        )
=======
        loss = fluid.layers.cross_entropy(x_v, label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return loss
    return x_v


def if_with_and_or_1(x, y=None):
<<<<<<< HEAD
    batch_size = paddle.shape(x)
=======
    batch_size = fluid.layers.shape(x)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if batch_size[0] > 1 and y is not None:
        x = x + 1
    if y is not None or batch_size[0] > 1:
        x = x - 1
    return x


def if_with_and_or_2(x, y=None):
<<<<<<< HEAD
    batch_size = paddle.shape(x)
=======
    batch_size = fluid.layers.shape(x)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if x is not None and batch_size[0] > 1 and y is not None:
        x = x + 1
    if batch_size[0] > 1 or y is not None or x is not None:
        x = x - 1
    return x


def if_with_and_or_3(x, y=None):
<<<<<<< HEAD
    batch_size = paddle.shape(x)
    mean_res = paddle.mean(x)
    if (
        x is not None
        and batch_size[0] > 1
        and y is not None
        and mean_res.numpy()[0] > 0
    ):
=======
    batch_size = fluid.layers.shape(x)
    mean_res = paddle.mean(x)
    if x is not None and batch_size[0] > 1 and y is not None and mean_res.numpy(
    )[0] > 0:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = x + 1
    if mean_res.numpy()[0] > 0 and (x is not None and batch_size[0] > 1) and y:
        x = x - 1
    return x


def if_with_and_or_4(x, y=None):
<<<<<<< HEAD
    batch_size = paddle.shape(x)
    mean_res = paddle.mean(x)
    if (x is not None and batch_size[0] > 1) or (
        y is not None and mean_res.numpy()[0] > 0
    ):
        x = x + 1
    if (x is not None or batch_size[0] > 1) and (
        y is not None or mean_res.numpy()[0] > 0
    ):
=======
    batch_size = fluid.layers.shape(x)
    mean_res = paddle.mean(x)
    if (x is not None and batch_size[0] > 1) or (y is not None
                                                 and mean_res.numpy()[0] > 0):
        x = x + 1
    if (x is not None or batch_size[0] > 1) and (y is not None
                                                 or mean_res.numpy()[0] > 0):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = x - 1
    return x


def if_with_class_var(x, y=None):
<<<<<<< HEAD
    class Foo:
=======

    class Foo(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def __init__(self):
            self.a = 1
            self.b = 2

    foo = Foo()
<<<<<<< HEAD
    batch_size = paddle.shape(x)
=======
    batch_size = fluid.layers.shape(x)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    mean_res = paddle.mean(x)

    if batch_size[0] > foo.a:
        x = x + foo.b
    else:
        x = x - foo.b
    return x


def if_tensor_case(x):
    x = fluid.dygraph.to_variable(x)

    mean = paddle.mean(x)
    # It is equivalent to `if mean != 0`
    if mean:
        for i in range(0, 10):
            if i > 5:
                x += 1
                break
            x += 1
    else:
        for i in range(0, 37):
            x += 1
            break
            x += i

    # join `and`/`or`
    if paddle.mean(x) + 1 and mean > 1 and x is not None or 2 > 1:
        x -= 1

    # `not` statement
    if not (x[0][0] and (mean * x)[0][0]):
        x += 1

    return x


def dyfunc_ifelse_ret_int1(x):
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return y, index
    else:
        y = x[index] + 2
        index = index + 1
        return y, index


def dyfunc_ifelse_ret_int2(x):
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return y, index
    else:
        y = x[index] + 2
        index = index + 1
        return y


def dyfunc_ifelse_ret_int3(x):
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return index
    else:
        y = x[index] + 2
        return y


def dyfunc_ifelse_ret_int4(x):
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return 'unsupport ret'
    else:
        y = x[index] + 2
        return y
