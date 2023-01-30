#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy
import numpy as np
import op_test

=======
from __future__ import print_function

import op_test
import unittest
import numpy
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


def create_test_class(op_type, typename, callback):
<<<<<<< HEAD
    class Cls(op_test.OpTest):
=======

    class Cls(op_test.OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            a = numpy.random.random(size=(10, 7)).astype(typename)
            b = numpy.random.random(size=(10, 7)).astype(typename)
            c = callback(a, b)
            self.python_api = eval("paddle." + op_type)
            self.inputs = {'X': a, 'Y': b}
            self.outputs = {'Out': c}
            self.op_type = op_type

        def test_output(self):
            self.check_output(check_eager=False)

        def test_errors(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
<<<<<<< HEAD
                x = paddle.static.data(name='x', shape=[-1, 2], dtype='int32')
                y = paddle.static.data(name='y', shape=[-1, 2], dtype='int32')
                a = paddle.static.data(name='a', shape=[-1, 2], dtype='int16')
                op = eval("paddle.%s" % self.op_type)
=======
                x = fluid.layers.data(name='x', shape=[2], dtype='int32')
                y = fluid.layers.data(name='y', shape=[2], dtype='int32')
                a = fluid.layers.data(name='a', shape=[2], dtype='int16')
                if self.op_type == "less_than":
                    self.assertRaises(TypeError,
                                      fluid.layers.less_than,
                                      x=x,
                                      y=y,
                                      force_cpu=1)
                op = eval("fluid.layers.%s" % self.op_type)
                self.assertRaises(TypeError, op, x=x, y=y, cond=1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                self.assertRaises(TypeError, op, x=x, y=a)
                self.assertRaises(TypeError, op, x=a, y=y)

    cls_name = "{0}_{1}".format(op_type, typename)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


for _type_name in {'float32', 'float64', 'int32', 'int64', 'float16'}:
    if _type_name == 'float64' and core.is_compiled_with_rocm():
        _type_name = 'float32'
    if _type_name == 'float16' and (not core.is_compiled_with_cuda()):
        continue

    create_test_class('less_than', _type_name, lambda _a, _b: _a < _b)
    create_test_class('less_equal', _type_name, lambda _a, _b: _a <= _b)
    create_test_class('greater_than', _type_name, lambda _a, _b: _a > _b)
    create_test_class('greater_equal', _type_name, lambda _a, _b: _a >= _b)
    create_test_class('equal', _type_name, lambda _a, _b: _a == _b)
    create_test_class('not_equal', _type_name, lambda _a, _b: _a != _b)


def create_paddle_case(op_type, callback):
<<<<<<< HEAD
    class PaddleCls(unittest.TestCase):
=======

    class PaddleCls(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.op_type = op_type
            self.input_x = np.array([1, 2, 3, 4]).astype(np.int64)
            self.input_y = np.array([1, 3, 2, 4]).astype(np.int64)
            self.real_result = callback(self.input_x, self.input_y)
            self.place = fluid.CPUPlace()
            if core.is_compiled_with_cuda():
                self.place = paddle.CUDAPlace(0)

        def test_api(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = fluid.data(name='x', shape=[4], dtype='int64')
                y = fluid.data(name='y', shape=[4], dtype='int64')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = fluid.Executor(self.place)
<<<<<<< HEAD
                (res,) = exe.run(
                    feed={"x": self.input_x, "y": self.input_y},
                    fetch_list=[out],
                )
=======
                res, = exe.run(feed={
                    "x": self.input_x,
                    "y": self.input_y
                },
                               fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual((res == self.real_result).all(), True)

        def test_api_float(self):
            if self.op_type == "equal":
                paddle.enable_static()
                with program_guard(Program(), Program()):
                    x = fluid.data(name='x', shape=[4], dtype='int64')
                    y = fluid.data(name='y', shape=[1], dtype='int64')
                    op = eval("paddle.%s" % (self.op_type))
                    out = op(x, y)
                    exe = fluid.Executor(self.place)
<<<<<<< HEAD
                    (res,) = exe.run(
                        feed={"x": self.input_x, "y": 1.0}, fetch_list=[out]
                    )
=======
                    res, = exe.run(feed={
                        "x": self.input_x,
                        "y": 1.0
                    },
                                   fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((res == self.real_result).all(), True)

        def test_dynamic_api(self):
            paddle.disable_static()
            x = paddle.to_tensor(self.input_x)
            y = paddle.to_tensor(self.input_y)
            op = eval("paddle.%s" % (self.op_type))
            out = op(x, y)
            self.assertEqual((out.numpy() == self.real_result).all(), True)
            paddle.enable_static()

        def test_dynamic_api_int(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x = paddle.to_tensor(self.input_x)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, 1)
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_dynamic_api_float(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x = paddle.to_tensor(self.input_x)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, 1.0)
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_dynamic_api_inf_1(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('inf'), float('inf')]).astype(np.int64)
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-inf'), float('inf')]).astype(np.int64)
                y = paddle.to_tensor(y1)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
<<<<<<< HEAD
                    True,
                )
=======
                    True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.enable_static()

        def test_dynamic_api_inf_2(self):
            if self.op_type == "equal":
                paddle.disable_static()
<<<<<<< HEAD
                x1 = np.array([1, float('inf'), float('inf')]).astype(
                    np.float32
                )
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-inf'), float('inf')]).astype(
                    np.float32
                )
=======
                x1 = np.array([1, float('inf'),
                               float('inf')]).astype(np.float32)
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-inf'),
                               float('inf')]).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                y = paddle.to_tensor(y1)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
<<<<<<< HEAD
                    True,
                )
=======
                    True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.enable_static()

        def test_dynamic_api_inf_3(self):
            if self.op_type == "equal":
                paddle.disable_static()
<<<<<<< HEAD
                x1 = np.array([1, float('inf'), float('-inf')]).astype(
                    np.float32
                )
=======
                x1 = np.array([1, float('inf'),
                               float('-inf')]).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                x = paddle.to_tensor(x1)
                y1 = np.array([1, 2, 3]).astype(np.float32)
                y = paddle.to_tensor(y1)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
<<<<<<< HEAD
                    True,
                )
=======
                    True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.enable_static()

        def test_dynamic_api_nan_1(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('nan'), float('nan')]).astype(np.int64)
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-nan'), float('nan')]).astype(np.int64)
                y = paddle.to_tensor(y1)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
<<<<<<< HEAD
                    True,
                )
=======
                    True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.enable_static()

        def test_dynamic_api_nan_2(self):
            if self.op_type == "equal":
                paddle.disable_static()
<<<<<<< HEAD
                x1 = np.array([1, float('nan'), float('nan')]).astype(
                    np.float32
                )
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-nan'), float('nan')]).astype(
                    np.float32
                )
=======
                x1 = np.array([1, float('nan'),
                               float('nan')]).astype(np.float32)
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-nan'),
                               float('nan')]).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                y = paddle.to_tensor(y1)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
<<<<<<< HEAD
                    True,
                )
=======
                    True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.enable_static()

        def test_dynamic_api_nan_3(self):
            if self.op_type == "equal":
                paddle.disable_static()
<<<<<<< HEAD
                x1 = np.array([1, float('-nan'), float('nan')]).astype(
                    np.float32
                )
=======
                x1 = np.array([1, float('-nan'),
                               float('nan')]).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                x = paddle.to_tensor(x1)
                y1 = np.array([1, 2, 1]).astype(np.float32)
                y = paddle.to_tensor(y1)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
<<<<<<< HEAD
                    True,
                )
=======
                    True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                paddle.enable_static()

        def test_not_equal(self):
            if self.op_type == "not_equal":
                paddle.disable_static()
<<<<<<< HEAD
                x = paddle.to_tensor(
                    np.array([1.2e-8, 2, 2, 1]), dtype="float32"
                )
                y = paddle.to_tensor(
                    np.array([1.1e-8, 2, 2, 1]), dtype="float32"
                )
=======
                x = paddle.to_tensor(np.array([1.2e-8, 2, 2, 1]),
                                     dtype="float32")
                y = paddle.to_tensor(np.array([1.1e-8, 2, 2, 1]),
                                     dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                self.real_result = np.array([0, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_assert(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            def test_dynamic_api_string(self):
                if self.op_type == "equal":
                    paddle.disable_static()
                    x = paddle.to_tensor(self.input_x)
                    op = eval("paddle.%s" % (self.op_type))
                    out = op(x, "1.0")
                    paddle.enable_static()

            self.assertRaises(TypeError, test_dynamic_api_string)

        def test_dynamic_api_bool(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x = paddle.to_tensor(self.input_x)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, True)
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_broadcast_api_1(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
<<<<<<< HEAD
                x = paddle.static.data(
                    name='x', shape=[1, 2, 1, 3], dtype='int32'
                )
=======
                x = paddle.static.data(name='x',
                                       shape=[1, 2, 1, 3],
                                       dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                y = paddle.static.data(name='y', shape=[1, 2, 3], dtype='int32')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.int32)
                input_y = np.arange(0, 6).reshape((1, 2, 3)).astype(np.int32)
                real_result = callback(input_x, input_y)
<<<<<<< HEAD
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
=======
                res, = exe.run(feed={
                    "x": input_x,
                    "y": input_y
                },
                               fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual((res == real_result).all(), True)

        def test_broadcast_api_2(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[1, 2, 3], dtype='int32')
<<<<<<< HEAD
                y = paddle.static.data(
                    name='y', shape=[1, 2, 1, 3], dtype='int32'
                )
=======
                y = paddle.static.data(name='y',
                                       shape=[1, 2, 1, 3],
                                       dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(0, 6).reshape((1, 2, 3)).astype(np.int32)
                input_y = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.int32)
                real_result = callback(input_x, input_y)
<<<<<<< HEAD
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
=======
                res, = exe.run(feed={
                    "x": input_x,
                    "y": input_y
                },
                               fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual((res == real_result).all(), True)

        def test_broadcast_api_3(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[5], dtype='int32')
                y = paddle.static.data(name='y', shape=[3, 1], dtype='int32')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(0, 5).reshape((5)).astype(np.int32)
                input_y = np.array([5, 3, 2]).reshape((3, 1)).astype(np.int32)
                real_result = callback(input_x, input_y)
<<<<<<< HEAD
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
            self.assertEqual((res == real_result).all(), True)

        def test_zero_dim_api_1(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.randint(-3, 3, shape=[], dtype='int32')
                y = paddle.randint(-3, 3, shape=[], dtype='int32')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                (
                    x_np,
                    y_np,
                    res,
                ) = exe.run(fetch_list=[x, y, out])
                real_result = callback(x_np, y_np)
            self.assertEqual((res == real_result).all(), True)

        def test_zero_dim_api_2(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.randint(-3, 3, shape=[2, 3, 4], dtype='int32')
                y = paddle.randint(-3, 3, shape=[], dtype='int32')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                (
                    x_np,
                    y_np,
                    res,
                ) = exe.run(fetch_list=[x, y, out])
                real_result = callback(x_np, y_np)
            self.assertEqual((res == real_result).all(), True)

        def test_zero_dim_api_3(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.randint(-3, 3, shape=[], dtype='int32')
                y = paddle.randint(-3, 3, shape=[2, 3, 4], dtype='int32')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                (
                    x_np,
                    y_np,
                    res,
                ) = exe.run(fetch_list=[x, y, out])
                real_result = callback(x_np, y_np)
=======
                res, = exe.run(feed={
                    "x": input_x,
                    "y": input_y
                },
                               fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual((res == real_result).all(), True)

        def test_bool_api_4(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[3, 1], dtype='bool')
                y = paddle.static.data(name='y', shape=[3, 1], dtype='bool')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.array([True, False, True]).astype(np.bool_)
                input_y = np.array([True, True, False]).astype(np.bool_)
                real_result = callback(input_x, input_y)
<<<<<<< HEAD
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
=======
                res, = exe.run(feed={
                    "x": input_x,
                    "y": input_y
                },
                               fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual((res == real_result).all(), True)

        def test_bool_broadcast_api_4(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[3, 1], dtype='bool')
                y = paddle.static.data(name='y', shape=[1], dtype='bool')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.array([True, False, True]).astype(np.bool_)
                input_y = np.array([True]).astype(np.bool_)
                real_result = callback(input_x, input_y)
<<<<<<< HEAD
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
=======
                res, = exe.run(feed={
                    "x": input_x,
                    "y": input_y
                },
                               fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertEqual((res == real_result).all(), True)

        def test_attr_name(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
<<<<<<< HEAD
                x = paddle.static.data(name='x', shape=[-1, 4], dtype='int32')
                y = paddle.static.data(name='y', shape=[-1, 4], dtype='int32')
=======
                x = fluid.layers.data(name='x', shape=[4], dtype='int32')
                y = fluid.layers.data(name='y', shape=[4], dtype='int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                op = eval("paddle.%s" % (self.op_type))
                out = op(x=x, y=y, name="name_%s" % (self.op_type))
            self.assertEqual("name_%s" % (self.op_type) in out.name, True)

    cls_name = "TestCase_{}".format(op_type)
    PaddleCls.__name__ = cls_name
    globals()[cls_name] = PaddleCls


create_paddle_case('less_than', lambda _a, _b: _a < _b)
create_paddle_case('less_equal', lambda _a, _b: _a <= _b)
create_paddle_case('greater_than', lambda _a, _b: _a > _b)
create_paddle_case('greater_equal', lambda _a, _b: _a >= _b)
create_paddle_case('equal', lambda _a, _b: _a == _b)
create_paddle_case('not_equal', lambda _a, _b: _a != _b)


class TestCompareOpError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input x and y of compare_op must be Variable.
<<<<<<< HEAD
            x = paddle.static.data(name='x', shape=[-1, 1], dtype="float32")
            y = fluid.create_lod_tensor(
                numpy.array([[-1]]), [[1]], fluid.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.greater_equal, x, y)


class API_TestElementwise_Equal(unittest.TestCase):
=======
            x = fluid.layers.data(name='x', shape=[1], dtype="float32")
            y = fluid.create_lod_tensor(numpy.array([[-1]]), [[1]],
                                        fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.greater_equal, x, y)


class API_TestElementwise_Equal(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
            out = paddle.equal(x=label, y=limit)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
<<<<<<< HEAD
            (res,) = exe.run(fetch_list=[out])
=======
            res, = exe.run(fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual((res == np.array([True, False])).all(), True)

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
            limit = fluid.layers.assign(np.array([3, 3], dtype="int32"))
            out = paddle.equal(x=label, y=limit)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
<<<<<<< HEAD
            (res,) = exe.run(fetch_list=[out])
=======
            res, = exe.run(fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual((res == np.array([True, True])).all(), True)


class TestCompareOpPlace(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_place_1(self):
        paddle.enable_static()
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        label = fluid.layers.assign(np.array([3, 3], dtype="int32"))
        limit = fluid.layers.assign(np.array([3, 2], dtype="int32"))
<<<<<<< HEAD
        out = paddle.less_than(label, limit)
        exe = fluid.Executor(place)
        (res,) = exe.run(fetch_list=[out])
=======
        out = fluid.layers.less_than(label, limit, force_cpu=True)
        exe = fluid.Executor(place)
        res, = exe.run(fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual((res == np.array([False, False])).all(), True)

    def test_place_2(self):
        place = paddle.CPUPlace()
        data_place = place
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            data_place = paddle.CUDAPinnedPlace()
        paddle.disable_static(place)
        data = np.array([9], dtype="int64")
        data_tensor = paddle.to_tensor(data, place=data_place)
        result = data_tensor == 0
        self.assertEqual((result.numpy() == np.array([False])).all(), True)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
