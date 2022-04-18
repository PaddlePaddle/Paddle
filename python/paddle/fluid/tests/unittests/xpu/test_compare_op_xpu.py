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

from __future__ import print_function

import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test_xpu import OpTest, XPUOpTest
import paddle
from paddle.fluid import Program, program_guard


def create_test_class(op_type, typename, callback):
    class Cls(OpTest):
        def setUp(self):
            a = np.random.random(size=(10, 7)).astype(typename)
            b = np.random.random(size=(10, 7)).astype(typename)
            c = callback(a, b)
            self.inputs = {'X': a, 'Y': b}
            self.outputs = {'Out': c}
            self.op_type = op_type
            self.use_xpu = True
            self.attrs = {'use_xpu': True}

        def test_check_output(self):
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_errors(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = fluid.layers.data(name='x', shape=[2], dtype='int32')
                y = fluid.layers.data(name='y', shape=[2], dtype='int32')
                a = fluid.layers.data(name='a', shape=[2], dtype='int16')
                if self.op_type == "less_than":
                    self.assertRaises(
                        TypeError,
                        fluid.layers.less_than,
                        x=x,
                        y=y,
                        force_cpu=1)
                op = eval("fluid.layers.%s" % self.op_type)
                self.assertRaises(TypeError, op, x=x, y=y, cond=1)
                self.assertRaises(TypeError, op, x=x, y=a)
                self.assertRaises(TypeError, op, x=a, y=y)

    cls_name = "{0}_{1}".format(op_type, typename)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


for _type_name in {'int32'}:
    if _type_name == 'float64' and core.is_compiled_with_rocm():
        _type_name = 'float32'

    create_test_class('less_than', _type_name, lambda _a, _b: _a < _b)
    create_test_class('less_equal', _type_name, lambda _a, _b: _a <= _b)
    create_test_class('greater_than', _type_name, lambda _a, _b: _a > _b)
    create_test_class('greater_equal', _type_name, lambda _a, _b: _a >= _b)
    create_test_class('equal', _type_name, lambda _a, _b: _a == _b)
    create_test_class('not_equal', _type_name, lambda _a, _b: _a != _b)


def create_paddle_case(op_type, callback):
    class PaddleCls(unittest.TestCase):
        def setUp(self):
            self.op_type = op_type
            self.input_x = np.array([1, 2, 3, 4]).astype(np.int64)
            self.input_y = np.array([1, 3, 2, 4]).astype(np.int64)
            self.real_result = callback(self.input_x, self.input_y)
            self.place = fluid.XPUPlace(0) if fluid.core.is_compiled_with_xpu(
            ) else fluid.CPUPlace()

        def test_api(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = fluid.data(name='x', shape=[4], dtype='int64')
                y = fluid.data(name='y', shape=[4], dtype='int64')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = fluid.Executor(self.place)
                res, = exe.run(feed={"x": self.input_x,
                                     "y": self.input_y},
                               fetch_list=[out])
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
                    res, = exe.run(feed={"x": self.input_x,
                                         "y": 1.0},
                                   fetch_list=[out])
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

        def test_assert(self):
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
                x = paddle.static.data(
                    name='x', shape=[1, 2, 1, 3], dtype='int32')
                y = paddle.static.data(name='y', shape=[1, 2, 3], dtype='int32')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.int32)
                input_y = np.arange(0, 6).reshape((1, 2, 3)).astype(np.int32)
                real_result = callback(input_x, input_y)
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
            self.assertEqual((res == real_result).all(), True)

        def test_broadcast_api_2(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[1, 2, 3], dtype='int32')
                y = paddle.static.data(
                    name='y', shape=[1, 2, 1, 3], dtype='int32')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(0, 6).reshape((1, 2, 3)).astype(np.int32)
                input_y = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.int32)
                real_result = callback(input_x, input_y)
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
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
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
            self.assertEqual((res == real_result).all(), True)

        def test_bool_api_4(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[3, 1], dtype='bool')
                y = paddle.static.data(name='y', shape=[3, 1], dtype='bool')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.array([True, False, True]).astype(np.bool)
                input_y = np.array([True, True, False]).astype(np.bool)
                real_result = callback(input_x, input_y)
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
            self.assertEqual((res == real_result).all(), True)

        def test_bool_broadcast_api_4(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[3, 1], dtype='bool')
                y = paddle.static.data(name='y', shape=[1], dtype='bool')
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.array([True, False, True]).astype(np.bool)
                input_y = np.array([True]).astype(np.bool)
                real_result = callback(input_x, input_y)
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
            self.assertEqual((res == real_result).all(), True)

        def test_attr_name(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = fluid.layers.data(name='x', shape=[4], dtype='int32')
                y = fluid.layers.data(name='y', shape=[4], dtype='int32')
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

if __name__ == '__main__':
    unittest.main()
