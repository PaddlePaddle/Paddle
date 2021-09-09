#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def create_test_class(op_type, typename, callback):
    class Cls(OpTest):
        def setUp(self):
            self.set_npu()
            self.place = paddle.NPUPlace(0)
            x = np.random.random(size=(10, 7)).astype(typename)
            y = np.random.random(size=(10, 7)).astype(typename)
            out = callback(x, y)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': out}
            self.op_type = op_type

        def set_npu(self):
            self.__class__.use_npu = True

        def test_output(self):
            self.check_output_with_place(place=self.place)

        def test_errors(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                a = fluid.layers.data(name='a', shape=[2], dtype='float32')
                b = fluid.layers.data(name='b', shape=[2], dtype='float32')
                c = fluid.layers.data(name='c', shape=[2], dtype='int16')
                d = fluid.create_lod_tensor(np.array([[-1]]), [[1]], self.place)

                op = eval("fluid.layers.%s" % self.op_type)
                self.assertRaises(TypeError, op, x=a, y=b, axis=True)
                self.assertRaises(TypeError, op, x=a, y=b, force_cpu=1)
                self.assertRaises(TypeError, op, x=a, y=b, cond=1)
                self.assertRaises(TypeError, op, x=a, y=c)
                self.assertRaises(TypeError, op, x=c, y=a)
                self.assertRaises(TypeError, op, x=a, y=d)
                self.assertRaises(TypeError, op, x=d, y=a)
                self.assertRaises(TypeError, op, x=c, y=d)

        def test_dynamic_api(self):
            paddle.disable_static()
            paddle.set_device('npu:0')
            x = np.random.random(size=(10, 7)).astype(typename)
            y = np.random.random(size=(10, 7)).astype(typename)
            real_result = callback(x, y)
            x = paddle.to_tensor(x, dtype=typename)
            y = paddle.to_tensor(y, dtype=typename)
            op = eval("paddle.%s" % (self.op_type))
            out = op(x, y)
            self.assertEqual((out.numpy() == real_result).all(), True)

        @unittest.skipIf(typename == 'float16', "float16 is not supported now")
        def test_broadcast_api_1(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(
                    name='x', shape=[1, 2, 1, 3], dtype=typename)
                y = paddle.static.data(
                    name='y', shape=[1, 2, 3], dtype=typename)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(typename)
                input_y = np.arange(0, 6).reshape((1, 2, 3)).astype(typename)
                real_result = callback(input_x, input_y)
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
            self.assertEqual((res == real_result).all(), True)

        @unittest.skipIf(typename == 'float16', "float16 is not supported now")
        def test_broadcast_api_2(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(
                    name='x', shape=[1, 2, 3], dtype=typename)
                y = paddle.static.data(
                    name='y', shape=[1, 2, 1, 3], dtype=typename)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(0, 6).reshape((1, 2, 3)).astype(typename)
                input_y = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(typename)
                real_result = callback(input_x, input_y)
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
            self.assertEqual((res == real_result).all(), True)

        @unittest.skipIf(typename == 'float16', "float16 is not supported now")
        def test_broadcast_api_3(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = paddle.static.data(name='x', shape=[5], dtype=typename)
                y = paddle.static.data(name='y', shape=[3, 1], dtype=typename)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(0, 5).reshape((5)).astype(typename)
                input_y = np.array([5, 3, 2]).reshape((3, 1)).astype(typename)
                real_result = callback(input_x, input_y)
                res, = exe.run(feed={"x": input_x,
                                     "y": input_y},
                               fetch_list=[out])
            self.assertEqual((res == real_result).all(), True)

        @unittest.skipIf(typename == 'float16', "float16 is not supported now")
        def test_attr_name(self):
            paddle.enable_static()
            with program_guard(Program(), Program()):
                x = fluid.layers.data(name='x', shape=[4], dtype=typename)
                y = fluid.layers.data(name='y', shape=[4], dtype=typename)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x=x, y=y, name="name_%s" % (self.op_type))
            self.assertEqual("name_%s" % (self.op_type) in out.name, True)

    cls_name = "{0}_{1}".format(op_type, typename)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


for _type_name in {'float16', 'float32', 'int32', 'int64', 'bool'}:
    if _type_name == 'int32' or _type_name == 'bool':
        create_test_class('equal', _type_name, lambda _a, _b: _a == _b)
        continue
    create_test_class('equal', _type_name, lambda _a, _b: _a == _b)
    create_test_class('not_equal', _type_name, lambda _a, _b: _a != _b)
    create_test_class('less_than', _type_name, lambda _a, _b: _a < _b)
    create_test_class('less_equal', _type_name, lambda _a, _b: _a <= _b)
    create_test_class('greater_than', _type_name, lambda _a, _b: _a > _b)
    create_test_class('greater_equal', _type_name, lambda _a, _b: _a >= _b)

if __name__ == '__main__':
    unittest.main()
