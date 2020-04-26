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

import op_test
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def create_test_broadcast_class(op_type, args, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = np.random.random(size=args['x_size']).astype('int32')
            y = np.random.random(size=args['y_size']).astype('int32')
            z = callback(x, y)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type
            self.axis = args['axis']

        def test_output(self):
            self.check_output()

    cls_name = "{0}_{1}".format(op_type, 'broadcast')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


def create_test_not_equal_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = np.random.random(size=(10, 7)).astype(typename)
            y = np.random.random(size=(10, 7)).astype(typename)
            z = callback(x, y)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            self.check_output()

    cls_name = "{0}_{1}_{2}".format(op_type, typename, 'not_equal')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


def create_test_equal_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = y = np.random.random(size=(10, 7)).astype(typename)
            z = callback(x, y)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            self.check_output()

    cls_name = "{0}_{1}_{2}".format(op_type, typename, 'equal')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


def create_test_dim1_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = y = np.random.random(size=(1)).astype(typename)
            z = callback(x, y)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            self.check_output()

    cls_name = "{0}_{1}_{2}".format(op_type, typename, 'equal')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


np_equal = lambda _x, _y: np.array(np.array_equal(_x, _y))

for _type_name in {'float32', 'float64', 'int32', 'int64'}:
    create_test_not_equal_class('equal_reduce', _type_name, np_equal)
    create_test_equal_class('equal_reduce', _type_name, np_equal)
    create_test_dim1_class('equal_reduce', _type_name, np_equal)

broadcast_args = [{
    'x_size': (100, 2, 3),
    'y_size': (100),
    'axis': 0
}, {
    'x_size': (2, 100, 3),
    'y_size': (100),
    'axis': 1
}, {
    'x_size': (2, 3, 100),
    'y_size': (1, 1),
    'axis': -1
}, {
    'x_size': (2, 10, 12, 3),
    'y_size': (10, 12),
    'axis': 1
}, {
    'x_size': (100, 2, 3, 4),
    'y_size': (100, 1),
    'axis': 0
}, {
    'x_size': (10, 3, 12),
    'y_size': (10, 1, 12),
    'axis': -1
}, {
    'x_size': (2, 12, 3, 5),
    'y_size': (2, 12, 1, 5),
    'axis': -1
}, {
    'x_size': (2, 12, 3, 5),
    'y_size': (3, 5),
    'axis': 2
}]


def np_broadcast_equal(_x, _y):
    res = np.all(np.equal(_x, _y))
    return np.array(res)


for args in broadcast_args:
    create_test_broadcast_class('equal_reduce', args, np_broadcast_equal)


class TestEqualReduceAPI(unittest.TestCase):
    def test_name(self):
        x = fluid.layers.assign(np.array([3, 4], dtype="int32"))
        y = fluid.layers.assign(np.array([3, 4], dtype="int32"))
        out = paddle.equal(x, y, name='equal_res')
        assert 'equal_res' in out.name


if __name__ == '__main__':
    unittest.main()
