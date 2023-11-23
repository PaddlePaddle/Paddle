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

import unittest

import numpy as np
import op_test

import paddle


def create_test_not_equal_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = np.random.random(size=(10, 7)).astype(typename)
            y = np.random.random(size=(10, 7)).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            self.check_output(check_pir=True)

    cls_name = "{}_{}_{}".format(op_type, typename, 'not_equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


def create_test_not_shape_equal_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = np.random.random(size=(10, 7)).astype(typename)
            y = np.random.random(size=(10)).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            self.check_output(check_pir=True)

    cls_name = "{}_{}_{}".format(op_type, typename, 'not_shape_equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


def create_test_equal_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = y = np.random.random(size=(10, 7)).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            self.check_output(check_pir=True)

    cls_name = "{}_{}_{}".format(op_type, typename, 'equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


def create_test_dim1_class(op_type, typename, callback):
    class Cls(op_test.OpTest):
        def setUp(self):
            x = y = np.random.random(size=(1)).astype(typename)
            x = np.array([True, False, True]).astype(typename)
            x = np.array([False, False, True]).astype(typename)
            z = callback(x, y)
            self.python_api = paddle.tensor.equal_all
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': z}
            self.op_type = op_type

        def test_output(self):
            self.check_output(check_pir=True)

    cls_name = "{}_{}_{}".format(op_type, typename, 'equal_all')
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


np_equal = lambda _x, _y: np.array(np.array_equal(_x, _y))

for _type_name in {'float32', 'float64', 'int32', 'int64', 'bool'}:
    create_test_not_equal_class('equal_all', _type_name, np_equal)
    create_test_equal_class('equal_all', _type_name, np_equal)
    create_test_dim1_class('equal_all', _type_name, np_equal)


class TestEqualReduceAPI(unittest.TestCase):
    def test_dynamic_api(self):
        paddle.disable_static()
        x = paddle.ones(shape=[10, 10], dtype="int32")
        y = paddle.ones(shape=[10, 10], dtype="int32")
        out = paddle.equal_all(x, y)
        assert out.item() is True
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
