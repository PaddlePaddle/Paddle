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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestSequenceUnpadOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'sequence_unpad'
        self.use_dynamic_create_class = False

    class TestSequenceUnpadOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.initTestCase()
            self.set_xpu()
            self.op_type = 'sequence_unpad'
            self.place = paddle.XPUPlace(0)
            self.compute()

        def init_dtype(self):
            self.dtype = self.in_type

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def initTestCase(self):
            self.length = [2, 3, 4]
            self.x_shape = (3, 40)

        def compute(self):
            assert len(self.length) == self.x_shape[0]
            x = np.random.random(self.x_shape).astype(self.dtype)
            out_lod = [self.length]

            out = x[0, 0 : self.length[0]]
            for i in range(1, x.shape[0]):
                out = np.append(out, x[i, 0 : self.length[i]], axis=0)

            out_shape = (sum(self.length),)
            if len(self.x_shape) == 2:
                out_shape = out_shape + (1,)
            else:
                out_shape = out_shape + self.x_shape[2:]

            self.inputs = {
                'X': x,
                'Length': np.array(self.length).astype('int64'),
            }
            self.outputs = {'Out': (out.reshape(out_shape), out_lod)}

    class TestSequenceUnpadOp2(TestSequenceUnpadOp):
        def initTestCase(self):
            self.length = [2, 3, 4]
            self.x_shape = (3, 5, 4, 3)

    class TestSequenceUnpadOp3(TestSequenceUnpadOp):
        def initTestCase(self):
            self.length = [5, 2, 3, 4]
            self.x_shape = (4, 5, 3, 3, 6)

    class TestSequenceUnpadOp4(TestSequenceUnpadOp):
        def initTestCase(self):
            self.length = [5, 5, 5, 5]
            self.x_shape = (4, 5, 3, 3, 6)

    class TestSequenceUnpadOp5(TestSequenceUnpadOp):
        def initTestCase(self):
            self.length = [1, 4, 3, 1]
            self.x_shape = (4, 5, 3, 3, 6)


class TestSequenceUnpadOpError(unittest.TestCase):
    def test_error(self):
        """
        The type of 'x' in paddle.static.nn.sequence_unpad must be <class 'paddle.base.framework.Variable'>, but received <class 'numpy.ndarray'>.
        """

        def test_x_variable():
            x = np.random.random((10, 5)).astype("float64")
            len = paddle.static.data(name='length2', shape=[10], dtype='int64')
            paddle.static.nn.sequence_lod.sequence_unpad(x=x, length=len)

        self.assertRaises(TypeError, test_x_variable)
        """
        The type of 'length' in base.layers.sequence_unpad must be <class 'paddle.base.framework.Variable'>, but received <class 'numpy.ndarray'>.
        """

        def test_length_variable():
            x1 = paddle.static.data(name='x1', shape=[10, 5], dtype='float32')
            len1 = np.random.random(10).astype("int64")
            paddle.static.nn.sequence_lod.sequence_unpad(x=x1, length=len1)

        self.assertRaises(TypeError, test_length_variable)
        """
        The data type of 'x' in base.layers.sequence_unpad must be ['float32', 'float64', 'int32', 'int64'], but received float16
        """

        def test_x_dtype():
            x2 = paddle.static.data(name='x2', shape=[10, 5], dtype='float16')
            len2 = paddle.static.data(name='length2', shape=[10], dtype='int64')
            paddle.static.nn.sequence_lod.sequence_unpad(x=x2, length=len2)

        self.assertRaises(TypeError, test_x_dtype)
        """
        The data type of 'length' in base.layers.sequence_unpad must be ['int64'], but received int32
        """

        def test_length_dtype():
            x3 = paddle.static.data(name='x3', shape=[10, 5], dtype='float64')
            len3 = paddle.static.data(name='length3', shape=[10], dtype='int32')
            paddle.static.nn.sequence_lod.sequence_unpad(x=x3, length=len3)

        self.assertRaises(TypeError, test_length_dtype)


support_types = get_xpu_op_support_types('sequence_unpad')
for stype in support_types:
    create_test_class(globals(), XPUTestSequenceUnpadOp, stype)

if __name__ == '__main__':
    unittest.main()
