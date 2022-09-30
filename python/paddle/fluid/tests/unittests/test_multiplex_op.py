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
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import _test_eager_guard


class TestMultiplexOp(OpTest):

    def setUp(self):
        self.op_type = "multiplex"
        rows = 4
        index = np.arange(0, rows).astype('int32')
        np.random.shuffle(index)
        index = np.reshape(index, (rows, 1))
        ins1 = np.random.random((rows, 25)).astype("float64")
        ins2 = np.random.random((rows, 25)).astype("float64")
        ins3 = np.random.random((rows, 25)).astype("float64")
        ins4 = np.random.random((rows, 25)).astype("float64")
        self.inputs = {
            'Ids': index,
            'X': [('x1', ins1), ('x2', ins2), ('x3', ins3), ('x4', ins4)]
        }
        # multiplex output
        output = np.zeros_like(ins1)
        for i in range(0, rows):
            k = index[i][0]
            output[i] = self.inputs['X'][k][1][i]
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x1', 'x2', 'x3', 'x4'], 'Out')

    def test_check_grad_ignore_x1(self):
        self.check_grad(['x2', 'x3', 'x4'], 'Out', no_grad_set=set('x1'))

    def test_check_grad_ignore_x1_x2(self):
        self.check_grad(['x3', 'x4'], 'Out', no_grad_set=set(['x1', 'x2']))

    def test_check_grad_ignore_x3(self):
        self.check_grad(['x1', 'x2', 'x4'], 'Out', no_grad_set=set('x3'))


class TestMultiplexOpError(unittest.TestCase):

    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x1 = fluid.data(name='x1', shape=[None, 2], dtype='int64')
            x2 = fluid.data(name='x2', shape=[None, 2], dtype='int64')
            index = fluid.data(name='index', shape=[None, 1], dtype='int32')

            def test_list():
                # the inputs type must be list
                paddle.multiplex(inputs=x1, index=index)

            self.assertRaises(TypeError, test_list)

            def test_len():
                paddle.multiplex(inputs=[x1], index=index)

            self.assertRaises(ValueError, test_len)

            def test_type():
                y1 = fluid.data(name='y1', shape=[None, 2], dtype='int16')
                y2 = fluid.data(name='y2', shape=[None, 2], dtype='int16')
                paddle.multiplex(inputs=[y1, y2], index=index)

            self.assertRaises(TypeError, test_type)

            def test_type2():
                index2 = fluid.data(name='index2',
                                    shape=[None, 1],
                                    dtype='int16')
                paddle.multiplex(inputs=[x1, x2], index=index2)

            self.assertRaises(TypeError, test_type2)


class TestMultiplexODygrap(unittest.TestCase):

    def test_multiplex_dygraph(self):
        paddle.disable_static()
        img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
        img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
        inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
        index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
        res = paddle.multiplex(inputs, index)
        paddle.enable_static()

    def test_dygraph_api(self):
        with fluid.dygraph.guard():
            img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
            img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
            inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
            index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
            inputs[0].stop_gradient = False
            inputs[1].stop_gradient = False
            res = paddle.multiplex(inputs, index)
            res.backward()
            with _test_eager_guard():
                inputs_eager = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
                index_eager = paddle.to_tensor(
                    np.array([[1], [0]]).astype(np.int32))
                inputs_eager[0].stop_gradient = False
                inputs_eager[1].stop_gradient = False
                res_eager = paddle.multiplex(inputs_eager, index_eager)
                res_eager.backward()
                self.assertEqual((res.numpy() == res_eager.numpy()).all(), True)
                self.assertEqual(
                    (inputs[0].grad.numpy() == inputs_eager[0].grad.numpy()
                     ).all(), True)
                self.assertEqual(
                    (inputs[1].grad.numpy() == inputs_eager[1].grad.numpy()
                     ).all(), True)


if __name__ == '__main__':
    unittest.main()
