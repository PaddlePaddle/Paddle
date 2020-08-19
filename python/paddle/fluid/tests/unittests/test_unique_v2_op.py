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

import unittest
import numpy as np
from op_test import OpTest
import paddle


class TestUniqueV2Op(OpTest):
    def setUp(self):
        self.op_type = "unique_v2"
        self.init_config()

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.inputs = {'X': np.array([2, 3, 3, 1, 5, 3], dtype='int64')}
        unique, indices, inverse, count = np.unique(
            self.inputs['X'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=None)
        print(unique)
        self.attrs = {
            "return_index": True,
            "return_inverse": True,
            "return_counts": True,
            "axis": None
        }
        self.outputs = {
            'Out': unique,
            'Indices': indices,
            "Inverse": inverse,
            "Counts": count,
        }


class TestUniqueV2OpAxisNone(TestUniqueV2Op):
    def init_config(self):
        self.inputs = {'X': np.random.random((4, 7, 10)).astype('float64')}
        unique, indices, inverse, counts = np.unique(
            self.inputs['X'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=None)
        self.attrs = {
            "return_index": True,
            "return_inverse": True,
            "return_counts": True,
            "axis": None
        }
        self.outputs = {
            'Out': unique,
            'Indices': indices,
            "Inverse": inverse,
            "Counts": counts,
        }


class TestUniqueV2OpAxis1(TestUniqueV2Op):
    def init_config(self):
        self.inputs = {'X': np.random.random((3, 8, 8)).astype('float64')}
        unique, indices, inverse, counts = np.unique(
            self.inputs['X'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=1)
        self.attrs = {
            "return_index": True,
            "return_inverse": True,
            "return_counts": True,
            "axis": [1]
        }
        self.outputs = {
            'Out': unique,
            'Indices': indices,
            "Inverse": inverse,
            "Counts": counts,
        }


class TestUniqueAPI(unittest.TestCase):
    def test_api_out(self):
        paddle.disable_static()
        x_data = x_data = np.random.randint(0, 10, (120))
        x = paddle.to_tensor(x_data)
        out = paddle.unique(x)
        expected_out = np.unique(x_data)
        self.assertTrue((out.numpy() == expected_out).all(), True)
        paddle.enable_static()

    def test_api_attr(self):
        paddle.disable_static()
        x_data = np.random.random((3, 5, 5)).astype("float32")
        x = paddle.to_tensor(x_data)
        out, index, inverse, counts = paddle.unique(
            x,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0)
        np_out, np_index, np_inverse, np_counts = np.unique(
            x_data,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0)
        self.assertTrue((out.numpy() == np_out).all(), True)
        self.assertTrue((index.numpy() == np_index).all(), True)
        self.assertTrue((inverse.numpy() == np_inverse).all(), True)
        self.assertTrue((counts.numpy() == np_counts).all(), True)
        paddle.enable_static()


class TestUniqueError(unittest.TestCase):
    def test_input_dtype(self):
        def test_x_dtype():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                x = paddle.data(name='x', shape=[10, 10], dtype='float16')
                result = paddle.unique(x)

            self.assertRaises(TypeError, test_x_dtype)

    def test_attr(self):
        x = paddle.data(name='x', shape=[10, 10], dtype='float64')

        def test_return_index():
            result = paddle.unique(x, return_index=0)

        self.assertRaises(TypeError, test_return_index)

        def test_return_inverse():
            result = paddle.unique(x, return_inverse='s')

        self.assertRaises(TypeError, test_return_inverse)

        def test_return_counts():
            result = paddle.unique(x, return_counts=3)

        self.assertRaises(TypeError, test_return_counts)

        def test_axis():
            result = paddle.unique(x, axis='12')

        self.assertRaises(TypeError, test_axis)


if __name__ == "__main__":
    unittest.main()
