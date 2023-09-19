# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from op_mapper_test import OpMapperTest

import paddle


class TestCholeskyOp(OpMapperTest):
    def init_input_data(self):
        matrix = self.random([3, 3], "float32")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        self.feed_data = {
            'x': x,
        }

    def set_op_type(self):
        return "cholesky"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        return {'X': [x]}

    def set_op_attrs(self):
        return {"upper": False}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestCholeskyOpCase1(TestCholeskyOp):
    def init_input_data(self):
        matrix = self.random([5, 5], "float64")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        self.feed_data = {
            'x': x,
        }

    def set_op_attrs(self):
        return {"upper": True}


class TestCholeskyOpCase2(TestCholeskyOp):
    def init_input_data(self):
        matrix = self.random([3, 3], "float32")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        x = x * np.ones(shape=(3, 3, 3))
        self.feed_data = {
            'x': x,
        }

    def set_op_attrs(self):
        return {"upper": False}


class TestCholeskyOpCase3(TestCholeskyOp):
    def init_input_data(self):
        matrix = self.random([3, 3], "float32")
        matrix_t = np.transpose(matrix, [1, 0])
        x = np.dot(matrix, matrix_t)
        x = x * np.ones(shape=(2, 3, 3, 3))
        self.feed_data = {
            'x': x,
        }

    def set_op_attrs(self):
        return {"upper": True}


if __name__ == "__main__":
    unittest.main()
