# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""This is unit test of Test shuffle_batch Op."""

import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from op_test import OpTest
import os
import random


class TestShuffleBatchOpBase(OpTest):

    def gen_random_array(self, shape, low=0, high=1):
        rnd = (high - low) * np.random.random(shape) + low
        return rnd.astype(self.dtype)

    def get_shape(self):
        return (10, 10, 5)

    def _get_places(self):
        # NOTE: shuffle_batch is not supported on Windows
        if os.name == 'nt':
            return [fluid.CPUPlace()]
        return super(TestShuffleBatchOpBase, self)._get_places()

    def setUp(self):
        self.op_type = 'shuffle_batch'
        self.dtype = np.float64
        self.shape = self.get_shape()
        x = self.gen_random_array(self.shape)
        seed = np.random.random_integers(low=10, high=100,
                                         size=(1, )).astype('int64')
        self.inputs = {'X': x, 'Seed': seed}
        self.outputs = {
            'Out': np.array([]).astype(x.dtype),
            'ShuffleIdx': np.array([]).astype('int64'),
            'SeedOut': np.array([]).astype(seed.dtype),
        }
        self.attrs = {'startup_seed': 1}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        x = np.copy(self.inputs['X'])
        y = None
        for out in outs:
            if out.shape == x.shape:
                y = np.copy(out)
                break

        assert y is not None
        sort_x = self.sort_array(x)
        sort_y = self.sort_array(y)
        np.testing.assert_array_equal(sort_x, sort_y)

    def sort_array(self, array):
        shape = array.shape
        new_shape = [-1, shape[-1]]
        arr_list = np.reshape(array, new_shape).tolist()
        arr_list.sort(key=lambda x: x[0])
        return np.reshape(np.array(arr_list), shape)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestShuffleBatchOp2(TestShuffleBatchOpBase):

    def get_shape(self):
        return (4, 30)


if __name__ == '__main__':
    unittest.main()
