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

from __future__ import print_function, division
import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from op_test import OpTest
import random


class TestShuffleBatchOp(OpTest):
    def setUp(self):
        self.op_type = 'shuffle_batch'
        self.dtype = np.float64
        x = np.array(
            [np.arange(100), np.arange(100)]).astype(self.dtype).reshape(
                [2, 100])
        out = np.array(
            [np.arange(100), np.arange(100)]).astype(self.dtype).reshape(
                [2, 100])
        self.possible_res = [
            np.array([np.arange(100), np.arange(100)]).astype(self.dtype),
        ]
        self.inputs = {'X': x, 'Seed': np.array([1]).astype('int64')}
        self.outputs = {
            'Out': out,
            'ShuffleIdx': np.array([1, 0]).astype('int64'),
            'SeedOut': np.array([1]).astype('int64')
        }
        self.attrs = {'startup_seed': 1}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        for elem in outs:
            if elem.shape == self.outputs['Out'].shape:
                out = elem
                break
        is_equal = [np.all(out == res) for res in self.possible_res]
        self.assertIn(True, is_equal)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
