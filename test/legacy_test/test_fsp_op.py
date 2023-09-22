#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


def fsp_matrix(a, b):
    batch = a.shape[0]
    a_channel = a.shape[1]
    b_channel = b.shape[1]
    h = a.shape[2]
    w = a.shape[3]
    a_t = a.transpose([0, 2, 3, 1])
    a_t = a_t.reshape([batch, h * w, a_channel])
    b_t = b.transpose([0, 2, 3, 1]).reshape([batch, h * w, b_channel])
    a_r = (
        a_t.repeat(b_channel, axis=1)
        .reshape([batch, h * w, b_channel, a_channel])
        .transpose([0, 1, 3, 2])
    )
    b_r = b_t.repeat(a_channel, axis=1).reshape(
        [batch, h * w, a_channel, b_channel]
    )
    return np.mean(a_r * b_r, axis=1)


class TestFSPOp(OpTest):
    def setUp(self):
        self.op_type = "fsp"
        self.initTestCase()

        feature_map_0 = np.random.uniform(0, 10, self.a_shape).astype('float64')
        feature_map_1 = np.random.uniform(0, 10, self.b_shape).astype('float64')

        self.inputs = {'X': feature_map_0, 'Y': feature_map_1}
        self.outputs = {'Out': fsp_matrix(feature_map_0, feature_map_1)}

    def initTestCase(self):
        self.a_shape = (2, 3, 5, 6)
        self.b_shape = (2, 4, 5, 6)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


if __name__ == '__main__':
    unittest.main()
