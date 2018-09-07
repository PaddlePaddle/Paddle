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


def norm(input, scale, epsilon):
    s0, s1, s2, s3 = input.shape
    x_square = input * input
    for i in xrange(s0):
        input_batch = input[i:i + 1, :, :, :]
        input_batch = input_batch.reshape(s1, s2 * s3)
        x_square_batch = x_square[i:i + 1, :, :, :]
        x_square_batch = x_square_batch.reshape(s1, s2 * s3)
        square_colsum = x_square_batch.sum(axis=0) + epsilon
        tmp = pow(square_colsum, 0.5)
        tmp = np.reciprocal(tmp)
        tmp_tile = np.tile(tmp, s1)
        tmp_tile = tmp_tile.reshape(s1, s2 * s3)
        scale_tile = np.tile(scale, (1, s2 * s3))
        scale_tile = scale_tile.reshape(s1, s2 * s3)
        out_batch = input_batch * tmp_tile * scale_tile
        out_batch = out_batch.reshape(1, s1, s2, s3)
        if i == 0:
            out = out_batch
        else:
            out = np.concatenate((out, out_batch), 0)
    out.reshape(s0, s1, s2, s3)
    return out


class TestNormOp(OpTest):
    def setUp(self):
        self.op_type = "norm"
        self.init_test_case()
        input = np.random.random(self.shape).astype("float32")
        scale = np.array([10, 10, 10])
        self.inputs = {
            'X': input.astype('float32'),
            'Scale': scale.astype('float32')
        }
        self.attrs = {'epsilon': self.epsilon}
        output = norm(input, scale, self.epsilon)
        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 2, 2]
        self.epsilon = 1e-6


if __name__ == '__main__':
    unittest.main()
