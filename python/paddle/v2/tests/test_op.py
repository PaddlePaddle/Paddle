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

import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
import paddle.v2.op as op


class OpTest(unittest.TestCase):
    def test_op(self):
        x = layer.data(name='data', type=data_type.dense_vector(128))
        x = op.exp(x)
        x = op.sqrt(x)
        x = op.reciprocal(x)
        x = op.log(x)
        x = op.abs(x)
        x = op.sigmoid(x)
        x = op.tanh(x)
        x = op.square(x)
        x = op.relu(x)
        y = 1 + x
        y = y + 1
        y = x + y
        y = y - x
        y = y - 2
        y = 2 - y
        y = 2 * y
        y = y * 3
        z = layer.data(name='data_2', type=data_type.dense_vector(1))
        y = y * z
        y = z * y
        y = y + z
        y = z + y
        print layer.parse_network(y)


if __name__ == '__main__':
    unittest.main()
