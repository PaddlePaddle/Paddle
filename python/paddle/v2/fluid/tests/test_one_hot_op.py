#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import math
from op_test import OpTest
import paddle.v2.fluid.core as core


class TestOneHotOp(OpTest):
    def setUp(self):
        self.op_type = 'one_hot'
        depth = 10
        dimension = 12
        x_lod = [[0, 4, 5, 8, 11]]
        x = [np.random.randint(0, depth - 1) for i in xrange(x_lod[0][-1])]
        x = np.array(x).astype('int').reshape([x_lod[0][-1], 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]),
                              depth)).astype('float32')

        for i in xrange(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'depth': depth, 'dtype': int(core.DataType.FP32)}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
