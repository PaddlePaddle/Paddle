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

from __future__ import print_function

import six
import unittest
import numpy as np
from op_test import OpTest


class TestMaskExtract(OpTest):
    def compute(self):
        x_data = np.random.random((5,6)).astype('float32')
        mask = np.array([-1, 2, 3, 0, -1]).reshape([-1, 1]).astype('int64')

        out_len = np.sum(np.where(mask>=0, 1, 0))
        out_shape = list(x_data.shape)
        out_shape[0] = out_len
        
        out = np.zeros(out_shape).astype('float32')
        offset = np.zeros((out_len, 1)).astype('int64')
        ids = np.zeros((out_len, 1)).astype('int64')
        index = 0
        for i in six.moves.range(x_data.shape[0]):
            if (mask[i] >= 0):
                out[index] = x_data[i]
                offset[index] = i
                ids[index] = mask[i]
                index += 1
               
        self.inputs = {'X': x_data, 'Mask': mask}
        self.outputs = {'Out': out, 'Offset': offset, 'Ids': ids}
    
    def setUp(self):
        self.op_type = 'mask_extract'
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")




if __name__ == '__main__':
    unittest.main()
