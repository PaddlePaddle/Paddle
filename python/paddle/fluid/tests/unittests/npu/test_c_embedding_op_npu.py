#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()
SEED = 2021


def get_c_embedding(start, end, table, ids):
    index = ids.flatten()
    input_mask = (index < start) | (index >= end)
    masked_input = index - start
    masked_input[input_mask] = 0
    output = table[masked_input]
    output[input_mask] = 0.0
    return output


class TestCast(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = fluid.NPUPlace(0)
        self.dtype = np.float32
        np.random.seed(SEED)
        self.op_type = "c_embedding"
        table = np.ones([10, 20]).astype(self.dtype)
        ids = np.random.randint(low=0, high=20, size=(6, 8)).astype(np.int32)
        self.start_index = 2
        self.end_index = self.start_index + 10

        self.inputs = {'W': table, 'Ids': ids}
        np_out = get_c_embedding(self.start_index, self.end_index, table, ids)
        self.outputs = {'Out': np_out.reshape(6, 8, 20)}
        self.attrs = {'start_index': self.start_index}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['W'], 'Out')


if __name__ == '__main__':
    unittest.main()
