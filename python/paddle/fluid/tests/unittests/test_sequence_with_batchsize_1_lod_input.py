# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest
from test_reorder_lod_tensor import convert_to_offset


def compute_seqpool_avg(x, offset, out, pad_value=0.0):
    level = len(offset) - 1
    for i in range(len(offset[level]) - 1):
        if offset[level][i] == offset[level][i + 1]:
            out[i] = pad_value
        else:
            sub_x = x[offset[level][i]:offset[level][i + 1], :]
            out[i] = sub_x.mean(axis=0)


class TestSeqAvgPool(OpTest):
    def set_data(self):
        self.op_type = 'sequence_pool'
        x, lod = self.get_sequence_batch_size_1_input()
        self.inputs = {'X': (x, lod)}
        offset = convert_to_offset(lod)
        out = np.zeros((1, x.shape[1])).astype('float32')
        self.outputs = {'Out': out}
        return x, lod, offset, out

    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "AVERAGE"}
        compute_seqpool_avg(x, offset, out, self.attrs["pad_value"])

    def setUp(self):
        x, lod, offset, out = self.set_data()
        self.compute(x, offset, out)
        if len(offset) > 1:
            self.outputs = {'Out': (out, [lod[0]])}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        # Remove MaxIndex after check_grad is refined.
        out = self.outputs['Out']
        if isinstance(out, tuple): out = out[0]
        self.outputs['MaxIndex'] = \
            np.zeros(out.shape).astype('int32')
        self.check_grad(["X"], "Out", check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
