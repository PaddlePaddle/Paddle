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


class TestSplitOp(OpTest):
    def setUp(self):
        self._set_op_type()
        axis = 1
        x = np.random.random((4, 5, 6)).astype('float32')
        out = np.split(x, [2, 3], axis)
        self.inputs = {'X': x}
        self.attrs = {'axis': axis, 'sections': [2, 1, 2]}
        self.outputs = {'Out': [('out%d' % i, out[i]) \
            for i in xrange(len(out))]}

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


class TestSplitByrefOp(OpTest):
    def _set_op_type(self):
        self.op_type = "split_byref"


if __name__ == '__main__':
    unittest.main()
