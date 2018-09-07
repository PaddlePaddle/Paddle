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


class TestRankLossOp(OpTest):
    def setUp(self):
        self.op_type = "rank_loss"
        batch_size = 5
        # labels_{i} = {0, 1.0} or {0, 0.5, 1.0}
        label = np.random.randint(0, 2, size=(batch_size, 1)).astype("float32")
        left = np.random.random((batch_size, 1)).astype("float32")
        right = np.random.random((batch_size, 1)).astype("float32")
        loss = np.log(1.0 + np.exp(left - right)) - label * (left - right)
        self.inputs = {'Label': label, 'Left': left, 'Right': right}
        self.outputs = {'Out': loss}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Left", "Right"], "Out")

    def test_check_grad_ignore_left(self):
        self.check_grad(["Right"], "Out", no_grad_set=set('Left'))

    def test_check_grad_ignore_right(self):
        self.check_grad(["Left"], "Out", no_grad_set=set('Right'))


if __name__ == '__main__':
    unittest.main()
