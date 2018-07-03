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


class TestMarginRankLossOp(OpTest):
    def setUp(self):
        self.op_type = "margin_rank_loss"
        batch_size = 5
        margin = 0.5
        # labels_{i} = {-1, 1}
        label = 2 * np.random.randint(
            0, 2, size=(batch_size, 1)).astype("float32") - 1
        x1 = np.random.random((batch_size, 1)).astype("float32")
        x2 = np.random.random((batch_size, 1)).astype("float32")
        # loss = max(0, -label * (x1 - x2) + margin)
        loss = -label * (x1 - x2) + margin
        loss = np.where(loss > 0, loss, 0)
        act = np.where(loss > 0, 1., 0.)

        self.attrs = {'margin': margin}
        self.inputs = {'Label': label, 'X1': x1, 'X2': x2}
        self.outputs = {'Activated': act, 'Out': loss}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X1", "X2"], "Out")

    def test_check_grad_ignore_x1(self):
        self.check_grad(["X2"], "Out", no_grad_set=set('X1'))

    def test_check_grad_ignore_x2(self):
        self.check_grad(["X1"], "Out", no_grad_set=set('X2'))


if __name__ == '__main__':
    unittest.main()
