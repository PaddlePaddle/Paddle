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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core


class TestCenterLossOp(OpTest):
    def setUp(self):
        self.op_type = "center_loss"
        self.dtype = np.float32
        self.init_dtype_type()
        self.attrs = {}
        self.attrs['cluster_num'] = 3
        self.attrs['lambda'] = 0.1
        self.attrs['need_update'] = True
        labels = np.array([[0], [2], [0]]).astype(np.int64)

        feat = np.array([[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0],
                         [1.0, 2.0, 3.0, 4.0]]).astype(np.float32)
        centers = np.array([[2.0, 3.0, 4.0, 5.0], [5.0, 6.0, 7.0, 8.0],
                            [7.0, 8.0, 9.0, 10.0],
                            [13.0, 14.0, 15.0, 16.0]]).astype(np.float32)

        result = np.array([[1.93333, 2.93333, 3.93333, 4.93333],
                           [5.0, 6.0, 7.0, 8.0], [7.1, 8.1, 9.1, 10.1],
                           [13.0, 14.0, 15.0, 16.0]]).astype(np.float32)

        output = np.array([[-1.0, -1.0, -1.0, -1.0], [2.0, 2.0, 2.0, 2.0],
                           [-1.0, -1.0, -1.0, -1.0]]).astype(np.float32)
        loss = np.array([[2.0], [8.0], [2.0]]).astype(np.float32)
        rate = np.array([0.1]).astype(np.float32)

        self.inputs = {
            'X': feat,
            'Label': labels,
            'Centers': centers,
            'CenterUpdateRate': rate
        }
        self.outputs = {
            'CentersDiff': output,
            'Loss': loss,
            'CentersOut': result
        }

    def init_dtype_type(self):
        pass

    def test_check_grad(self):
        self.check_grad(['X'], 'Loss')

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
