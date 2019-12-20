#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
        self.dtype = np.float64
        self.init_dtype_type()
        batch_size = 12
        feet_dim = 10
        cluster_num = 8
        self.attrs = {}
        self.attrs['cluster_num'] = cluster_num
        self.attrs['lambda'] = 0.1
        self.config()
        self.attrs['need_update'] = self.need_update
        labels = np.random.randint(cluster_num, size=batch_size, dtype='int64')
        feat = np.random.random((batch_size, feet_dim)).astype(np.float64)
        centers = np.random.random((cluster_num, feet_dim)).astype(np.float64)
        var_sum = np.zeros((cluster_num, feet_dim), dtype=np.float64)
        centers_select = centers[labels]
        output = feat - centers_select
        diff_square = np.square(output).reshape(batch_size, feet_dim)
        loss = 0.5 * np.sum(diff_square, axis=1).reshape(batch_size, 1)
        cout = []
        for i in range(cluster_num):
            cout.append(0)
        for i in range(batch_size):
            cout[labels[i]] += 1
            var_sum[labels[i]] += output[i]
        for i in range(cluster_num):
            var_sum[i] /= (1 + cout[i])
        var_sum *= 0.1
        result = centers + var_sum
        rate = np.array([0.1]).astype(np.float64)

        self.inputs = {
            'X': feat,
            'Label': labels,
            'Centers': centers,
            'CenterUpdateRate': rate
        }

        if self.need_update == True:
            self.outputs = {
                'SampleCenterDiff': output,
                'Loss': loss,
                'CentersOut': result
            }
        else:
            self.outputs = {
                'SampleCenterDiff': output,
                'Loss': loss,
                'CentersOut': centers
            }

    def config(self):
        self.need_update = True

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Loss')


class TestCenterLossOpNoUpdate(TestCenterLossOp):
    def config(self):
        self.need_update = False


if __name__ == "__main__":
    unittest.main()
