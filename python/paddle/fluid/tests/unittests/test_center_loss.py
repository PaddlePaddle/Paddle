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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid


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


class BadInputTestCenterLoss(unittest.TestCase):

    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                data = [[1, 2, 3, 4], [5, 6, 7, 8]]
                label = fluid.layers.data(name='label',
                                          shape=[2, 1],
                                          dtype='int32')
                res = fluid.layers.center_loss(
                    data,
                    label,
                    num_classes=1000,
                    alpha=0.2,
                    param_attr=fluid.initializer.Xavier(uniform=False),
                    update_center=True)

            self.assertRaises(TypeError, test_bad_x)

            def test_bad_y():
                data = fluid.layers.data(name='data',
                                         shape=[2, 32],
                                         dtype='float32')
                label = [[2], [3]]
                res = fluid.layers.center_loss(
                    data,
                    label,
                    num_classes=1000,
                    alpha=0.2,
                    param_attr=fluid.initializer.Xavier(uniform=False),
                    update_center=True)

            self.assertRaises(TypeError, test_bad_y)

            def test_bad_alpha():
                data = fluid.layers.data(name='data2',
                                         shape=[2, 32],
                                         dtype='float32',
                                         append_batch_size=False)
                label = fluid.layers.data(name='label2',
                                          shape=[2, 1],
                                          dtype='int32',
                                          append_batch_size=False)
                alpha = fluid.layers.data(name='alpha',
                                          shape=[1],
                                          dtype='int64',
                                          append_batch_size=False)
                res = fluid.layers.center_loss(
                    data,
                    label,
                    num_classes=1000,
                    alpha=alpha,
                    param_attr=fluid.initializer.Xavier(uniform=False),
                    update_center=True)

            self.assertRaises(TypeError, test_bad_alpha)


if __name__ == "__main__":
    unittest.main()
