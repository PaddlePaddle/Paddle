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


def modified_huber_loss_forward(val):
    if val < -1:
        return -4. * val
    elif val < 1:
        return (1. - val) * (1. - val)
    else:
        return 0.


class TestModifiedHuberLossOp(OpTest):

    def setUp(self):
        self.op_type = 'modified_huber_loss'
        samples_num = 100

        x_np = np.random.uniform(-2., 2., (samples_num, 1)).astype('float32')
        y_np = np.random.choice([0, 1], samples_num).reshape(
            (samples_num, 1)).astype('float32')
        product_res = x_np * (2. * y_np - 1.)
        # keep away from the junction of piecewise function
        for pos, val in np.ndenumerate(product_res):
            while abs(val - 1.) < 0.05:
                x_np[pos] = np.random.uniform(-2., 2.)
                y_np[pos] = np.random.choice([0, 1])
                product_res[pos] = x_np[pos] * (2 * y_np[pos] - 1)
                val = product_res[pos]

        self.inputs = {'X': x_np, 'Y': y_np}
        loss = np.vectorize(modified_huber_loss_forward)(product_res)

        self.outputs = {
            'IntermediateVal': product_res.astype('float32'),
            'Out': loss.reshape((samples_num, 1)).astype('float32')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
