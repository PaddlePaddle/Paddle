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


class TestHingeLossOp(OpTest):

    def setUp(self):
        self.op_type = 'hinge_loss'
        samples_num = 100
        logits = np.random.uniform(-10, 10, (samples_num, 1)).astype('float32')
        labels = np.random.randint(0, 2, (samples_num, 1)).astype('float32')

        self.inputs = {
            'Logits': logits,
            'Labels': labels,
        }
        loss = np.maximum(1.0 - (2 * labels - 1) * logits, 0)
        self.outputs = {'Loss': loss}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Logits'], 'Loss')


if __name__ == '__main__':
    unittest.main()
