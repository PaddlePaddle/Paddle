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
from op_test import OpTest, randomize_probability


class TestBprLossOp1(OpTest):
    """Test BprLoss with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "bpr_loss"
        batch_size = 40
        class_num = 5
        X = randomize_probability(batch_size, class_num, dtype='float64')
        label = np.random.randint(0, class_num, (batch_size, 1), dtype="int64")
        bpr_loss_result = []
        for i in range(batch_size):
            sum = 0.0
            for j in range(class_num):
                if j == label[i][0]:
                    continue
                sum += (-np.log(1.0 + np.exp(X[i][j] - X[i][label[i][0]])))
            bpr_loss_result.append(-sum / (class_num - 1))
        bpr_loss = np.asmatrix([[x] for x in bpr_loss_result], dtype="float64")
        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": bpr_loss}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.001)


if __name__ == "__main__":
    unittest.main()
