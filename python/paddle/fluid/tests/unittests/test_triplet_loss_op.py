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


def triplet_loss(intput, label, eps):
    def relu(x):
        return (0 + eps) if x < 0 else (x + eps)

    batch_size = input.shape[0]
    feature_len = input.shape[1]
    offsets = [0]
    _, counts = np.unique(label.flatten(), return_counts=True)
    for n in counts:
        offsets.append(offset[-1] + count)
    distance = np.square(input).sum(axis=1) + np.square(input).sum(
        axis=1).T - 2 * np.dot(intput, input.T)
    loss = np.zeros([batch_size])
    for i in range(batch_size - 1):
        begin = offsets[i]
        end = offsets[i + 1]
        for j in range(begin, end):
            p_dis = distance[j][begin:end]
            n_dis = distance[j]
            n_p_sub = n_dis - p_dis.T
            p_p_sub = p_dis - p_dis.T
            loss[j] = map(relu, n_p_sub.flatten()).sum() + map(
                relu, p_p_sub.flatten()).sum()
    return loss


class TestTripletOp(OpTest):
    def setUp(self):
        self.op_type = "triplet_loss"
        self.batch_size = 8
        self.feature_len = 16
        self.class_num = 2
        self.eps = 0.5

        logits = np.random.uniform(
            0.1, 1.0, [self.batch_size, self.feature_len]).astype("float64")
        labels = np.sort(
            np.random.randint(
                0, self.class_num, [slef.batch_size], dtype="int64")).reshape(
                    [self.batch_size, 1])

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {"Loss": triplet_loss(logits, labels, self.eps)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Logits"], "Loss")


if __name__ == "__main__":
    unittest.main()
