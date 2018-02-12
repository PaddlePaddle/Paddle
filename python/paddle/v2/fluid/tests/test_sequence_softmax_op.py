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
from test_softmax_op import stable_softmax


class TestSequenceSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = "sequence_softmax"
        x = np.random.uniform(0.1, 1, (11, 1)).astype("float32")
        lod = [[0, 4, 5, 8, 11]]

        out = np.zeros((11, 1)).astype("float32")
        for i in range(4):
            sub_x = x[lod[0][i]:lod[0][i + 1], :]
            sub_x = sub_x.reshape(1, lod[0][i + 1] - lod[0][i])
            sub_out = stable_softmax(sub_x)
            out[lod[0][i]:lod[0][i + 1], :] = sub_out.reshape(
                lod[0][i + 1] - lod[0][i], 1)

        self.inputs = {"X": (x, lod)}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", max_relative_error=0.01)


if __name__ == "__main__":
    unittest.main()
