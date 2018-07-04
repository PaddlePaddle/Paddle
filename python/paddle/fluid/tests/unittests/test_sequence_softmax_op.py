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
import paddle.fluid.core as core


class TestSequenceSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = "sequence_softmax"
        self.use_cudnn = False
        self.init_op_type()

        x = np.random.uniform(0.1, 1, (11, 1)).astype("float32")
        lod = [[4, 1, 3, 3]]

        out = np.zeros((11, 1)).astype("float32")
        offset = 0
        for i in range(len(lod[0])):
            sub_x = x[offset:offset + lod[0][i], :]
            sub_x = sub_x.reshape(1, lod[0][i])
            sub_out = stable_softmax(sub_x)
            out[offset:offset + lod[0][i], :] = sub_out.reshape(lod[0][i], 1)
            offset += lod[0][i]

        self.inputs = {"X": (x, lod)}
        self.outputs = {"Out": out}
        self.attrs = {'use_cudnn': self.use_cudnn, }

    def init_op_type(self):
        pass

    def test_check_output(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ["X"], "Out", max_relative_error=0.01)
        else:
            self.check_grad(["X"], "Out", max_relative_error=0.01)


# ----------------cudnn Sequencesoftmax----------------
class TestSequenceSoftmaxCUDNNOp(TestSequenceSoftmaxOp):
    def init_op_type(self):
        self.use_cudnn = True


if __name__ == "__main__":
    unittest.main()
