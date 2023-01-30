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

<<<<<<< HEAD
import sys
import unittest

import numpy as np

import paddle.fluid.core as core
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
import sys
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

sys.path.append("../")
from op_test import OpTest
from test_softmax_op import stable_softmax


class TestSequenceSoftmaxOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "sequence_softmax"
        self.use_cudnn = False
        self.init_op_type()
        self.dtype = "float32" if core.is_compiled_with_rocm() else "float64"
        x = np.random.uniform(0.1, 1, (110, 1)).astype(self.dtype)
        self.init_lod()
        out = np.zeros((110, 1)).astype(self.dtype)
        offset = 0
        for i in range(len(self.lod[0])):
<<<<<<< HEAD
            if self.lod[0][i] == 0:
                continue
            sub_x = x[offset : offset + self.lod[0][i], :]
            sub_x = sub_x.reshape(1, self.lod[0][i])
            sub_out = stable_softmax(sub_x)
            out[offset : offset + self.lod[0][i], :] = sub_out.reshape(
                self.lod[0][i], 1
            )
=======
            if (self.lod[0][i] == 0):
                continue
            sub_x = x[offset:offset + self.lod[0][i], :]
            sub_x = sub_x.reshape(1, self.lod[0][i])
            sub_out = stable_softmax(sub_x)
            out[offset:offset + self.lod[0][i], :] = sub_out.reshape(
                self.lod[0][i], 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            offset += self.lod[0][i]

        self.inputs = {"X": (x, self.lod)}
        self.outputs = {"Out": out}
        self.attrs = {
            'use_cudnn': self.use_cudnn,
        }

    def init_lod(self):
        self.lod = [[40, 10, 30, 30]]

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
            self.check_grad_with_place(place, ["X"], "Out")
        else:
            self.check_grad(["X"], "Out")


# ----------------cudnn Sequencesoftmax----------------
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestSequenceSoftmaxCUDNNOp(TestSequenceSoftmaxOp):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSequenceSoftmaxCUDNNOp(TestSequenceSoftmaxOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_op_type(self):
        self.use_cudnn = True


class TestSequenceSoftmaxOpSeqLen0Case0(TestSequenceSoftmaxOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_lod(self):
        self.lod = [[40, 0, 40, 30]]


class TestSequenceSoftmaxOpSeqLen0Case1(TestSequenceSoftmaxOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_lod(self):
        self.lod = [[0, 40, 70, 0]]


class TestSequenceSoftmaxOpSeqLen0Case2(TestSequenceSoftmaxOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_lod(self):
        self.lod = [[0, 0, 0, 110]]


if __name__ == "__main__":
    unittest.main()
