#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from scipy.special import expit, erf

from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

class TestAssignPosAPI(unittest.TestCase):
    def init(self):
        self.dtype = 'int64'
        self.shape = [10, 10] # (batch_size * seq_len, d_model)
        self.topK = 2
        self.num_expert = 2
        self.world_size = 2
        self.tot_expert = self.num_expert * self.world_size

    def setUp(self):
        self.init()
        self.gate_idx = np.random.randint(low=0, high=self.tot_expert-1, \
                                            size=(self.shape[0], self.topK))
        local_expert_count = np.zeros(self.tot_expert).astype(self.dtype)
        self.gate = self.gate_idx.flatten()
        nums = len(self.gate)
        for i in range(nums):
            local_expert_count[self.gate[i]] += 1
        self.lec_cum = np.zeros(len(local_expert_count), dtype=np.int64)
        self.lec_cum[0] = local_expert_count[0]
        for i in range(1, len(local_expert_count)):
            self.lec_cum[i] = local_expert_count[i] + self.lec_cum[i-1]
        self.lec_cum_np = self.lec_cum.copy()
        self.pos_np = np.zeros((self.lec_cum[-1], ))
        for i in range(0, len(self.gate)):
            idx = self.gate[i]
            p = self.lec_cum_np[idx]
            self.lec_cum_np[idx] -= 1
            self.pos_np[p-1] = i
        self.place = [paddle.CUDAPlace(0)]

    # def test_static_api(self):
    #     paddle.enable_static()

    #     def run(place):
    #         with paddle.static.program_guard(paddle.static.Program()):
    #             X = paddle.fluid.data('X', self.shape, dtype=self.dtype)
    #             out = paddle.expm1(X)
    #             exe = paddle.static.Executor(place)
    #             res = exe.run(feed={'X': self.x})
    #         for r in res:
    #             self.assertEqual(np.allclose(self.out_ref, r), True)

    #     for place in self.place:
    #         run(place)

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            gate_idx = paddle.to_tensor(self.gate_idx, dtype="int32")
            lec_cum = paddle.to_tensor(self.lec_cum, dtype="int64")
            pos = paddle.distributed.utils.assign_pos(x=gate_idx, cum_count=lec_cum)
            self.assertEqual(np.allclose(self.pos_np, pos), True)
            paddle.enable_static()

            # print("gate_idx: ", self.gate_idx)
            # print("lec_cum: ", self.lec_cum)
            # print("pos np: ", self.pos_np)
            # print("pos my: ", pos)

        for place in self.place:
            run(place)

    # def test_errors(self):
    #     paddle.enable_static()
    #     with paddle.static.program_guard(paddle.static.Program()):
    #         X = paddle.fluid.data('X', self.shape, dtype='int32')
    #         self.assertRaises(TypeError, paddle.expm1, X)
    #     # The input dtype must be float16, float32, float64.

if __name__ == "__main__":
    unittest.main()