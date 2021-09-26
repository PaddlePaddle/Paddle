# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np


class TestPruneGateByCapacityOp(unittest.TestCase):
    def init_test_case(self):
        self.n_expert = 8
        self.n_worker = 1
        self.gate_idx = np.array([1, 3, 3, 3, 3, 2, 1, 1])
        self.expert_count = np.array([0, 3, 1, 3, 0, 0, 0, 0])

    def setUp(self):
        self.init_test_case()
        self.place = paddle.CUDAPlace(0)

    def compute_case(self):
        tot_expert = self.n_expert * self.n_worker
        expert_count = [0] * tot_expert
        gate_idx = [0] * tot_expert

        for i in range(len(gate_idx)):
            gate_idx[i] = random.randint(1, 5) % tot_expert
            expert_count[gate_idx[i]] += 1
        for i in range(tot_expert):
            expert_count[i] >>= 1
        return gate_idx, expert_count

    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                gate_idx_tensor = paddle.static.data(
                    'GateIdx', shape=self.gate_idx.shape, dtype="int32")
                expert_count_tensor = paddle.static.data(
                    'ExpertCount', shape=self.expert_count.shape, dtype="int64")
                out = paddle.distributed.utils.prune_gate_by_capacity(
                    gate_idx_tensor, expert_count_tensor, self.n_expert,
                    self.n_worker)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={
                    'GateIdx': self.gate_idx.astype("int32"),
                    'ExpertCount': self.expert_count.astype("int64"),
                },
                              fetch_list=out)

            print("---------------------------------")
            print("static_api:")
            print("gate_idx:", self.gate_idx)
            print("expert_count:", self.expert_count)
            print("new_gate_idx:", res)
            print("----------------------------------")

        run(self.place)

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            gate_idx_tensor = paddle.to_tensor(self.gate_idx.astype('int32'))
            expert_count_tensor = paddle.to_tensor(
                self.expert_count.astype('int64'))
            out = paddle.distributed.utils.prune_gate_by_capacity(
                gate_idx_tensor, expert_count_tensor, self.n_expert,
                self.n_worker)

            print("---------------------------------")
            print("dygraph_api:")
            print("gate_idx:", self.gate_idx)
            print("expert_count:", self.expert_count)
            print("new_gate_idx:", out)
            print("----------------------------------")

        run(self.place)


if __name__ == '__main__':
    unittest.main()
