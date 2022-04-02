# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.models.moe import utils
from paddle.fluid import core


def count(x, upper_num):
    res = np.zeros((upper_num, )).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res


def limit_by_capacity(expert_count, _capacity, n_worker):
    capacity = np.copy(_capacity)
    old_shape = expert_count.shape
    expert_count = np.reshape(expert_count, (n_worker, len(capacity)))
    output = np.zeros_like(expert_count)
    for wid in range(len(expert_count)):
        for eid in range(len(expert_count[wid])):
            last_cap = capacity[eid]
            if last_cap >= 0:
                capacity[eid] -= expert_count[wid][eid]
            if last_cap >= expert_count[wid][eid]:
                output[wid][eid] = expert_count[wid][eid]
            elif last_cap >= 0:
                output[wid][eid] = last_cap
    return output.reshape(old_shape)


def prune_gate_by_capacity(gate_idx, expert_count, n_expert, n_worker):
    new_gate_idx = np.copy(gate_idx)
    expert_count = np.copy(expert_count)
    for i in range(len(gate_idx)):
        idx = gate_idx[i]
        last_cap = expert_count[idx]
        if last_cap > 0:
            expert_count[idx] -= 1
        else:
            new_gate_idx[i] = -1
    return new_gate_idx


def assert_allclose(output, expected, n_expert):
    c1 = count(output, n_expert)
    c2 = count(expected, n_expert)
    assert np.allclose(c1, c2)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestPruneGateByCapacityAPI1(unittest.TestCase):
    def init_test_case(self):
        self.gate_idx = np.random.randint(
            0, self.n_expert, size=(200, )).astype(self.dtype)
        expert_count = count(self.gate_idx, self.n_expert * self.n_worker)
        capacity = np.random.randint(10, 200, size=(self.n_expert, ))
        self.expert_count = limit_by_capacity(expert_count, capacity,
                                              self.n_worker).astype(self.dtype)
        self.out = prune_gate_by_capacity(self.gate_idx, self.expert_count,
                                          self.n_expert,
                                          self.n_worker).astype(self.dtype)
        self.place = paddle.CUDAPlace(0)

    def setUp(self):
        self.n_expert = 24
        self.n_worker = 2
        self.dtype = "int64"
        self.init_test_case()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            gate_idx_tensor = paddle.static.data(
                'GateIdx', shape=self.gate_idx.shape, dtype="int64")
            expert_count_tensor = paddle.static.data(
                'ExpertCount', shape=self.expert_count.shape, dtype="int64")
            out = utils._prune_gate_by_capacity(gate_idx_tensor,
                                                expert_count_tensor,
                                                self.n_expert, self.n_worker)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={
                'GateIdx': self.gate_idx,
                'ExpertCount': self.expert_count,
            },
                          fetch_list=out)
        assert_allclose(res[0], self.out, self.n_expert)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        gate_idx_tensor = paddle.to_tensor(self.gate_idx)
        expert_count_tensor = paddle.to_tensor(self.expert_count)
        out = utils._prune_gate_by_capacity(
            gate_idx_tensor, expert_count_tensor, self.n_expert, self.n_worker)
        assert_allclose(out.numpy(), self.out, self.n_expert)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestPruneGateByCapacityAPI2(TestPruneGateByCapacityAPI1):
    def setUp(self):
        self.n_expert = 12
        self.n_worker = 1
        self.dtype = "int64"
        self.init_test_case()


if __name__ == '__main__':
    unittest.main()
