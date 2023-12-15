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

import numpy as np

import paddle
from paddle.distributed import fleet


class TestNewGroupAPI:
    def __init__(self):
        paddle.distributed.init_parallel_env()
        topo = fleet.CommunicateTopology(
            ["data", "model", "sharding", "pipe"], [2, 1, 1, 1]
        )
        self.hcg = fleet.HybridCommunicateGroup(topo)

        d1 = np.array([1, 2, 3])
        d2 = np.array([2, 3, 4])
        self.tensor1 = paddle.to_tensor(d1)
        self.tensor2 = paddle.to_tensor(d2)

    def test_all(self):
        topo = self.hcg.topology()
        global_rank = self.hcg.get_data_parallel_rank()

        dp_rank = self.hcg.get_data_parallel_rank()
        dp_gp = self.hcg.get_data_parallel_group()
        dp_world_size = self.hcg.get_data_parallel_world_size()
        dp_src_rank = self.hcg.get_data_parallel_group_src_rank()
        np.testing.assert_array_equal(dp_world_size, 2)
        np.testing.assert_array_equal(dp_src_rank, 0)

        mp_rank = self.hcg.get_model_parallel_rank()
        mp_gp = self.hcg.get_model_parallel_group()
        mp_world_size = self.hcg.get_model_parallel_world_size()
        mp_src_rank = self.hcg.get_model_parallel_group_src_rank()
        np.testing.assert_array_equal(mp_world_size, 1)

        tmp = np.array([0, 0, 0])
        result = paddle.to_tensor(tmp)
        paddle.distributed.scatter(
            result,
            [self.tensor2, self.tensor1],
            src=dp_src_rank,
            group=dp_gp,
            sync_op=True,
        )
        if dp_rank == 0:
            np.testing.assert_array_equal(result, self.tensor2)
        elif dp_rank == 1:
            np.testing.assert_array_equal(result, self.tensor1)
        print("test scatter api ok")

        paddle.distributed.broadcast(result, src=1, group=dp_gp, sync_op=True)
        np.testing.assert_array_equal(result, self.tensor1)
        print("test broadcast api ok")

        paddle.distributed.reduce(
            result, dst=dp_src_rank, group=dp_gp, sync_op=True
        )
        if dp_rank == 0:
            np.testing.assert_array_equal(
                result, paddle.add(self.tensor1, self.tensor1)
            )
        elif dp_rank == 1:
            np.testing.assert_array_equal(result, self.tensor1)
        print("test reduce api ok")

        paddle.distributed.all_reduce(result, sync_op=True)
        np.testing.assert_array_equal(
            result,
            paddle.add(paddle.add(self.tensor1, self.tensor1), self.tensor1),
        )
        print("test all_reduce api ok")

        paddle.distributed.wait(result, dp_gp, use_calc_stream=True)
        paddle.distributed.wait(result, dp_gp, use_calc_stream=False)
        print("test wait api ok")

        result = []
        paddle.distributed.all_gather(
            result, self.tensor1, group=dp_gp, sync_op=True
        )
        np.testing.assert_array_equal(result[0], self.tensor1)
        np.testing.assert_array_equal(result[1], self.tensor1)
        print("test all_gather api ok")

        paddle.distributed.barrier(group=dp_gp)
        print("test barrier api ok")


if __name__ == "__main__":
    gpt = TestNewGroupAPI()
    gpt.test_all()
