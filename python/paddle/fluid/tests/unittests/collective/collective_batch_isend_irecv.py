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
import paddle.distributed as dist


class TestCollectiveBatchIsendIrecv(unittest.TestCase):

    def setUp(self):
        dist.init_parallel_env()
        paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})

    def test_collective_batch_isend_irecv(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        send_t = paddle.arange(2) + rank
        # paddle.tensor([0, 1])  # Rank-0
        # paddle.tensor([1, 2])  # Rank-1
        recv_t = paddle.empty(shape=[2], dtype=send_t.dtype)
        send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)
        recv_op = dist.P2POp(dist.irecv, recv_t,
                             (rank - 1 + world_size) % world_size)
        tasks = dist.batch_isend_irecv([send_op, recv_op])

        for task in tasks:
            task.wait()

        if rank == 0:
            np.testing.assert_allclose(recv_t.numpy(), [1, 2])
        elif rank == 1:
            np.testing.assert_allclose(recv_t.numpy(), [0, 1])


if __name__ == '__main__':
    unittest.main()
