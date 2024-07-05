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

import numpy as np

import paddle
import paddle.distributed as dist


class TestCollectiveBatchIsendIrecv(unittest.TestCase):
    def setUp(self):
        dist.init_parallel_env()

    def test_collective_batch_isend_irecv(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        paddle.seed(1024)
        length = 2000

        base = paddle.randint(0, 100, [length])
        send_t = base + rank
        recv_t = paddle.empty(shape=[length], dtype=send_t.dtype)

        send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)
        recv_op = dist.P2POp(
            dist.irecv, recv_t, (rank - 1 + world_size) % world_size
        )
        tasks = dist.batch_isend_irecv([send_op, recv_op])

        for task in tasks:
            task.wait()

        res = recv_t - (rank - 1 + world_size) % world_size
        res = paddle.sum(base - res)
        np.testing.assert_allclose(res.numpy(), [0])


if __name__ == '__main__':
    unittest.main()
