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

from __future__ import division
from __future__ import print_function

import unittest

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle import framework


class TestCollectiveReduceScatter(unittest.TestCase):

    def setUp(self):
        dist.init_parallel_env()
        paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})

    def test_collective_reduce_scatter_sum(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            t1 = paddle.to_tensor([0, 1])
            t2 = paddle.to_tensor([2, 3])
        else:
            t1 = paddle.to_tensor([4, 5])
            t2 = paddle.to_tensor([6, 7])

        input_list = [t1, t2]

        output = paddle.empty(shape=[2], dtype=input_list[0].dtype)
        dist.reduce_scatter(output, input_list)

        if rank == 0:
            np.testing.assert_allclose(output.numpy(), [4, 6])
        elif rank == 1:
            np.testing.assert_allclose(output.numpy(), [8, 10])

    def test_collective_reduce_scatter_max(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            t1 = paddle.to_tensor([0, 1], dtype="float16")
            t2 = paddle.to_tensor([2, 3], dtype="float16")
        else:
            t1 = paddle.to_tensor([4, 5], dtype="float16")
            t2 = paddle.to_tensor([6, 7], dtype="float16")

        input_list = [t1, t2]

        output = paddle.empty(shape=[2], dtype=input_list[0].dtype)
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.MAX)

        if rank == 0:
            np.testing.assert_allclose(output.numpy(), [4, 5])
        elif rank == 1:
            np.testing.assert_allclose(output.numpy(), [6, 7])

    def test_collective_reduce_scatter_base(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        input = paddle.arange(4) + rank
        # [0, 1, 2, 3]  # Rank-0
        # [1, 2, 3, 4]  # Rank-1

        output = paddle.empty(shape=[2], dtype=input.dtype)
        task = paddle.distributed.collective._reduce_scatter_base(output,
                                                                  input,
                                                                  sync_op=False)

        task.wait()

        if rank == 0:
            np.testing.assert_allclose(output.numpy(), [1, 3])
        elif rank == 1:
            np.testing.assert_allclose(output.numpy(), [5, 7])


if __name__ == '__main__':
    unittest.main()
