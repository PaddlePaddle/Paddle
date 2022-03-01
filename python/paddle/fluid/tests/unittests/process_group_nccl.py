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

from __future__ import print_function

import unittest
import random
import numpy as np
import os
import shutil

import paddle
from paddle.fluid import core
from datetime import timedelta
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.dygraph.parallel import ParallelEnv

ProcessGroupStrategy = core.ProcessGroupStrategy


def init_process_group(strategy=None):
    # this will remove
    if strategy is None:
        strategy = ProcessGroupStrategy()
        strategy.nranks = ParallelEnv().nranks
        strategy.local_rank = ParallelEnv().local_rank
        strategy.trainer_endpoints = ParallelEnv().trainer_endpoints
        strategy.current_endpoint = ParallelEnv().current_endpoint
    if strategy.nranks < 2:
        return

    pg_group = core.ProcessGroupNCCL(strategy, strategy.local_rank,
                                     strategy.nranks)

    return pg_group


class TestProcessGroupFp32(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float32"
        self.shape = (2, 10, 5)

    def test_create_process_group_nccl(self):
        with _test_eager_guard():
            paddle.set_device('gpu:%d' %
                              paddle.distributed.ParallelEnv().dev_id)

            pg = init_process_group()
            print("rank:", pg.rank(), "size:", pg.size(), "name:", pg.name())
            print("test new group api ok")

            # test allreduce sum
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            sum_result = tensor_x + tensor_y
            if pg.rank() == 0:
                task = pg.allreduce(tensor_x)
                task.wait()
                assert np.array_equal(tensor_x, sum_result)
            else:
                task = pg.allreduce(tensor_y)
                task.wait()
                assert np.array_equal(tensor_y, sum_result)

            print("test allreduce sum api ok")

            # test allreduce max
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            max_result = paddle.maximum(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = pg.allreduce(tensor_x, core.ReduceOp.MAX)
                task.wait()
                assert np.array_equal(tensor_x, max_result)
            else:
                task = pg.allreduce(tensor_y, core.ReduceOp.MAX)
                task.wait()
                assert np.array_equal(tensor_y, max_result)

            print("test allreduce max api ok")

            # test broadcast
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            broadcast_result = paddle.assign(tensor_x)
            if pg.rank() == 0:
                task = pg.broadcast(tensor_x, 0)
                task.synchronize()
                paddle.device.cuda.synchronize()
                assert task.is_completed()
                assert np.array_equal(broadcast_result, tensor_x)
            else:
                task = pg.broadcast(tensor_y, 0)
                task.synchronize()
                paddle.device.cuda.synchronize()
                assert task.is_completed()
                assert np.array_equal(broadcast_result, tensor_y)

            print("test broadcast api ok")


class TestProcessGroupFp16(TestProcessGroupFp32):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float16"
        self.shape = (4, 20, 20)


if __name__ == "__main__":
    unittest.main()
