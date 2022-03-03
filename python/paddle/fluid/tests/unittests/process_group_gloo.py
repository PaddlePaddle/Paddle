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
import datetime
from datetime import timedelta
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.dygraph.parallel import ParallelEnv


class TestProcessGroupFp32(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float32"
        self.shape = (2, 10, 5)

    def test_create_process_group_gloo(self):
        with _test_eager_guard():
            nranks = ParallelEnv().nranks
            rank = ParallelEnv().local_rank
            is_master = True if rank == 0 else False
            store = paddle.fluid.core.TCPStore("127.0.0.1", 6172, is_master,
                                               nranks, datetime.timedelta(0))
            gloo_store = paddle.fluid.core.GlooStore(store)
            opt = paddle.fluid.core.GlooOptions()
            pg = paddle.fluid.core.ProcessGroupGloo(gloo_store, rank, nranks)

            # test allreduce sum
            # rank 0
            paddle.device.set_device('cpu')
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            sum_result = x + y
            if rank == 0:
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

            if rank == 0:
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
            if rank == 0:
                task = pg.broadcast(tensor_x, 0)
                task.synchronize()
                assert task.is_completed()
                assert np.array_equal(broadcast_result, tensor_x)
            else:
                task = pg.broadcast(tensor_y, 0)
                task.synchronize()
                assert task.is_completed()
                assert np.array_equal(broadcast_result, tensor_y)
            print("test broadcast api ok")


if __name__ == "__main__":
    unittest.main()
