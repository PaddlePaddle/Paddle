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

import random
import unittest

import numpy as np

import paddle
from paddle.fluid import core
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
        nranks = ParallelEnv().nranks
        rank = ParallelEnv().local_rank
        is_master = True if rank == 0 else False
        store = paddle.fluid.core.TCPStore(
            "127.0.0.1", 6272, is_master, nranks, 30
        )
        pg = paddle.fluid.core.ProcessGroupGloo.create(store, rank, nranks)

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
            np.testing.assert_equal(tensor_x, sum_result)
        else:
            task = pg.allreduce(tensor_y)
            task.wait()
            np.testing.assert_equal(tensor_y, sum_result)

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
            assert np.array_equal(broadcast_result, tensor_x)
        else:
            task = pg.broadcast(tensor_y, 0)
            assert np.array_equal(broadcast_result, tensor_y)
        print("test broadcast api ok")

        # test barrier
        # rank 0
        if pg.rank() == 0:
            task = pg.barrier()
            task.wait()
        # rank 1
        else:
            task = pg.barrier()
            task.wait()

        print("test barrier api ok\n")

        # test allgather
        # rank 0
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        out_shape = list(self.shape)
        out_shape[0] *= 2
        out = np.random.random(out_shape).astype(self.dtype)
        tensor_out = paddle.to_tensor(out)
        if pg.rank() == 0:
            task = pg.all_gather(tensor_x, tensor_out)
            task.wait()
            paddle.device.cuda.synchronize()
        # rank 1
        else:
            task = pg.all_gather(tensor_y, tensor_out)
            task.wait()
        out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
        out_2 = paddle.slice(
            tensor_out, [0], [out_shape[0] // 2], [out_shape[0]]
        )
        assert np.array_equal(tensor_x, out_1)
        assert np.array_equal(tensor_y, out_2)
        print("test allgather api ok\n")

        # test Reduce
        # rank 0
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        sum_result = tensor_x + tensor_y
        if pg.rank() == 0:
            task = pg.reduce(tensor_x, 0)
            task.wait()
        # rank 1
        else:
            task = pg.reduce(tensor_y, 0)
            task.wait()
        if pg.rank() == 0:
            assert np.array_equal(tensor_x, sum_result)
        print("test reduce sum api ok\n")

        # test Scatter
        # rank 0
        in_shape = list(self.shape)
        in_shape[0] *= 2
        x = np.random.random(in_shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        if pg.rank() == 0:
            task = pg.scatter(tensor_x, tensor_y, 0)
            task.wait()
        # rank 1
        else:
            task = pg.scatter(tensor_x, tensor_y, 0)
            task.wait()
        out1 = paddle.slice(tensor_x, [0], [0], [self.shape[0]])
        out2 = paddle.slice(tensor_x, [0], [self.shape[0]], [self.shape[0] * 2])
        if pg.rank() == 0:
            assert np.array_equal(tensor_y, out1)
        else:
            assert np.array_equal(tensor_y, out2)
        print("test scatter api ok\n")


if __name__ == "__main__":
    unittest.main()
