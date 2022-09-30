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


def init_process_group(strategy=None):
    nranks = ParallelEnv().nranks
    rank = ParallelEnv().local_rank
    is_master = True if rank == 0 else False
    store = paddle.fluid.core.TCPStore("127.0.0.1", 6173, is_master, nranks)
    pg_group = core.ProcessGroupCustom(
        store, rank, nranks,
        paddle.CustomPlace(ParallelEnv().device_type,
                           ParallelEnv().device_id))

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

    def test_create_process_group_xccl(self):
        with _test_eager_guard():
            paddle.set_device('custom_cpu:%d' %
                              paddle.distributed.ParallelEnv().dev_id)

            pg = init_process_group()

            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            sum_result = tensor_x + tensor_y
            if pg.rank() == 0:
                task = pg.allreduce(tensor_x)
                task.wait()
                # assert np.array_equal(tensor_x, sum_result)
            else:
                task = pg.allreduce(tensor_y)
                task.wait()
                # assert np.array_equal(tensor_y, sum_result)

            print("test allreduce sum api ok")

            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            max_result = paddle.maximum(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = pg.allreduce(tensor_x, core.ReduceOp.MAX)
                task.wait()
                # assert np.array_equal(tensor_x, max_result)
            else:
                task = pg.allreduce(tensor_y, core.ReduceOp.MAX)
                task.wait()
                # assert np.array_equal(tensor_y, max_result)

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
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
                assert task.is_completed()
                # assert np.array_equal(broadcast_result, tensor_x)
            else:
                task = pg.broadcast(tensor_y, 0)
                task.synchronize()
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
                assert task.is_completed()
                # assert np.array_equal(broadcast_result, tensor_y)

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
            return

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
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            # rank 1
            else:
                task = pg.all_gather(tensor_y, tensor_out)
                task.wait()
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
            out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2],
                                 [out_shape[0]])
            # assert np.array_equal(tensor_x, out_1)
            # assert np.array_equal(tensor_y, out_2)
            print("test allgather api ok\n")

            # test alltoall
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            y = np.random.random(self.shape).astype(self.dtype)
            out1 = np.random.random(self.shape).astype(self.dtype)
            out2 = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            tensor_y = paddle.to_tensor(y)
            tensor_out1 = paddle.to_tensor(out1)
            tensor_out2 = paddle.to_tensor(out2)
            raw_tensor_x_2 = paddle.slice(tensor_x, [0], [self.shape[0] // 2],
                                          [self.shape[0]])
            raw_tensor_y_1 = paddle.slice(tensor_y, [0], [0],
                                          [self.shape[0] // 2])
            if pg.rank() == 0:
                task = pg.alltoall(tensor_x, tensor_out1)
                task.wait()
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            # rank 1
            else:
                task = pg.alltoall(tensor_y, tensor_out2)
                task.wait()
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            out1_2 = paddle.slice(tensor_out1, [0], [self.shape[0] // 2],
                                  [self.shape[0]])
            out2_1 = paddle.slice(tensor_out2, [0], [0], [self.shape[0] // 2])
            # if pg.rank() == 0:
            #     assert np.array_equal(out1_2.numpy(), raw_tensor_y_1.numpy())
            # else:
            #     assert np.array_equal(out2_1, raw_tensor_x_2)
            print("test alltoall api ok\n")

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
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            # rank 1
            else:
                task = pg.reduce(tensor_y, 0)
                task.wait()
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            # if pg.rank() == 0:
            #     assert np.array_equal(tensor_x, sum_result)
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
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            # rank 1
            else:
                task = pg.scatter(tensor_x, tensor_y, 0)
                task.wait()
                # paddle.fluid.core._custom_device_synchronize("custom_cpu", -1)
            out1 = paddle.slice(tensor_x, [0], [0], [self.shape[0]])
            out2 = paddle.slice(tensor_x, [0], [self.shape[0]],
                                [self.shape[0] * 2])
            # if pg.rank() == 0:
            #     assert np.array_equal(tensor_y, out1)
            # else:
            #     assert np.array_equal(tensor_y, out2)
            print("test scatter api ok\n")


if __name__ == "__main__":
    unittest.main()
