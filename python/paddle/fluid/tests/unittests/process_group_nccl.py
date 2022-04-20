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
import paddle.distributed as dist


def init_process_group(strategy=None):
    nranks = ParallelEnv().nranks
    rank = ParallelEnv().local_rank
    is_master = True if rank == 0 else False
    pg_group = dist.init_parallel_env()

    return pg_group.process_group


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
                task = dist.all_reduce(tensor_x)
                assert np.array_equal(tensor_x, sum_result)
            else:
                task = dist.all_reduce(tensor_y)
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
                task = dist.all_reduce(
                    tensor_x, dist.ReduceOp.MAX, use_calc_stream=False)
                task.wait()
                assert np.array_equal(tensor_x, max_result)
            else:
                task = dist.all_reduce(
                    tensor_y, dist.ReduceOp.MAX, use_calc_stream=False)
                task.wait()
                assert np.array_equal(tensor_y, max_result)

            print("test allreduce max api ok")

            # test allreduce min
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            min_result = paddle.minimum(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = dist.all_reduce(
                    tensor_x, dist.ReduceOp.MIN, use_calc_stream=False)
                task.wait()
                assert np.array_equal(tensor_x, min_result)
            else:
                task = dist.all_reduce(
                    tensor_y, dist.ReduceOp.MIN, use_calc_stream=False)
                task.wait()
                assert np.array_equal(tensor_y, min_result)

            print("test allreduce min api ok")

            # test broadcast
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            broadcast_result = paddle.assign(tensor_x)
            if pg.rank() == 0:
                task = dist.broadcast(tensor_x, 0, use_calc_stream=False)
                task.synchronize()
                paddle.device.cuda.synchronize()
                assert task.is_completed()
                assert np.array_equal(broadcast_result, tensor_x)
            else:
                task = dist.broadcast(tensor_y, 0)
                paddle.device.cuda.synchronize()
                assert np.array_equal(broadcast_result, tensor_y)

            print("test broadcast api ok")

            # test barrier
            # rank 0
            if pg.rank() == 0:
                dist.barrier()
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
                tensor_out_list = [
                    paddle.empty_like(tensor_x), paddle.empty_like(tensor_x)
                ]
                task = dist.all_gather(
                    tensor_out_list, tensor_y, use_calc_stream=False)
                paddle.device.cuda.synchronize()
                tensor_out = paddle.concat(tensor_out_list)
            out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
            out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2],
                                 [out_shape[0]])
            assert np.array_equal(tensor_x, out_1)
            assert np.array_equal(tensor_y, out_2)
            print("test allgather api ok\n")

            if pg.rank() == 0:
                task = pg.all_gather(tensor_x, tensor_out)
                task.wait()
                paddle.device.cuda.synchronize()
            # rank 1
            else:
                tensor_out_list = []
                task = dist.all_gather(
                    tensor_out_list, tensor_y, use_calc_stream=False)
                paddle.device.cuda.synchronize()
                tensor_out = paddle.concat(tensor_out_list)
            out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
            out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2],
                                 [out_shape[0]])
            assert np.array_equal(tensor_x, out_1)
            assert np.array_equal(tensor_y, out_2)
            print("test allgather api2 ok\n")

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
            # rank 1
            else:
                in_1, in_2 = paddle.split(tensor_y, 2)
                out_1, out_2 = paddle.split(tensor_out2, 2)
                out_tensor_list = [out_1, out_2]
                task = dist.alltoall([in_1, in_2], out_tensor_list)
                paddle.device.cuda.synchronize()
                tensor_out2 = paddle.concat(out_tensor_list)
            out1_2 = paddle.slice(tensor_out1, [0], [self.shape[0] // 2],
                                  [self.shape[0]])
            out2_1 = paddle.slice(tensor_out2, [0], [0], [self.shape[0] // 2])
            if pg.rank() == 0:
                assert np.array_equal(out1_2.numpy(), raw_tensor_y_1.numpy())
            else:
                assert np.array_equal(out2_1, raw_tensor_x_2)
            print("test alltoall api ok\n")

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
            # rank 1
            else:
                in_1, in_2 = paddle.split(tensor_y, 2)
                out_1, out_2 = paddle.split(tensor_out2, 2)
                out_tensor_list = []
                task = dist.alltoall([in_1, in_2], out_tensor_list)
                paddle.device.cuda.synchronize()
                tensor_out2 = paddle.concat(out_tensor_list)
            out1_2 = paddle.slice(tensor_out1, [0], [self.shape[0] // 2],
                                  [self.shape[0]])
            out2_1 = paddle.slice(tensor_out2, [0], [0], [self.shape[0] // 2])
            if pg.rank() == 0:
                assert np.array_equal(out1_2.numpy(), raw_tensor_y_1.numpy())
            else:
                assert np.array_equal(out2_1, raw_tensor_x_2)
            print("test alltoall api2 ok\n")

            # test Reduce
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            tensor_y = paddle.to_tensor(y)
            sum_result = tensor_x + tensor_y
            if pg.rank() == 0:
                task = dist.reduce(tensor_x, 0, use_calc_stream=True)
                paddle.device.cuda.synchronize()
            # rank 1
            else:
                task = dist.reduce(tensor_y, 0, use_calc_stream=False)
                task.wait()
                paddle.device.cuda.synchronize()
            if pg.rank() == 0:
                assert np.array_equal(tensor_x, sum_result)
            print("test reduce sum api ok\n")

            # test reduce max
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            max_result = paddle.maximum(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = dist.reduce(
                    tensor_x, 0, dist.ReduceOp.MAX, use_calc_stream=False)
                task.wait()
                assert np.array_equal(tensor_x, max_result)
            else:
                task = dist.reduce(
                    tensor_y, 0, dist.ReduceOp.MAX, use_calc_stream=False)
                task.wait()

            print("test reduce max api ok")

            # test reduce min
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            min_result = paddle.minimum(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = dist.reduce(
                    tensor_x, 0, dist.ReduceOp.MIN, use_calc_stream=False)
                task.wait()
                assert np.array_equal(tensor_x, min_result)
            else:
                task = dist.reduce(
                    tensor_y, 0, dist.ReduceOp.MIN, use_calc_stream=False)
                task.wait()

            print("test reduce min api ok")

            # test Scatter
            # rank 0
            in_shape = list(self.shape)
            in_shape[0] *= 2
            x = np.random.random(in_shape).astype(self.dtype)
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            tensor_y = paddle.to_tensor(y)
            if pg.rank() == 0:
                in_1, in_2 = paddle.split(tensor_x, 2)
                task = dist.scatter(
                    tensor_y, [in_1, in_2], 0, use_calc_stream=True)
                #task.wait()
                paddle.device.cuda.synchronize()
            # rank 1
            else:
                task = dist.scatter(tensor_y, [], 0, use_calc_stream=False)
                task.wait()
                paddle.device.cuda.synchronize()
            out1 = paddle.slice(tensor_x, [0], [0], [self.shape[0]])
            out2 = paddle.slice(tensor_x, [0], [self.shape[0]],
                                [self.shape[0] * 2])
            if pg.rank() == 0:
                assert np.array_equal(tensor_y, out1)
            else:
                assert np.array_equal(tensor_y, out2)
            print("test scatter api ok\n")

            # test send min
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            if pg.rank() == 0:
                task = dist.send(tensor_x, 1, use_calc_stream=False)
                task.wait()
            else:
                task = dist.recv(tensor_y, 0, use_calc_stream=False)
                task.wait()
                assert np.array_equal(tensor_y, tensor_x)

            print("test send api ok")

            # test send min
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            if pg.rank() == 0:
                task = dist.send(tensor_x, 1, use_calc_stream=True)
            else:
                task = dist.recv(tensor_y, 0, use_calc_stream=True)
                assert np.array_equal(tensor_y, tensor_x)

            print("test send api ok")


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
