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
import sys

import paddle
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.dygraph.parallel import ParallelEnv


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

    def test_create_process_group_bkcl(self):
        with _test_eager_guard():
            device_id = paddle.distributed.ParallelEnv().dev_id
            paddle.set_device('xpu:%d' % device_id)

            pg = init_process_group()
            sys.stdout.write(
                "rank {}: size {} name {}\n".format(
                    pg.rank(), pg.size(), pg.name()
                )
            )
            sys.stdout.write(
                "rank {}: test new group api ok\n".format(pg.rank())
            )

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

            sys.stdout.write(
                "rank {}: test allreduce sum api ok\n".format(pg.rank())
            )

            # TODO
            # test allreduce max/min/prod

            # test broadcast
            # rank 0
            x = np.random.random(self.shape).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random(self.shape).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            broadcast_result = paddle.assign(tensor_x)
            if pg.rank() == 0:
                # XPU don't support event query by now, so just use sync op here
                task = dist.broadcast(tensor_x, 0)
                paddle.device.xpu.synchronize()
                assert np.array_equal(broadcast_result, tensor_x)
            else:
                task = dist.broadcast(tensor_y, 0)
                paddle.device.xpu.synchronize()
                assert np.array_equal(broadcast_result, tensor_y)

            sys.stdout.write(
                "rank {}: test broadcast api ok\n".format(pg.rank())
            )

            # test barrier
            # rank 0
            if pg.rank() == 0:
                pg.barrier(device_id)
            # rank 1
            else:
                task = pg.barrier(device_id)
                task.wait()

            sys.stdout.write("rank {}: test barrier api ok\n".format(pg.rank()))

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
                paddle.device.xpu.synchronize()
            # rank 1
            else:
                tensor_out_list = [
                    paddle.empty_like(tensor_x),
                    paddle.empty_like(tensor_x),
                ]
                task = dist.all_gather(tensor_out_list, tensor_y)
                paddle.device.xpu.synchronize()
                tensor_out = paddle.concat(tensor_out_list)
            out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
            out_2 = paddle.slice(
                tensor_out, [0], [out_shape[0] // 2], [out_shape[0]]
            )
            assert np.array_equal(tensor_x, out_1)
            assert np.array_equal(tensor_y, out_2)
            sys.stdout.write(
                "rank {}: test allgather api ok\n".format(pg.rank())
            )

            if pg.rank() == 0:
                task = pg.all_gather(tensor_x, tensor_out)
                task.wait()
                paddle.device.xpu.synchronize()
            # rank 1
            else:
                tensor_out_list = []
                task = dist.all_gather(tensor_out_list, tensor_y)
                paddle.device.xpu.synchronize()
                tensor_out = paddle.concat(tensor_out_list)
            out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
            out_2 = paddle.slice(
                tensor_out, [0], [out_shape[0] // 2], [out_shape[0]]
            )
            assert np.array_equal(tensor_x, out_1)
            assert np.array_equal(tensor_y, out_2)
            sys.stdout.write(
                "rank {}: test allgather api2 ok\n".format(pg.rank())
            )


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
