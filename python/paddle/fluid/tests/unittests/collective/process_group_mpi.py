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

import paddle
from paddle.fluid import core
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard
from paddle.distributed.collective import Group
from paddle.distributed.collective import _default_group_name
from paddle.distributed.collective import _set_group_map
from paddle.distributed.collective import _set_group_map_by_name
from paddle.distributed.collective import _set_group_map_backend
from paddle.fluid.framework import _set_expected_place
import paddle.distributed as dist
import ctypes

ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)


def init_process_group(strategy=None):
    gid = 0
    pg = core.ProcessGroupMPI.create([], gid)
    rank = pg.get_rank()
    world_size = pg.get_world_size()

    # support CPU
    place = core.CPUPlace()
    _set_expected_place(place)

    group = Group(rank,
                  world_size,
                  id=0,
                  ranks=list(range(world_size)),
                  pg=pg,
                  name=_default_group_name)
    _set_group_map_by_name(_default_group_name, group)
    _set_group_map(gid, group)
    _set_group_map_backend(group, "mpi")

    return group


def test_allreduce_sum(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    sum_result = tensor_x + tensor_y
    if pg.rank() == 0:
        task = dist.all_reduce(tensor_x)
        assert np.array_equal(tensor_x, sum_result)
    else:
        task = dist.all_reduce(tensor_y)
        assert np.array_equal(tensor_y, sum_result)
    print("test allreduce sum api ok")


def test_allreduce_max(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    max_result = paddle.maximum(tensor_x, tensor_y)

    if pg.rank() == 0:
        task = dist.all_reduce(tensor_x,
                               dist.ReduceOp.MAX,
                               use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_x, max_result)
    else:
        task = dist.all_reduce(tensor_y,
                               dist.ReduceOp.MAX,
                               use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_y, max_result)
    print("test allreduce max api ok")


def test_allreduce_min(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    min_result = paddle.minimum(tensor_x, tensor_y)

    if pg.rank() == 0:
        task = dist.all_reduce(tensor_x,
                               dist.ReduceOp.MIN,
                               use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_x, min_result)
    else:
        task = dist.all_reduce(tensor_y,
                               dist.ReduceOp.MIN,
                               use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_y, min_result)
    print("test allreduce min api ok")


def test_allreduce_prod(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    prod_result = np.multiply(x, y)

    if pg.rank() == 0:
        task = dist.all_reduce(tensor_x,
                               dist.ReduceOp.PROD,
                               use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_x, prod_result)
    else:
        task = dist.all_reduce(tensor_y,
                               dist.ReduceOp.PROD,
                               use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_y, prod_result)
    print("test allreduce prod api ok")


def test_broadcast(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    broadcast_result = paddle.assign(tensor_x)
    if pg.rank() == 0:
        task = dist.broadcast(tensor_x, 0, use_calc_stream=False)
        task.synchronize()
        assert task.is_completed()
        assert np.array_equal(broadcast_result, tensor_x)
    else:
        task = dist.broadcast(tensor_y, 0)
        assert np.array_equal(broadcast_result, tensor_y)
    print("test broadcast api ok")


def test_barrair(pg):
    # rank 0
    if pg.rank() == 0:
        dist.barrier()
    # rank 1
    else:
        task = pg.barrier()
        task.wait()
    print("test barrier api ok\n")


def test_allgather(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    y = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    tensor_y = paddle.to_tensor(y)
    out_shape = list(shape)
    out_shape[0] *= 2
    out = np.random.random(out_shape).astype(dtype)
    tensor_out = paddle.to_tensor(out)
    if pg.rank() == 0:
        task = pg.all_gather(tensor_x, tensor_out)
        task.wait()
    # rank 1
    else:
        tensor_out_list = [
            paddle.empty_like(tensor_x),
            paddle.empty_like(tensor_x)
        ]
        task = dist.all_gather(tensor_out_list, tensor_y, use_calc_stream=False)
        tensor_out = paddle.concat(tensor_out_list)
    out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
    out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
    assert np.array_equal(tensor_x, out_1)
    assert np.array_equal(tensor_y, out_2)
    print("test allgather api ok\n")

    if pg.rank() == 0:
        task = pg.all_gather(tensor_x, tensor_out)
        task.wait()
    # rank 1
    else:
        tensor_out_list = []
        task = dist.all_gather(tensor_out_list, tensor_y, use_calc_stream=False)
        tensor_out = paddle.concat(tensor_out_list)
    out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
    out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
    assert np.array_equal(tensor_x, out_1)
    assert np.array_equal(tensor_y, out_2)
    print("test allgather api2 ok\n")


def test_all2all(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    y = np.random.random(shape).astype(dtype)
    out1 = np.random.random(shape).astype(dtype)
    out2 = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    tensor_y = paddle.to_tensor(y)
    tensor_out1 = paddle.to_tensor(out1)
    tensor_out2 = paddle.to_tensor(out2)
    raw_tensor_x_2 = paddle.slice(tensor_x, [0], [shape[0] // 2], [shape[0]])
    raw_tensor_y_1 = paddle.slice(tensor_y, [0], [0], [shape[0] // 2])
    if pg.rank() == 0:
        task = pg.alltoall(tensor_x, tensor_out1)
        task.wait()
    # rank 1
    else:
        in_1, in_2 = paddle.split(tensor_y, 2)
        out_1, out_2 = paddle.split(tensor_out2, 2)
        out_tensor_list = [out_1, out_2]
        task = dist.alltoall([in_1, in_2], out_tensor_list)
        tensor_out2 = paddle.concat(out_tensor_list)
    out1_2 = paddle.slice(tensor_out1, [0], [shape[0] // 2], [shape[0]])
    out2_1 = paddle.slice(tensor_out2, [0], [0], [shape[0] // 2])
    if pg.rank() == 0:
        assert np.array_equal(out1_2.numpy(), raw_tensor_y_1.numpy())
    else:
        assert np.array_equal(out2_1, raw_tensor_x_2)
    print("test alltoall api ok\n")

    x = np.random.random(shape).astype(dtype)
    y = np.random.random(shape).astype(dtype)
    out1 = np.random.random(shape).astype(dtype)
    out2 = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    tensor_y = paddle.to_tensor(y)
    tensor_out1 = paddle.to_tensor(out1)
    tensor_out2 = paddle.to_tensor(out2)
    raw_tensor_x_2 = paddle.slice(tensor_x, [0], [shape[0] // 2], [shape[0]])
    raw_tensor_y_1 = paddle.slice(tensor_y, [0], [0], [shape[0] // 2])
    if pg.rank() == 0:
        task = pg.alltoall(tensor_x, tensor_out1)
        task.wait()
    # rank 1
    else:
        in_1, in_2 = paddle.split(tensor_y, 2)
        out_1, out_2 = paddle.split(tensor_out2, 2)
        out_tensor_list = []
        task = dist.alltoall([in_1, in_2], out_tensor_list)
        tensor_out2 = paddle.concat(out_tensor_list)
    out1_2 = paddle.slice(tensor_out1, [0], [shape[0] // 2], [shape[0]])
    out2_1 = paddle.slice(tensor_out2, [0], [0], [shape[0] // 2])
    if pg.rank() == 0:
        assert np.array_equal(out1_2.numpy(), raw_tensor_y_1.numpy())
    else:
        assert np.array_equal(out2_1, raw_tensor_x_2)
    print("test alltoall api2 ok\n")


def test_reduce_sum(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    y = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    tensor_y = paddle.to_tensor(y)
    sum_result = tensor_x + tensor_y
    if pg.rank() == 0:
        task = dist.reduce(tensor_x, 0, use_calc_stream=True)
    # rank 1
    else:
        task = dist.reduce(tensor_y, 0, use_calc_stream=False)
        task.wait()
    if pg.rank() == 0:
        assert np.array_equal(tensor_x, sum_result)
    print("test reduce sum api ok\n")


def test_reduce_max(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    max_result = paddle.maximum(tensor_x, tensor_y)

    if pg.rank() == 0:
        task = dist.reduce(tensor_x,
                           0,
                           dist.ReduceOp.MAX,
                           use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_x, max_result)
    else:
        task = dist.reduce(tensor_y,
                           0,
                           dist.ReduceOp.MAX,
                           use_calc_stream=False)
        task.wait()
    print("test reduce max api ok")


def test_reduce_min(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    min_result = paddle.minimum(tensor_x, tensor_y)

    if pg.rank() == 0:
        task = dist.reduce(tensor_x,
                           0,
                           dist.ReduceOp.MIN,
                           use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_x, min_result)
    else:
        task = dist.reduce(tensor_y,
                           0,
                           dist.ReduceOp.MIN,
                           use_calc_stream=False)
        task.wait()
    print("test reduce min api ok")


def test_reduce_prod(pg, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    prod_result = np.multiply(x, y)

    if pg.rank() == 0:
        task = dist.reduce(tensor_x,
                           0,
                           dist.ReduceOp.PROD,
                           use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_x, prod_result)
    else:
        task = dist.reduce(tensor_y,
                           0,
                           dist.ReduceOp.PROD,
                           use_calc_stream=False)
        task.wait()
    print("test reduce prod api ok")


def test_scatter(pg, shape, dtype):
    # rank 0
    in_shape = list(shape)
    in_shape[0] *= 2
    x = np.random.random(in_shape).astype(dtype)
    y = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    tensor_y = paddle.to_tensor(y)
    if pg.rank() == 0:
        in_1, in_2 = paddle.split(tensor_x, 2)
        task = dist.scatter(tensor_y, [in_1, in_2], 0, use_calc_stream=True)
    # rank 1
    else:
        task = dist.scatter(tensor_y, [], 0, use_calc_stream=False)
        task.wait()
    out1 = paddle.slice(tensor_x, [0], [0], [shape[0]])
    out2 = paddle.slice(tensor_x, [0], [shape[0]], [shape[0] * 2])
    if pg.rank() == 0:
        assert np.array_equal(tensor_y, out1)
    else:
        assert np.array_equal(tensor_y, out2)
    print("test scatter api ok\n")


def test_send_recv(pg, sub_group, shape, dtype):
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    if pg.rank() == 0:
        task = dist.send(tensor_x, 1, group=sub_group, use_calc_stream=False)
        task.wait()
    elif pg.rank() == 1:
        task = dist.recv(tensor_y, 0, group=sub_group, use_calc_stream=False)
        task.wait()
        assert np.array_equal(tensor_y, tensor_x)

    print("test send api ok")

    # test send min
    # rank 0
    x = np.random.random(shape).astype(dtype)
    tensor_x = paddle.to_tensor(x)
    # rank 1
    y = np.random.random(shape).astype(dtype)
    tensor_y = paddle.to_tensor(y)

    if pg.rank() == 0:
        task = dist.send(tensor_x, 1, group=sub_group, use_calc_stream=True)
    elif pg.rank() == 1:
        task = dist.recv(tensor_y, 0, group=sub_group, use_calc_stream=True)
        assert np.array_equal(tensor_y, tensor_x)

    print("test send api ok")


class TestProcessGroup(unittest.TestCase):

    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float32"
        self.shape = (2, 10, 5)

    def test_create_process_group_mpi(self):
        with _test_eager_guard():
            group = init_process_group()
            pg = group.process_group

            # test allreduce sum
            test_allreduce_sum(pg, self.shape, self.dtype)

            # test allreduce max
            test_allreduce_max(pg, self.shape, self.dtype)

            # test allreduce min
            test_allreduce_min(pg, self.shape, self.dtype)

            # test allreduce prod
            test_allreduce_prod(pg, self.shape, self.dtype)

            # test broadcast
            test_broadcast(pg, self.shape, self.dtype)

            # test barrier
            test_barrair(pg)

            # test allgather
            test_allgather(pg, self.shape, self.dtype)

            # test alltoall
            test_all2all(pg, self.shape, self.dtype)

            # test Reduce
            test_reduce_sum(pg, self.shape, self.dtype)

            # test reduce max
            test_reduce_max(pg, self.shape, self.dtype)

            # test reduce min
            test_reduce_min(pg, self.shape, self.dtype)

            # test reduce product
            test_reduce_prod(pg, self.shape, self.dtype)

            # test Scatter
            test_scatter(pg, self.shape, self.dtype)

            # test send recv.
            test_send_recv(pg, group, self.shape, self.dtype)


if __name__ == "__main__":
    unittest.main()
