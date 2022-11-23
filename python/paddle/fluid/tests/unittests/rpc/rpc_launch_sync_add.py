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

import numpy as np
import paddle
import paddle.distributed as dist

paddle.device.set_device("cpu")


def add(a, b):
    a = paddle.to_tensor(a, dtype="float32")
    b = paddle.to_tensor(b, dtype="float32")
    res = paddle.add(a, b).numpy()
    return res


def rpc_add(to, args):
    res = dist.rpc.rpc_sync(to, add, args=args)
    return res


def worker_name(rank):
    return "worker{}".format(rank)


def main():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.rpc.init_rpc(worker_name(rank))
    if rank == 0:
        mmap_data1 = np.memmap(
            "rpc_launch_data1.npy",
            dtype=np.float32,
            mode="r",
            shape=(10 * world_size, 100),
        )
        mmap_data2 = np.memmap(
            "rpc_launch_data2.npy",
            dtype=np.float32,
            mode="r",
            shape=(10 * world_size, 100),
        )
        mmap_out = np.memmap(
            "rpc_launch_result.npy",
            dtype=np.float32,
            mode="w+",
            shape=(10 * world_size, 100),
        )
        for i in range(world_size):
            a = mmap_data1[i * 10:(i + 1) * 10, :]
            b = mmap_data2[i * 10:(i + 1) * 10, :]
            args = (a, b)
            out = rpc_add(worker_name(i), args)
            mmap_out[i * 10:(i + 1) * 10, :] = out[:]
    dist.rpc.shutdown()


if __name__ == "__main__":
    main()
