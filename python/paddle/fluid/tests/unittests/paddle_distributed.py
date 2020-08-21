# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import sys

backend = sys.argv[1]
place = None
if backend == "gloo":
    place = fluid.CPUPlace()
elif backend == "nccl":
    place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
with fluid.dygraph.guard(place=place):
    rank = fluid.dygraph.ParallelEnv().local_rank
    paddle.distributed.init_process_group(backend, 100, 2, rank)
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        np_data = np.array([[4, 5, 6], [4, 5, 6]])
    else:
        np_data = np.array([[1, 2, 3], [1, 2, 3]])
    data = paddle.to_tensor(np_data)
    paddle.distributed.broadcast(data, 1)
    out = data.numpy()
    assert (np.allclose(np.array([[1, 2, 3], [1, 2, 3]]), out))
    print("broadcast passed.")
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        np_data = np.array([[4, 5, 6], [4, 5, 6]])
    else:
        np_data = np.array([[1, 2, 3], [1, 2, 3]])
    data = paddle.to_tensor(np_data)
    paddle.distributed.all_reduce(data)
    out = data.numpy()
    assert (np.allclose(np.array([[5, 7, 9], [5, 7, 9]]), out))
    print("allreduce passed.")
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        np_data = np.array([[4, 5, 6], [4, 5, 6]])
    else:
        np_data = np.array([[1, 2, 3], [1, 2, 3]])
    data = paddle.to_tensor(np_data)
    paddle.distributed.reduce(data, 0)
    out = data.numpy()
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        assert (np.allclose(np.array([[5, 7, 9], [5, 7, 9]]), out))
    print("reduce passed.")
    tensor_list = []
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
        np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        paddle.distributed.all_gather(tensor_list, data1)
    else:
        np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
        np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        out = paddle.distributed.all_gather(tensor_list, data2)
    out1 = tensor_list[0].numpy()
    out2 = tensor_list[1].numpy()
    assert (np.allclose(np.array([[4, 5, 6], [4, 5, 6]]), out1))
    assert (np.allclose(np.array([[1, 2, 3], [1, 2, 3]]), out2))
    print("all_gather passed.")
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        np_data1 = np.array([7, 8, 9])
        np_data2 = np.array([10, 11, 12])
    else:
        np_data1 = np.array([1, 2, 3])
        np_data2 = np.array([4, 5, 6])
    data1 = paddle.to_tensor(np_data1)
    data2 = paddle.to_tensor(np_data2)
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        paddle.distributed.scatter(data1, src=1)
    else:
        paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
    out = data1.numpy()
    if fluid.dygraph.ParallelEnv().local_rank == 0:
        assert (np.allclose(np.array([1, 2, 3]), out))
    else:
        assert (np.allclose(np.array([4, 5, 6]), out))
    print("scatter passed.")
