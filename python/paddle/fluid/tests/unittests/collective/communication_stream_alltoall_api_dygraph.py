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

import os
import numpy as np
import paddle
import paddle.distributed as dist
import test_collective_api_base as test_collective_base


class StreamAllToAllTestCase():

    def __init__(self):
        self._sync_op = eval(os.getenv("sync_op"))
        self._use_calc_stream = eval(os.getenv("use_calc_stream"))
        self._backend = os.getenv("backend")
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        if self._backend not in ["nccl", "gloo"]:
            raise NotImplementedError(
                "Only support nccl and gloo as the backend for now.")
        os.environ["PADDLE_DISTRI_BACKEND"] = self._backend

    def run_test_case(self):
        dist.init_parallel_env()

        test_data_list = []
        for seed in self._seeds:
            test_data_list.append(
                test_collective_base.create_test_data(shape=self._shape,
                                                      dtype=self._dtype,
                                                      seed=seed))

        nranks = len(test_data_list)
        data1 = test_data_list[0]
        data2 = test_data_list[1]
        result1 = np.vstack(
            [data1[0:data1.shape[0] // 2, :], data2[0:data2.shape[0] // 2, :]])
        result2 = np.vstack(
            [data1[data1.shape[0] // 2:, :], data2[data2.shape[0] // 2:, :]])

        rank = dist.get_rank()
        tensor = paddle.to_tensor(test_data_list[rank])
        t1, t2 = paddle.split(tensor, nranks, axis=0)

        # case 1: pass an empty tensor list
        empty_tensor_list = []
        task = dist.stream.alltoall(empty_tensor_list, [t1, t2],
                                    sync_op=self._sync_op,
                                    use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        result_tensor_list = np.vstack(empty_tensor_list)
        if rank == 0:
            assert np.allclose(result_tensor_list,
                               result1,
                               rtol=1e-05,
                               atol=1e-05)
        else:
            assert np.allclose(result_tensor_list,
                               result2,
                               rtol=1e-05,
                               atol=1e-05)

        # case 2: pass a pre-sized tensor list
        full_tensor_list = [paddle.empty_like(t1) for _ in test_data_list]
        task = dist.stream.alltoall(full_tensor_list, [t1, t2],
                                    sync_op=self._sync_op,
                                    use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        result_tensor_list = np.vstack(full_tensor_list)
        if rank == 0:
            assert np.allclose(result_tensor_list,
                               result1,
                               rtol=1e-05,
                               atol=1e-05)
        else:
            assert np.allclose(result_tensor_list,
                               result2,
                               rtol=1e-05,
                               atol=1e-05)

        # case 3: pass a pre-sized tensor
        out_tensor = paddle.empty_like(tensor)
        task = dist.stream.alltoall(out_tensor,
                                    tensor,
                                    sync_op=self._sync_op,
                                    use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        if rank == 0:
            assert np.allclose(out_tensor, result1, rtol=1e-05, atol=1e-05)
        else:
            assert np.allclose(out_tensor, result2, rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    StreamAllToAllTestCase().run_test_case()
