# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestSemiAutoParallelTensorPatchMethods:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_dist_tensor_item(self):
        np_array = np.random.randn(10, 10)
        gpu_tensor = paddle.to_tensor(
            np_array, device=paddle.CUDAPlace(dist.get_rank())
        )
        cpu_tensor = paddle.to_tensor(np_array, device=paddle.CPUPlace())

        dist_tensor_r = dist.shard_tensor(
            gpu_tensor, self._mesh, [dist.Replicate()]
        )
        dist_tensor_s = dist.shard_tensor(
            gpu_tensor, self._mesh, [dist.Shard(0)]
        )
        cpu_dist_tensor = dist.shard_tensor(
            cpu_tensor, self._mesh, [dist.Shard(0)]
        )

        for i in range(10):
            for j in range(10):
                np.testing.assert_equal(
                    np_array[i, j], dist_tensor_r.item(i, j)
                )
                np.testing.assert_equal(
                    np_array[i, j], dist_tensor_s.item(i, j)
                )
                np.testing.assert_equal(
                    np_array[i, j], cpu_dist_tensor.item(i, j)
                )

    def run_test_case(self):
        self.test_dist_tensor_item()


if __name__ == '__main__':
    TestSemiAutoParallelTensorPatchMethods().run_test_case()
