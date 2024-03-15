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
import unittest

import numpy as np

import paddle
import paddle.distributed as dist


class TestDistTensorLocalAPI(unittest.TestCase):
    def setUp(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seed = 2023
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        paddle.seed(self._seed)
        np.random.seed(self._seed)

    def run_test_dist_tensor_with_local_tensor_shard(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        global_tensor0 = paddle.rand([4, 10])
        local_tensor_list0 = paddle.split(
            global_tensor0, num_or_sections=2, axis=0
        )
        local_tensor0 = local_tensor_list0[dist.get_rank()]

        dist_tensor_shard0 = dist.auto_parallel.api.dtensor_from_local(
            local_tensor0,
            mesh=self._mesh,
            placements=[dist.Shard(0)],
        )

        np.testing.assert_equal(
            dist_tensor_shard0._local_value().numpy(),
            local_tensor0.numpy(),
        )
        np.testing.assert_equal(
            dist_tensor_shard0.numpy(),
            global_tensor0.numpy(),
        )

        self.assertEqual(dist_tensor_shard0.shape, [4, 10])

        global_tensor1 = paddle.rand([2, 20])
        local_tensor_list1 = paddle.split(
            global_tensor1, num_or_sections=2, axis=1
        )
        local_tensor1 = local_tensor_list1[dist.get_rank()]
        dist_tensor_shard1 = dist.auto_parallel.api.dtensor_from_local(
            local_tensor1,
            mesh=self._mesh,
            placements=[dist.Shard(1)],
        )

        np.testing.assert_equal(
            dist_tensor_shard1._local_value().numpy(),
            local_tensor1.numpy(),
        )
        np.testing.assert_equal(
            dist_tensor_shard1.numpy(),
            global_tensor1.numpy(),
        )

        self.assertEqual(dist_tensor_shard1.shape, [2, 20])

    def run_test_dist_tensor_with_local_tensor_replicate(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        local_tensor = paddle.rand([2, 10])
        dist_tensor = dist.auto_parallel.api.dtensor_from_local(
            local_tensor,
            mesh=self._mesh,
            placements=[dist.Replicate()],
        )

        np.testing.assert_equal(
            dist_tensor._local_value().numpy(),
            local_tensor.numpy(),
        )
        np.testing.assert_equal(
            dist_tensor.numpy(),
            local_tensor.numpy(),
        )

        self.assertEqual(dist_tensor.shape, [2, 10])

    def test_case(self):
        self.run_test_dist_tensor_with_local_tensor_shard()
        self.run_test_dist_tensor_with_local_tensor_replicate()


if __name__ == "__main__":
    unittest.main()
