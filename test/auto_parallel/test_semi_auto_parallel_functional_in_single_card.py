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

import unittest

import numpy as np
import paddle
import paddle.distributed as dist


class TestSemiAutoParallelFunctionalInSingleCard(unittest.TestCase):
    def test_tensor_copy_to(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(
            dense_tensor,
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]),
        )
        dist_tensor._copy_to(paddle.CPUPlace(), True)

    def test_tensor_strides(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dense_tensor = dense_tensor.reshape([20, 10])
        dist_tensor = dist.shard_tensor(
            dense_tensor,
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]),
        )
        strides = dist_tensor.get_strides()
        is_contiguous = dist_tensor.is_contiguous()
        dist_tensor = dist_tensor.contiguous()

    def test_tensor_uva(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        place = paddle.CPUPlace()
        np_value = np.random.random(size=[10, 30]).astype('float32')
        dense_tensor = paddle.to_tensor(np_value, place=place)
        dist_tensor = dist.shard_tensor(
            dense_tensor,
            place=place,
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]),
        )
        dist_tensor._uva()

if __name__ == "__main__":
    unittest.main()
