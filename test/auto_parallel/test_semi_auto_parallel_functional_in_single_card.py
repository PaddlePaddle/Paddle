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

import paddle
import paddle.distributed as dist


class TestSemiAutoParallelFunctionalInSingleCard(unittest.TestCase):
    def test_tensor_copy_to(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(
            dense_tensor,
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, 'x']),
        )
        dist_tensor._copy_to(paddle.CPUPlace(), True)


if __name__ == "__main__":
    unittest.main()
