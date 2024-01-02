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
        p = paddle.rand(shape=[10, 10])
        q = dist.shard_tensor(p, self._mesh, [dist.Replicate()])
        for i in range(10):
            for j in range(10):
                np.testing.assert_equal(p.numpy()[i, j], q.item(i, j))

        q = dist.shard_tensor(p, self._mesh, [dist.Shard(0)])
        for i in range(10):
            for j in range(10):
                np.testing.assert_equal(p.numpy()[i, j], q.item(i, j))

    def run_test_case(self):
        self.test_dist_tensor_item()


if __name__ == '__main__':
    TestSemiAutoParallelTensorPatchMethods().run_test_case()
