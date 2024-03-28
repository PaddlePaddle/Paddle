# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist


class TestItemApiForSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()
        paddle.seed(self._seed)
        np.random.seed(self._seed)

    def test_item_api(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        a = paddle.rand(shape=[6, 8])
        b = dist.shard_tensor(a, mesh, [dist.Shard(0)])
        np.testing.assert_equal(b.item(0, 0), a[0][0].item())
        np.testing.assert_equal(b.item(3, 5), a[3][5].item())

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_item_api()


if __name__ == '__main__':
    TestItemApiForSemiAutoParallel().run_test_case()
