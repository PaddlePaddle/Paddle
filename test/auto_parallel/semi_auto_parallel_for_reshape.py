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

from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist
from paddle.distributed import Replicate, Shard

"""
test for reshape
"""


class TestReshapeSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()

    def check_dim_mapping(self, output, expected_dim_mapping):
        assert (
            output.dist_attr.dims_mapping == expected_dim_mapping
        ), f"{output.dist_attr.dims_mapping}  vs {expected_dim_mapping}"

    def test_reshape_forward(self):
        shape = [200, 30]
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        input = dist.shard_tensor(
            paddle.rand(shape=[10, 20, 30]),
            mesh,
            [Shard(0), Replicate(), Replicate()],
        )
        input.stop_gradient = False
        output = paddle.reshape(input, shape)
        output.backward()

        self.check_dim_mapping(output, [0, -1])
        self.check_dim_mapping(input.grad, [0, -1, -1])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")
        self.test_reshape_forward()


if __name__ == '__main__':
    TestReshapeSemiAutoParallel().run_test_case()
