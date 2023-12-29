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

import paddle
import paddle.distributed as dist
from paddle import LazyGuard


class TestSemiAutoParallelLazyInit:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_replicate(self):
        paddle.distributed.auto_parallel.parallel_manual_seed(self._seed)
        with LazyGuard():
            linear = paddle.nn.Linear(10, 10)
            linear.weight = dist.shard_tensor(
                linear.weight, self._mesh, [dist.Replicate()]
            )
            linear.bias = dist.shard_tensor(
                linear.bias, self._mesh, [dist.Replicate()]
            )
        for param in linear.parameters():
            assert not param._is_initialized()
            param.initialize()
            assert param._is_initialized()

        local_weight_md5 = linear.weight._local_value()._md5sum()
        mesh0 = dist.ProcessMesh([0], dim_names=["x"])
        mesh1 = dist.ProcessMesh([1], dim_names=["x"])
        tmp = paddle.distributed.auto_parallel.api.dtensor_from_local(
            linear.weight._local_value(),
            mesh0 if dist.get_rank() == 0 else mesh1,
            [dist.Replicate()],
        )
        tmp = dist.reshard(
            tmp, mesh1 if dist.get_rank() == 0 else mesh0, [dist.Replicate()]
        )
        tmp_md5 = tmp._local_value()._md5sum()
        assert local_weight_md5 == tmp_md5

    def run_test_case(self):
        self.test_replicate()


if __name__ == '__main__':
    TestSemiAutoParallelLazyInit().run_test_case()
