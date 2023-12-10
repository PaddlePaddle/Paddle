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


class TestReshardSToP:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._other_mesh = dist.ProcessMesh([1, 0], dim_names=["x"])

    def reshard_same_mesh(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")

        a = paddle.ones(self._shape)

        in_placement = [dist.Shard(0)]
        input_tensor = dist.shard_tensor(
            a, mesh=self._mesh, placements=in_placement
        )
        assert input_tensor._local_shape[0] == self._shape[0] // 2

        out = dist.reshard(
            input_tensor,
            mesh=self._mesh,
            placements=[dist.Partial(dist.ReduceType.kRedSum)],
        )

        if dist.get_rank() == 0:
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())
        else:
            zeros = paddle.zeros(self._shape)
            np.testing.assert_equal(out._local_value().numpy(), zeros.numpy())
        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, input_tensor.shape).all()

    def reshard_cross_mesh(self):
        if self._backend != "gpu":
            return

        a = paddle.ones([10, 10])
        input_tensor = dist.shard_tensor(
            a, mesh=self._mesh, placements=[dist.Shard(0)]
        )
        dist.reshard(
            input_tensor,
            mesh=self._other_mesh,
            placements=[dist.Partial(dist.ReduceType.kRedSum)],
        )

    def run_test_case(self):
        self.reshard_same_mesh()
        self.reshard_cross_mesh()


if __name__ == '__main__':
    TestReshardSToP().run_test_case()
