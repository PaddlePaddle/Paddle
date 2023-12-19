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
from paddle.framework import core


class TestReshardSToS:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._other_mesh = dist.ProcessMesh([1, 0], dim_names=["x"])

    def test_body(self, in_shard, out_shard):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        dev_ctx = core.DeviceContext.create(place)
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(a, self._mesh, [dist.Shard(in_shard)])
        out = dist.reshard(input_tensor, self._mesh, [dist.Shard(out_shard)])

        out_shape = list(self._shape)
        out_shape[out_shard] = out_shape[out_shard] // 2

        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, out_shape).all()

    def test_case1(self):
        self.test_body(0, len(self._shape) - 1)

    def test_case2(self):
        self.test_body(len(self._shape) - 1, 0)

    def reshard_cross_mesh(self):
        if self._backend != "gpu":
            return
        a = paddle.ones([10, 10])
        input_tensor = dist.shard_tensor(a, self._mesh, [dist.Shard(0)])
        dist.reshard(input_tensor, self._other_mesh, [dist.Shard(1)])

    def run_test_case(self):
        self.test_case1()
        self.test_case2()
        self.reshard_cross_mesh()


if __name__ == '__main__':
    TestReshardSToS().run_test_case()
