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
import paddle.nn.functional as F


class TestReplicatedSPmdApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_relu(self):
        dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None])
        x = dist.dtensor_from_fn(paddle.randn, dist_attr, [64, 32], self._dtype)
        out = F.relu(x)
        # verify output local shape and dist attr
        np.testing.assert_equal(out._local_shape, [64, 32], verbose=True)
        np.testing.assert_equal(
            out.dist_attr.dims_mapping, [-1, -1], verbose=True
        )

    def test_cross_entropy(self):
        N = 100
        C = 200
        dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None])
        input = dist.dtensor_from_fn(
            paddle.randn, dist_attr, [N, C], self._dtype
        )
        label = dist.dtensor_from_fn(
            paddle.randint, dist_attr, 0, C, [N], self._dtype
        )
        weight = dist.dtensor_from_fn(paddle.randn, dist_attr, [C], self._dtype)

        cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
            weight=weight, reduction='mean'
        )
        out = cross_entropy_loss(input, label)
        print(out)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_relu()
        self.test_cross_entropy()


if __name__ == '__main__':
    TestReplicatedSPmdApiForSemiAutoParallel().run_test_case()
