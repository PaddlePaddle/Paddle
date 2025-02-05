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

import numpy as np
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

    def check_placements(self, output, expected_placements):
        assert (
            output.placements == expected_placements
        ), f"{output.placements}  vs {expected_placements}"

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

        self.check_placements(output, [dist.Shard(0)])
        self.check_placements(input.grad, [dist.Shard(0)])

    def test_reshape_infer_shape(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        x = paddle.ones([10, 20, 30])
        x = dist.shard_tensor(x, mesh, [Shard(0)])
        y = x.reshape([-1, 0, x.shape[0]])
        assert y.shape == [30, 20, 10]
        assert y._local_shape == [15, 20, 10]

    def test_shape_api_with_reshape(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        a = paddle.rand(shape=[4, 6, 8])
        b = dist.shard_tensor(a, mesh, [dist.Shard(0)])

        dist_shape = paddle.shape(b)
        b = b.reshape((-1, dist_shape[-1]))
        assert b.shape == [24, 8]
        assert b._local_shape == [12, 8]

    def test_reshape_grad_with_reshard_x_grad(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        x = paddle.rand(shape=[2, 8, 4, 48])
        x.stop_gradient = False
        dist_x = dist.shard_tensor(x, mesh, [Shard(2)])
        dist_out = dist_x.reshape([64, 48])  # dist_x needs reshard

        # calling reshape_grad, its output dist_x.grad is replicated
        # on the mesh, different from dist_x's placements,
        # so it will reshard to dist_x's placements [Shard(2)]
        dist_out.backward()

        np.testing.assert_equal(dist_x._local_shape, [2, 8, 2, 48])
        np.testing.assert_equal(dist_out._local_shape, [64, 48])
        np.testing.assert_equal(dist_x.grad._local_shape, [2, 8, 2, 48])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")
        self.test_reshape_forward()
        self.test_reshape_infer_shape()
        self.test_shape_api_with_reshape()
        self.test_reshape_grad_with_reshard_x_grad()


if __name__ == '__main__':
    TestReshapeSemiAutoParallel().run_test_case()
