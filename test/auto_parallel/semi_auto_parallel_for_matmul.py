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


class TestMatmulApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + dist.get_rank())
        else:
            raise ValueError("Only support cpu or gpu backend.")

        x_shape = [64, 32]
        y_shape = [32, 48]
        x = paddle.randn(x_shape, self._dtype)
        y = paddle.randn(y_shape, self._dtype)

        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None])
        y_dist_attr = dist.DistAttr(
            mesh=self._mesh, sharding_specs=[None, None]
        )

        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_y = dist.shard_tensor(y, dist_attr=y_dist_attr)

        # case 1: mk[0,-1],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[]
        dist_out = paddle.matmul(dist_x, dist_y)

        # verify shape
        out_shape = [64, 48]
        out_local_shape = [32, 48]
        print("dist_out.shape: ", dist_out.shape, "out_shape: ", out_shape)
        print(
            "dist_out._local_shape: ",
            dist_out._local_shape,
            "out_local_shape: ",
            out_local_shape,
        )
        assert np.equal(dist_out.shape, out_shape).all()
        assert np.equal(dist_out._local_shape, out_local_shape).all()


if __name__ == '__main__':
    TestMatmulApiForSemiAutoParallel().run_test_case()
