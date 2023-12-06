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
from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist


class TestExpandApiForSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_dim_mapping(self, output, expected_dim_mapping):
        assert (
            output.dist_attr.dims_mapping == expected_dim_mapping
        ), f"{output.dist_attr.dims_mapping}  vs {expected_dim_mapping}"

    # NOTE: raise error
    def test_expand_shard_0(self):
        x_shape = ([10, 1, 8],)
        x_specs = (['x', None, None],)
        _, output = self.runfunc_and_check(
            x_shape,
            x_specs,
            op_func=paddle.expand,
            with_backward=True,
            shape=[10, 8, 8],
        )
        self.check_dim_mapping(output, [0, -1, -1])

    def test_body(self, x_shape, out_shape, x_placements, op_func):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        x = paddle.randn(x_shape, self._dtype)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(x, self._mesh, x_placements)
        dist_x.stop_gradient = False

        dist_out = op_func(dist_x, shape=out_shape)
        out = op_func(x, shape=out_shape)
        self.check_tensor_eq(out, dist_out)
        np.testing.assert_equal(dist_out.shape, out_shape, verbose=True)

        dist_out.backward()
        out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)

    # NOTE: raise error
    def test_expand_shard_on_0(self):
        self.test_body(
            x_shape=[10, 1, 8],
            out_shape=[10, 8, 8],
            x_placements=[dist.Shard(0)],
            op_func=paddle.expand,
        )

    # NOTE: raise error
    def test_expand_shard_on_2(self):
        self.test_body(
            x_shape=[10, 1, 8],
            out_shape=[10, 8, 8],
            x_placements=[dist.Shard(2)],
            op_func=paddle.expand,
        )

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_expand_shard_0()
        self.test_expand_shard_on_0()
        self.test_expand_shard_on_2()


if __name__ == '__main__':
    TestExpandApiForSemiAutoParallel().run_test_case()
