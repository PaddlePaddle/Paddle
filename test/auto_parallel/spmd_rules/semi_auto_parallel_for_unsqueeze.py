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
import parameterized as param

import paddle
import paddle.distributed as dist


class TestUnsqueezeApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    @param.parameterized(
        (
            ((2, 3), (2, 1, 3), ('x', None), 1),
            paddle.unsqueeze(
                (2, 3), (0, 2, 3, 1), (None, 'x'), (0, -1), paddle.unsqueeze
            ),
        )
    )
    def test_unsqueeze(self, x_shape, out_shape, x_specs, axis, op_func):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        x = paddle.randn(x_shape, self._dtype)
        x.stop_gradient = False

        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)

        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_x.stop_gradient = False

        dist_out = op_func(dist_x, axis=axis)
        out = op_func(x, axis=axis)
        self.check_tensor_eq(out, dist_out)
        np.testing.assert_equal(dist_out.shape, out_shape, verbose=True)

        # TODO(cxxly) add backward prop
        # dist_out.backward()
        # out.backward()
        # self.check_tensor_eq(x.grad, dist_x.grad)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_sum_x_shard()
        self.test_sum_x_shard_on_axis()
        self.test_sum_x_shard_on_axis_keepdim()
        self.test_mean_x_shard()


if __name__ == '__main__':
    TestUnsqueezeApiForSemiAutoParallel().run_test_case()
