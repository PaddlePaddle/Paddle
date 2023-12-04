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


class TestSqueezeApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-06, verbose=True)

    def test_body(self, x_shape, out_shape, x_placements, axis, op_func):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        x = paddle.randn(x_shape, self._dtype)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(x, self._mesh, x_placements)
        dist_x.stop_gradient = False

        dist_out = op_func(dist_x, axis=axis)
        out = op_func(x, axis=axis)
        self.check_tensor_eq(out, dist_out)
        np.testing.assert_equal(dist_out.shape, out_shape, verbose=True)

        dist_out.backward()
        out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)

    def test_squeeze(self):
        self.test_body(
            x_shape=[1, 4, 1, 6],
            out_shape=[4, 1, 6],
            x_placements=[dist.Shard(1)],
            axis=0,
            op_func=paddle.squeeze,
        )

    def test_squeeze_multi_axes(self):
        self.test_body(
            x_shape=[1, 4, 1, 6],
            out_shape=[4, 6],
            x_placements=[dist.Shard(1)],
            axis=(0, 2),
            op_func=paddle.squeeze,
        )

    def test_unsqueeze(self):
        self.test_body(
            x_shape=[4, 6],
            out_shape=[1, 4, 6],
            x_placements=[dist.Shard(0)],
            axis=0,
            op_func=paddle.unsqueeze,
        )

    def test_unsqueeze_multi_axes(self):
        self.test_body(
            x_shape=[4, 6],
            out_shape=[1, 4, 6, 1],
            x_placements=[dist.Shard(1)],
            axis=(0, 3),
            op_func=paddle.unsqueeze,
        )

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_squeeze()
        self.test_squeeze_multi_axes()
        self.test_unsqueeze()
        self.test_unsqueeze_multi_axes()


if __name__ == '__main__':
    TestSqueezeApiForSemiAutoParallel().run_test_case()
