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


class TestSoftmaxApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        self._rtol = 1e-6
        self._atol = 0

        # The gradient of softmax is tiny, ref test_softmax_op.py, use atol
        # to check the backward precision.
        self._bwd_rtol = 0
        self._bwd_atol = 1e-6

        paddle.seed(self._seed)
        np.random.seed(self._seed)

    def check_tensor_eq(self, a, b, rtol=1e-6, atol=0):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=rtol, atol=atol, verbose=True)

    def test_body(self, x_shape, out_shape, x_placements, func):
        x = paddle.rand(x_shape, dtype=self._dtype)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(x, self._mesh, x_placements)
        dist_x.stop_gradient = False

        dist_out = func(dist_x)
        out = func(x)
        self.check_tensor_eq(out, dist_out, self._rtol, self._atol)

        dist_out.sum().backward()
        out.sum().backward()
        self.check_tensor_eq(
            x.grad, dist_x.grad, self._bwd_rtol, self._bwd_atol
        )

    def test_softmax_shard(self):
        self.test_body(
            x_shape=[20, 30],
            out_shape=[4, 4],
            x_placements=[dist.Shard(0)],
            func=lambda x: paddle.nn.functional.softmax(x, axis=1),
        )

    def test_softmax_shard_along_axis(self):
        self.test_body(
            x_shape=[20, 30],
            out_shape=[20, 30],
            x_placements=[dist.Shard(1)],
            func=lambda x: paddle.nn.functional.softmax(x, axis=1),
        )

    def test_multi_axes(self):
        self.test_body(
            x_shape=[2, 4, 6, 10],
            out_shape=[2, 4, 6, 10],
            x_placements=[dist.Shard(0)],
            func=lambda x: paddle.nn.functional.softmax(x, axis=1),
        )

    def test_multi_axes_along_axis(self):
        self.test_body(
            x_shape=[2, 4, 6, 10],
            out_shape=[2, 4, 6, 10],
            x_placements=[dist.Shard(0)],
            func=lambda x: paddle.nn.functional.softmax(x, axis=0),
        )

    def test_negative_axis(self):
        self.test_body(
            x_shape=[2, 4, 6, 10],
            out_shape=[2, 4, 6, 10],
            x_placements=[dist.Shard(0)],
            func=lambda x: paddle.nn.functional.softmax(x, axis=-4),
        )

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_softmax_shard()
        self.test_softmax_shard_along_axis()
        self.test_multi_axes()
        self.test_multi_axes_along_axis()
        self.test_negative_axis()


if __name__ == '__main__':
    TestSoftmaxApiForSemiAutoParallel().run_test_case()
