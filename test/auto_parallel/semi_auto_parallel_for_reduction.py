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


class TestReductionApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_body(
        self, x_shape, out_shape, x_placements, axis, keepdim, op_func
    ):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        is_op_func_all = op_func == paddle.all

        x = paddle.randn(x_shape, self._dtype)
        if is_op_func_all:
            x = x > 0
        x.stop_gradient = False

        dist_x = dist.shard_tensor(x, self._mesh, x_placements)
        dist_x.stop_gradient = False

        dist_out = op_func(dist_x, axis=axis, keepdim=keepdim)
        out = op_func(x, axis=axis, keepdim=keepdim)
        self.check_tensor_eq(out, dist_out)
        np.testing.assert_equal(dist_out.shape, out_shape, verbose=True)

        if not is_op_func_all:
            dist_out.backward()
            out.backward()
            self.check_tensor_eq(x.grad, dist_x.grad)

    def test_sum_x_shard(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4, 6],
            x_placements=[dist.Shard(0)],
            axis=1,
            keepdim=False,
            op_func=paddle.sum,
        )

    def test_sum_x_shard_on_axis(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4],
            x_placements=[dist.Shard(1)],
            axis=[1, 2],
            keepdim=False,
            op_func=paddle.sum,
        )

    def test_sum_x_shard_on_axis_keepdim(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4, 1, 6],
            x_placements=[dist.Shard(1)],
            axis=1,
            keepdim=True,
            op_func=paddle.sum,
        )

    def test_mean_x_shard(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[8, 6],
            x_placements=[dist.Shard(0)],
            axis=-3,
            keepdim=False,
            op_func=paddle.mean,
        )

    def test_max_x_shard(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4, 6],
            x_placements=[dist.Shard(0)],
            axis=1,
            keepdim=False,
            op_func=paddle.max,
        )

    def test_max_x_shard_on_axis(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4, 6],
            x_placements=[dist.Shard(1)],
            axis=1,
            keepdim=False,
            op_func=paddle.max,
        )

    def test_all_x_shard(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4, 6],
            x_placements=[dist.Shard(0)],
            axis=1,
            keepdim=False,
            op_func=paddle.all,
        )

    def test_all_x_shard_on_axis(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4, 6],
            x_placements=[dist.Shard(1)],
            axis=1,
            keepdim=False,
            op_func=paddle.all,
        )

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
        self.test_max_x_shard()
        self.test_max_x_shard_on_axis()
        self.test_all_x_shard()
        self.test_all_x_shard_on_axis()


if __name__ == '__main__':
    TestReductionApiForSemiAutoParallel().run_test_case()
