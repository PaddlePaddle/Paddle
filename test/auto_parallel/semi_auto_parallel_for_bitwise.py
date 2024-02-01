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


class TestBitwiseApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._check_grad = False
        self._rtol = 1e-6
        self._atol = 0.0
        paddle.seed(self._seed)
        np.random.seed(self._seed)

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(
            np1, np2, rtol=self._rtol, atol=self._atol, verbose=True
        )

    def test_unary_body(self, x_shape, out_shape, x_placements, unary_func):
        x = paddle.randint(0, 100, x_shape, self._dtype)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(x, self._mesh, x_placements)
        dist_x.stop_gradient = False

        dist_out = unary_func(dist_x)
        out = unary_func(x)
        self.check_tensor_eq(out, dist_out)
        if self._check_grad:
            dist_out.backward()
            out.backward()
            self.check_tensor_eq(x.grad, dist_x.grad)

    def test_binary_body(
        self,
        x_shape,
        y_shape,
        out_shape,
        x_placements,
        y_placements,
        binary_func,
    ):
        x = paddle.randint(0, 100, x_shape, self._dtype)
        y = paddle.randint(0, 100, y_shape, self._dtype)
        x.stop_gradient = False
        y.stop_gradient = False

        dist_x = dist.shard_tensor(x, self._mesh, x_placements)
        dist_y = dist.shard_tensor(y, self._mesh, y_placements)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False

        dist_out = binary_func(dist_x, dist_y)
        out = binary_func(x, y)
        self.check_tensor_eq(out, dist_out)

        if self._check_grad:
            dist_out.backward()
            out.backward()
            self.check_tensor_eq(x.grad, dist_x.grad)
            self.check_tensor_eq(y.grad, dist_y.grad)

    def test_bitwise_and_x_shard(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Replicate()],
            binary_func=paddle.bitwise_and,
        )

    def test_bitwise_and_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Replicate()],
            binary_func=paddle.bitwise_and,
        )

    def test_bitwise_and_x_y_shard(self):
        if self._backend == "cpu":
            return
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[16, 32],
            out_shape=[16, 32],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Shard(1)],
            binary_func=paddle.bitwise_and,
        )

    def test_bitwise_and_x_y_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[4, 16, 32],
            y_shape=[16, 32],
            out_shape=[4, 16, 32],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Replicate()],
            binary_func=paddle.bitwise_and,
        )

    def test_bitwise_not_x_shard(self):
        self.test_unary_body(
            x_shape=[16, 32],
            out_shape=[16, 32],
            x_placements=[dist.Shard(0)],
            unary_func=paddle.bitwise_not,
        )

    def test_bitwise_not_x_shard_broadcast(self):
        self.test_binary_body(
            x_shape=[16, 32],
            y_shape=[2, 16, 32],
            out_shape=[2, 16, 32],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Replicate()],
            binary_func=paddle.bitwise_not,
        )

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_bitwise_and_x_shard()
        self.test_bitwise_and_x_shard_broadcast()
        self.test_bitwise_and_x_y_shard()
        self.test_bitwise_and_x_y_shard_broadcast()
        self.test_bitwise_not_x_shard()


if __name__ == '__main__':
    TestBitwiseApiForSemiAutoParallel().run_test_case()
