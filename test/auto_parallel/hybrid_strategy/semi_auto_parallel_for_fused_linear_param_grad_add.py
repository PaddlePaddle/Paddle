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
from paddle import _C_ops


class TestFusedParamGradAddForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_body(self):
        x_shape = [4, 16, 32]
        y_shape = [4, 16, 64]

        paddle.seed(self._seed)
        np.random.seed(self._seed)

        x_np = np.random.random(size=x_shape).astype(self._dtype)
        y_np = np.random.random(size=y_shape).astype(self._dtype)

        def run_acc_step(x, y):
            weight_grad = None
            bias_grad = None
            for _ in range(2):
                weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(
                    x,
                    y,
                    weight_grad,
                    bias_grad,
                    False,
                    True,
                )
            return weight_grad, bias_grad

        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = True
        y.stop_gradient = True

        weight_grad, bias_grad = run_acc_step(x, y)

        # test mp col split
        x_placements = [dist.Shard(0), dist.Replicate()]
        y_placements = [dist.Shard(0), dist.Shard(2)]

        dist_x = dist.shard_tensor(x_np, self._mesh, x_placements)
        dist_y = dist.shard_tensor(y_np, self._mesh, y_placements)
        dist_x.stop_gradient = True
        dist_y.stop_gradient = True

        weight_grad_dist, bias_grad_dist = run_acc_step(dist_x, dist_y)
        self.check_tensor_eq(weight_grad, weight_grad_dist)
        self.check_tensor_eq(bias_grad, bias_grad_dist)

        # test mp row split
        x_placements = [dist.Shard(0), dist.Shard(2)]
        y_placements = [dist.Shard(0), dist.Replicate()]
        dist_x = dist.shard_tensor(x_np, self._mesh, x_placements)
        dist_y = dist.shard_tensor(y_np, self._mesh, y_placements)
        dist_x.stop_gradient = True
        dist_y.stop_gradient = True
        weight_grad_dist, bias_grad_dist = run_acc_step(dist_x, dist_y)
        self.check_tensor_eq(weight_grad, weight_grad_dist)
        self.check_tensor_eq(bias_grad, bias_grad_dist)

    def test_fused_linear_param_grad_add(self):
        self.test_body()

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_fused_linear_param_grad_add()


if __name__ == '__main__':
    TestFusedParamGradAddForSemiAutoParallel().run_test_case()
