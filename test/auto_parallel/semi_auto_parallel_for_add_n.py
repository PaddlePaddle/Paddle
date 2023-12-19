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


class TestAddNApiForSemiAutoParallel:
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
        self,
        x_shape,
        y_shape,
        x_placements,
        y_placements,
        trans_x=False,
        trans_y=False,
    ):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        x_np = np.random.random(size=x_shape).astype(self._dtype)
        y_np = np.random.random(size=y_shape).astype(self._dtype)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False

        dist_x = dist.shard_tensor(x_np, self._mesh, x_placements)
        dist_y = dist.shard_tensor(y_np, self._mesh, y_placements)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False

        out = paddle.add_n([x, y])
        dist_out = paddle.add_n([dist_x, dist_y])
        self.check_tensor_eq(out, dist_out)

        out.backward()
        dist_out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)
        self.check_tensor_eq(y.grad, dist_y.grad)

        return dist_out, dist_x.grad, dist_y.grad

    def test_add_n(self):
        self.test_body(
            x_shape=[64, 32],
            y_shape=[64, 32],
            x_placements=[dist.Replicate()],
            y_placements=[dist.Replicate()],
        )

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_add_n()


if __name__ == '__main__':
    TestAddNApiForSemiAutoParallel().run_test_case()
