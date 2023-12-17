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
from paddle.distributed import Replicate, Shard


class TestEmbeddingApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_body(self, x_shape, w_shape, x_placements, w_placements):
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        x_np = np.random.randint(0, 10, size=x_shape)
        w_np = np.random.random(size=w_shape).astype(self._dtype)

        x = paddle.to_tensor(x_np)
        w = paddle.to_tensor(w_np)
        x.stop_gradient = False
        w.stop_gradient = False

        dist_x = dist.shard_tensor(x_np, self._mesh, x_placements)
        dist_w = dist.shard_tensor(w_np, self._mesh, w_placements)
        dist_x.stop_gradient = False
        dist_w.stop_gradient = False

        out = paddle.nn.functional.embedding(x, weight=w)
        dist_out = paddle.nn.functional.embedding(dist_x, weight=dist_w)
        self.check_tensor_eq(out, dist_out)

        out.backward()
        dist_out.backward()
        self.check_tensor_eq(w.grad, dist_w.grad)

        return dist_out, dist_w.grad

    def test_non_shard(self):
        self.test_body(
            x_shape=[12, 16],
            w_shape=[10, 4],
            x_placements=[Replicate()],
            w_placements=[Replicate()],
        )

    def test_x_row_shard(self):
        self.test_body(
            x_shape=[12, 16],
            w_shape=[10, 4],
            x_placements=[Shard(0)],
            w_placements=[Replicate()],
        )

    def test_x_col_shard(self):
        self.test_body(
            x_shape=[12, 16],
            w_shape=[10, 4],
            x_placements=[Shard(1)],
            w_placements=[Replicate()],
        )

    def test_w_row_shard(self):
        self.test_body(
            x_shape=[12, 16],
            w_shape=[10, 4],
            x_placements=[Replicate()],
            w_placements=[Shard(0)],
        )

    def test_w_col_shard(self):
        self.test_body(
            x_shape=[12, 16],
            w_shape=[10, 4],
            x_placements=[Replicate()],
            w_placements=[Shard(1)],
        )

    def test_x_row_w_col_shard(self):
        try:
            self.test_body(
                x_shape=[12, 16],
                w_shape=[10, 4],
                x_placements=[Shard(0)],
                w_placements=[Shard(1)],
            )
        except RuntimeError as e:
            assert 'sharded by same mesh dimension ' in str(e)

    def test_x_col_w_row_shard(self):
        # Unimplemented cpu kernel for CReduceScatterOp
        if self._backend == "cpu":
            return

        self.test_body(
            x_shape=[12, 16],
            w_shape=[10, 4],
            x_placements=[Shard(1)],
            w_placements=[Shard(0)],
        )

    def test_both_col_shard(self):
        try:
            self.test_body(
                x_shape=[12, 16],
                w_shape=[10, 4],
                x_placements=[Shard(1)],
                w_placements=[Shard(1)],
            )
        except RuntimeError as e:
            assert 'sharded by same mesh dimension', str(e)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_non_shard()
        self.test_x_row_shard()
        self.test_x_col_shard()
        self.test_w_row_shard()
        self.test_w_col_shard()
        self.test_x_row_w_col_shard()
        self.test_x_col_w_row_shard()
        self.test_both_col_shard()


if __name__ == '__main__':
    TestEmbeddingApiForSemiAutoParallel().run_test_case()
