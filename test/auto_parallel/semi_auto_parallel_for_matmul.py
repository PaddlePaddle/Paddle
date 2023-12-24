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
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-04, verbose=True)

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

        out = paddle.matmul(x, y, transpose_x=trans_x, transpose_y=trans_y)
        dist_out = paddle.matmul(
            dist_x, dist_y, transpose_x=trans_x, transpose_y=trans_y
        )

        self.check_tensor_eq(out, dist_out)

        out.backward()
        dist_out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)
        self.check_tensor_eq(y.grad, dist_y.grad)

        return dist_out, dist_x.grad, dist_y.grad

    def test_matmul_x_row_shard(self):
        # case1: mk[0,-1],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[]
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_shape=[64, 32],
            y_shape=[32, 48],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Replicate()],
        )
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.placements, [dist.Shard(0)], verbose=True
        )
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [32, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.placements, [dist.Shard(0)], verbose=True
        )
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [32, 48], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.placements, [dist.Partial()], verbose=True
        )

    def test_batch_matmul(self):
        # case1: amk[0,-1, -1],kn[-1,-1] -> amk[0,-1,-1],kn[-1,-1] = amn[0,-1, -1] partial[]
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_shape=[2, 64, 32],
            y_shape=[32, 48],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Replicate()],
        )
        # verify output local shape and dist attr
        np.testing.assert_equal(
            dist_out._local_shape, [1, 64, 48], verbose=True
        )
        np.testing.assert_equal(
            dist_out.placements, [dist.Shard(0)], verbose=True
        )
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [1, 64, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.placements, [dist.Shard(0)], verbose=True
        )
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [32, 48], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.placements, [dist.Partial()], verbose=True
        )

    def test_matmul_x_column_shard(self):
        # case2: mk[-1, 0],kn[-1,-1] --> mk[-1, 0],kn[0, -1] = nm[-1, -1] partial[0]
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_shape=[64, 32],
            y_shape=[32, 48],
            x_placements=[dist.Shard(1)],
            y_placements=[dist.Replicate()],
        )
        # verify local shape
        np.testing.assert_equal(dist_out._local_shape, [64, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.placements, [dist.Partial()], verbose=True
        )
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [64, 16], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.placements, [dist.Shard(1)], verbose=True
        )
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [32, 48], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.placements, [dist.Replicate()], verbose=True
        )

    def test_matmul_x_column_shard_trans_x_y(self):
        # case1: mk[-1,0],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[], trans x, trans y
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_shape=[32, 64],
            y_shape=[48, 32],
            x_placements=[dist.Shard(1)],
            y_placements=[dist.Replicate()],
            trans_x=True,
            trans_y=True,
        )
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.placements, [dist.Shard(0)], verbose=True
        )
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [32, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.placements, [dist.Shard(1)], verbose=True
        )
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [48, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.placements, [dist.Partial()], verbose=True
        )

    def test_matmul_x_column_shard_trans_x(self):
        # case1: mk[-1,0],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[], trans x
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_shape=[32, 64],
            y_shape=[32, 48],
            x_placements=[dist.Shard(1)],
            y_placements=[dist.Replicate()],
            trans_x=True,
            trans_y=False,
        )
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.placements, [dist.Shard(0)], verbose=True
        )
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [32, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.placements, [dist.Shard(1)], verbose=True
        )
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [32, 48], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.placements, [dist.Partial()], verbose=True
        )

    def test_matmul_x_row_shard_trans_y(self):
        # case1: mk[0,-1],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[], trans y
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_shape=[64, 32],
            y_shape=[48, 32],
            x_placements=[dist.Shard(0)],
            y_placements=[dist.Replicate()],
            trans_x=False,
            trans_y=True,
        )
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.placements, [dist.Shard(0)], verbose=True
        )
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [32, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.placements, [dist.Shard(0)], verbose=True
        )
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [48, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.placements, [dist.Partial()], verbose=True
        )

    def test_matmul_with_complex_type(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        x_np = np.random.random(size=[64, 32]).astype(np.complex128)
        y_np = np.random.random(size=[32, 48]).astype(np.float32)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False

        dist_x = dist.shard_tensor(x_np, self._mesh, [dist.Replicate()])
        dist_y = dist.shard_tensor(y_np, self._mesh, [dist.Replicate()])
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False

        out = paddle.matmul(x, y, transpose_x=False, transpose_y=False)
        dist_out = paddle.matmul(
            dist_x, dist_y, transpose_x=False, transpose_y=False
        )

        self.check_tensor_eq(out, dist_out)

        out.backward()
        dist_out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)
        self.check_tensor_eq(y.grad, dist_y.grad)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_matmul_x_row_shard()
        self.test_matmul_x_column_shard()
        self.test_matmul_x_column_shard_trans_x_y()
        self.test_matmul_x_column_shard_trans_x()
        self.test_matmul_x_row_shard_trans_y()
        self.test_matmul_with_complex_type()
        self.test_batch_matmul()


if __name__ == '__main__':
    TestMatmulApiForSemiAutoParallel().run_test_case()
