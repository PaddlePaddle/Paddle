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
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_body(self, x_specs, y_specs):
        x_shape = [64, 32]
        y_shape = [32, 48]
        x = paddle.randn(x_shape, self._dtype)
        y = paddle.randn(y_shape, self._dtype)
        x.stop_gradient = False
        y.stop_gradient = False

        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)
        y_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=y_specs)

        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_y = dist.shard_tensor(y, dist_attr=y_dist_attr)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False

        dist_out = paddle.matmul(dist_x, dist_y)
        # verify global shape
        out_shape = [64, 48]
        np.testing.assert_equal(dist_out.shape, out_shape, verbose=True)

        dist_out.backward()
        np.testing.assert_equal(dist_x.grad.shape, x_shape, verbose=True)
        np.testing.assert_equal(dist_y.grad.shape, y_shape, verbose=True)

        return dist_out, dist_x.grad, dist_y.grad

    def test_case1(self):
        # case1: mk[0,-1],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[]
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_specs=['x', None], y_specs=[None, None]
        )
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert dist_out.dist_attr._is_partial() is False
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [32, 32], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert dist_x_grad.dist_attr._is_partial() is False
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [32, 48], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert dist_y_grad.dist_attr._is_partial() is False

    def test_case2(self):
        # case2: mk[-1, 0],kn[-1,-1] --> mk[-1, 0],kn[0, -1] = nm[-1, -1] partial[0]
        dist_out, dist_x_grad, dist_y_grad = self.test_body(
            x_specs=[None, 'x'], y_specs=[None, None]
        )
        # verify local shape
        np.testing.assert_equal(dist_out._local_shape, [64, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert dist_out.dist_attr._is_partial() is False
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(
            dist_x_grad._local_shape, [64, 16], verbose=True
        )
        np.testing.assert_equal(
            dist_x_grad.dist_attr.dims_mapping, [-1, 0], verbose=True
        )
        assert dist_x_grad.dist_attr._is_partial() is False
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(
            dist_y_grad._local_shape, [32, 48], verbose=True
        )
        np.testing.assert_equal(
            dist_y_grad.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert dist_y_grad.dist_attr._is_partial() is False

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        # backward need P2R
        self.test_case1()
        # forward need P2R
        self.test_case2()


if __name__ == '__main__':
    TestMatmulApiForSemiAutoParallel().run_test_case()
