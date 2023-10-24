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


import numpy as np

import paddle
import paddle.distributed as dist

from .semi_auto_parallel_util import SemiAutoParallelTestBase


class TestMatmulApiForSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()

    def test_matmul_x_row_shard(self):
        # case1: mk[0,-1],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[]
        inputs_shape = ([64, 32], [32, 48])
        inputs_specs = (['x', None], [None, None])
        dist_input, dist_out = self.test_body(
            inputs_shape, inputs_specs, paddle.matmul
        )
        x, y = dist_input
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert dist_out.dist_attr._is_partial() is False
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(x.grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(
            x.grad.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert x.grad.dist_attr._is_partial() is False
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(y.grad._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            y.grad.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert y.grad.dist_attr._is_partial() is False

    def test_matmul_x_column_shard(self):
        # case2: mk[-1, 0],kn[-1,-1] --> mk[-1, 0],kn[0, -1] = nm[-1, -1] partial[0]

        inputs_shape = ([64, 32], [32, 48])
        inputs_specs = ([None, 'x'], [None, None])
        dist_input, dist_out = self.test_body(
            inputs_shape, inputs_specs, paddle.matmul
        )
        x, y = dist_input
        # verify local shape
        np.testing.assert_equal(dist_out._local_shape, [64, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert dist_out.dist_attr._is_partial() is False
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(x.grad._local_shape, [64, 16], verbose=True)
        np.testing.assert_equal(
            x.grad.dist_attr.dims_mapping, [-1, 0], verbose=True
        )
        assert x.grad.dist_attr._is_partial() is False
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(y.grad._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            y.grad.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert y.grad.dist_attr._is_partial() is False

    def test_matmul_x_column_shard_trans_x_y(self):
        # case1: mk[-1,0],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[], trans x, trans y
        inputs_shape = ([32, 64], [48, 32])
        inputs_specs = ([None, 'x'], [None, None])
        dist_input, dist_out = self.test_body(
            inputs_shape,
            inputs_specs,
            trans_x=True,
            trans_y=True,
        )
        x, y = dist_input
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert dist_out.dist_attr._is_partial() is False
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(x.grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(
            x.grad.dist_attr.dims_mapping, [-1, 0], verbose=True
        )
        assert x.grad.dist_attr._is_partial() is False
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(y.grad._local_shape, [48, 32], verbose=True)
        np.testing.assert_equal(
            y.grad.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert y.grad.dist_attr._is_partial() is False

    def test_matmul_x_column_shard_trans_x(self):
        # case1: mk[-1,0],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[], trans x
        inputs_shape = ([32, 64], [32, 48])
        inputs_specs = ([None, 'x'], [None, None])
        dist_input, dist_out = self.test_body(
            inputs_shape,
            inputs_specs,
            trans_x=True,
            trans_y=False,
        )
        x, y = dist_input
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert dist_out.dist_attr._is_partial() is False
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(x.grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(
            x.grad.dist_attr.dims_mapping, [-1, 0], verbose=True
        )
        assert x.grad.dist_attr._is_partial() is False
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(y.grad._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            y.grad.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert y.grad.dist_attr._is_partial() is False

    def test_matmul_x_row_shard_trans_y(self):
        # case1: mk[0,-1],kn[-1,-1] -> mk[0,-1],kn[-1,-1] = mn[0,-1] partial[], trans y
        inputs_shape = ([64, 32], [48, 32])
        inputs_specs = (['x', None], [None, None])
        dist_input, dist_out = self.test_body(
            inputs_shape,
            inputs_specs,
            trans_x=False,
            trans_y=True,
        )
        x, y = dist_input
        # verify output local shape and dist attr
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert dist_out.dist_attr._is_partial() is False
        # verify x_grad local shape and dist attr
        np.testing.assert_equal(x.grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(
            x.grad.dist_attr.dims_mapping, [0, -1], verbose=True
        )
        assert x.grad.dist_attr._is_partial() is False
        # verify y_grad local shape and dist attr
        np.testing.assert_equal(y.grad._local_shape, [48, 32], verbose=True)
        np.testing.assert_equal(
            y.grad.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        assert y.grad.dist_attr._is_partial() is False

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


if __name__ == '__main__':
    TestMatmulApiForSemiAutoParallel().run_test_case()
