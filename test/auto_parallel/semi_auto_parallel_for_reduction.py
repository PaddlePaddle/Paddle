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

from .semi_auto_parallel_util import SemiAutoParallelTestBase


class TestReductionApiForSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_body(self, x_shape, out_shape, x_specs, axis, keepdim, op_func):
        dist_input, dist_out = super().test_body(
            x_shape,
            x_specs,
            op_func,
            axis=axis,
            keepdim=keepdim,
        )
        np.testing.assert_equal(dist_out.shape, out_shape, verbose=True)

    def test_reduce_x_shard(self):
        for op_func in [paddle.sum, paddle.mean]:
            self.test_body(
                x_shape=[4, 8, 6],
                out_shape=[4, 6],
                x_specs=['x', None, None],
                axis=1,
                keepdim=False,
                op_func=op_func,
            )
            self.test_body(
                x_shape=[4, 8, 6],
                out_shape=[8, 6],
                x_specs=['x', None, None],
                axis=-3,
                keepdim=False,
                op_func=op_func,
            )

    def test_sum_x_shard_on_axis(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4],
            x_specs=[None, 'x', None],
            axis=[1, 2],
            keepdim=False,
            op_func=paddle.sum,
        )

    def test_sum_x_shard_on_axis_keepdim(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[4, 1, 6],
            x_specs=[None, 'x', None],
            axis=1,
            keepdim=True,
            op_func=paddle.sum,
        )

    def test_mean_x_shard(self):
        self.test_body(
            x_shape=[4, 8, 6],
            out_shape=[8, 6],
            x_specs=['x', None, None],
            axis=-3,
            keepdim=False,
            op_func=paddle.mean,
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


if __name__ == '__main__':
    TestReductionApiForSemiAutoParallel().run_test_case()
