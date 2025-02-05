# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
import paddle.distributed as dist


class TestMoEUtils:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seed = eval(os.getenv("seed"))
        self._backend = os.getenv("backend")
        self._mesh0 = dist.ProcessMesh([[0], [1]], dim_names=["x", "y"])
        self._mesh1 = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"])

    def test_local_reshape(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        tgt_shape = [h // 2, w * 2]
        x = paddle.arange(0, h * w).reshape(src_shape)
        x.stop_gradient = False
        np_x = x.numpy()

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        dist_y = dist.auto_parallel.moe_utils._dist_reshape(
            dist_x, [-1, w * 2], self._mesh0, [dist.Shard(1), dist.Replicate()]
        )

        splitted_np_x = np.split(np_x, 2, axis=1)
        for i in range(len(splitted_np_x)):
            splitted_np_x[i] = splitted_np_x[i].reshape([h // 2, w])
        np.testing.assert_array_equal(
            splitted_np_x[dist.get_rank()], dist_y._local_value().numpy()
        )

        label = paddle.ones(tgt_shape, dtype=paddle.int64)
        label.stop_gradient = False
        dist_label = dist.shard_tensor(
            label, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        loss = dist_y - dist_label
        loss.backward()

        np_grad = np.ones(src_shape, dtype="int64")
        splitted_np_grad = np.split(np_grad, 2, axis=1)
        np.testing.assert_array_equal(
            splitted_np_grad[dist.get_rank()],
            dist_x.grad._local_value().numpy(),
        )

        with unittest.TestCase().assertRaises(AssertionError):
            dist_z = dist.auto_parallel.moe_utils._dist_reshape(
                dist_x,
                dist_x.shape,
                self._mesh1,
                [dist.Replicate(), dist.Replicate()],
            )

        # test the warning log message
        dist_z = dist.auto_parallel.moe_utils._dist_reshape(
            dist_x, dist_x.shape, self._mesh0, [dist.Shard(1), dist.Shard(1)]
        )

    def test_nd_mesh_alltoall(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        dist_y = dist.reshard(
            dist_x, self._mesh0, [dist.Shard(0), dist.Replicate()]
        )
        dist_y.backward()

        assert dist_y.placements == [dist.Shard(0), dist.Replicate()]
        assert dist_x.grad.placements == [dist.Shard(1), dist.Replicate()]
        np_grad = np.ones(src_shape, dtype="int64")
        splitted_np_grad = np.split(np_grad, 2, axis=1)
        np.testing.assert_array_equal(
            splitted_np_grad[dist.get_rank()],
            dist_x.grad._local_value().numpy(),
        )

    def test_reshard_mesh_shape(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        dist_y = dist.reshard(
            dist_x, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )

        assert dist_y.process_mesh == self._mesh1
        np.testing.assert_array_equal(
            dist_y._local_value().numpy(), dist_x._local_value().numpy()
        )

    def run_test_case(self):
        self.test_local_reshape()
        self.test_nd_mesh_alltoall()
        self.test_reshard_mesh_shape()


if __name__ == '__main__':
    TestMoEUtils().run_test_case()
