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


class TestSemiAutoParallel3DGlobalMeshReshard:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._global_mesh = dist.ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=['pp', 'dp', 'mp']
        )
        self._mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'mp'])
        self._mesh1 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=['dp', 'mp'])
        paddle.set_device(self._backend)

    def test_basic(self):
        global_input = dist.shard_tensor(
            paddle.ones(shape=[6, 8], dtype='float32'),
            self._global_mesh,
            [dist.Replicate(), dist.Replicate(), dist.Replicate()],
        )  # 1.0
        global_input.stop_gradient = False
        # forward on mesh0
        input_mesh0 = dist.reshard(
            global_input, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        output = input_mesh0 + 1.0  # 2.0

        # forward on mesh1
        output = dist.reshard(
            output, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )
        input_mesh1 = dist.reshard(
            global_input, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )
        output = output + input_mesh1  # 3.0
        loss = paddle.sum(output)  # 144.0
        np.testing.assert_allclose(
            loss.numpy(), 144.0, rtol=1e-06, verbose=True
        )
        loss.backward()
        np.testing.assert_allclose(
            global_input.grad.numpy(),
            np.full(shape=(6, 8), fill_value=2.0, dtype=np.float32),
            rtol=1e-06,
            verbose=True,
        )

    def test_3d_mesh_with_any_status(self):
        dense_tensor = paddle.ones(shape=[2, 6], dtype='float32')
        dist_tensor = dist.shard_tensor(
            dense_tensor,
            self._global_mesh,
            [dist.Replicate(), dist.Shard(0), dist.Replicate()],
        )
        np.testing.assert_equal(dist_tensor._local_shape, [1, 6])

    def run_test_case(self):
        self.test_basic()
        self.test_3d_mesh_with_any_status()


if __name__ == '__main__':
    TestSemiAutoParallel3DGlobalMeshReshard().run_test_case()
