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


class TestSemiAutoParallel2DGlobalMeshReshard:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._global_mesh = dist.ProcessMesh(
            [[0, 1], [2, 3]], dim_names=["pp", "dp"]
        )
        self._mesh0 = dist.ProcessMesh([0, 1], dim_names=["dp"])
        self._mesh1 = dist.ProcessMesh([2, 3], dim_names=["dp"])
        paddle.set_device(self._backend)

    def test_basic(self):
        input = paddle.ones(shape=[2, 3], dtype='float32')
        input = dist.shard_tensor(
            input, self._global_mesh, [dist.Replicate(), dist.Shard(0)]
        )
        input.stop_gradient = False
        global_input = input + 1.0  # global_input: 2.0

        # forward on pp0
        input_pp0 = dist.reshard(global_input, self._mesh0, [dist.Shard(0)])
        output = input_pp0 + 1.0  # output_pp0: 3.0

        # forward on pp1
        output = dist.reshard(output, self._mesh1, [dist.Shard(0)])
        input_pp1 = dist.reshard(global_input, self._mesh1, [dist.Shard(0)])
        output = input_pp1 + output  # output_pp1: 5.0
        loss = paddle.sum(output)  # 30.0
        np.testing.assert_allclose(loss.numpy(), 30.0, rtol=1e-06, verbose=True)
        loss.backward()
        np.testing.assert_allclose(
            input.grad.numpy(),
            np.full(shape=(2, 3), fill_value=2.0, dtype=np.float32),
            rtol=1e-06,
            verbose=True,
        )

    def test_split_dim1(self):
        global_mesh = dist.ProcessMesh([[0, 1], [2, 3]])
        mesh0 = dist.ProcessMesh([[0], [2]])
        mesh1 = dist.ProcessMesh([[1], [3]])

        input = paddle.ones(shape=[2, 3], dtype='float32')
        input = dist.shard_tensor(
            input, global_mesh, [dist.Shard(0), dist.Replicate()]
        )
        input.stop_gradient = False
        global_input = input + 1.0  # global_input: 2.0

        # forward on pp0
        input_pp0 = dist.reshard(
            global_input, mesh0, [dist.Shard(0), dist.Replicate()]
        )
        output = input_pp0 + 1.0  # output_pp0: 3.0

        # forward on pp1
        output = dist.reshard(output, mesh1, [dist.Shard(0), dist.Replicate()])
        input_pp1 = dist.reshard(
            global_input, mesh1, [dist.Shard(0), dist.Replicate()]
        )
        output = input_pp1 + output  # output_pp1: 5.0
        loss = paddle.sum(output)  # 30.0
        np.testing.assert_allclose(loss.numpy(), 30.0, rtol=1e-06, verbose=True)
        loss.backward()
        np.testing.assert_allclose(
            input.grad.numpy(),
            np.full(shape=(2, 3), fill_value=2.0, dtype=np.float32),
            rtol=1e-06,
            verbose=True,
        )

    def run_test_case(self):
        self.test_basic()
        self.test_split_dim1()


if __name__ == '__main__':
    TestSemiAutoParallel2DGlobalMeshReshard().run_test_case()
