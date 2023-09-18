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
import paddle.nn.functional as F


class TestReplicatedSPmdApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.seed(2023)
        np.random.seed(2023)

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05)

    def create_local_and_dist_tensor_pair(self, np_array):
        local_t = paddle.to_tensor(np_array, dtype=np_array.dtype)

        dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None])
        dist_t = dist.shard_tensor(np_array, dist_attr=dist_attr)

        local_t.stop_gradient = True
        dist_t.stop_gradient = True

        return local_t, dist_t

    def test_relu(self):
        x = np.random.random(size=[4, 4]).astype("float32")
        local_in, dist_in = self.create_local_and_dist_tensor_pair(x)
        local_out = F.relu(local_in)
        dist_out = F.relu(dist_in)
        # verify output dist attr and value
        np.testing.assert_equal(
            dist_out.dist_attr.dims_mapping, [-1, -1], verbose=True
        )
        self.check_tensor_eq(local_out, dist_out)

    def test_mse_loss(self):
        input = dist.dtensor_from_fn(
            paddle.randn,
            dist.DistAttr(mesh=self._mesh, sharding_specs=['x', None]),
            [4, 4],
            dtype=self._dtype,
        )
        label = dist.dtensor_from_fn(
            paddle.randn,
            dist.DistAttr(mesh=self._mesh, sharding_specs=[None]),
            [4],
            self._dtype,
        )

        mes_loss = paddle.nn.loss.MSELoss()
        out = mes_loss(input, label)
        print(out)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        # self.test_relu()
        self.test_mse_loss()


if __name__ == '__main__':
    TestReplicatedSPmdApiForSemiAutoParallel().run_test_case()
