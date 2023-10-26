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


class TestReshardAPI:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._shard = eval(os.getenv("shard"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_cases(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        self.test_case_p_to_r()

    def test_case_p_to_r(self):
        a = paddle.ones(self._shape)
        in_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs = [None for i in range(len(self._shape))]
        dist_attr = dist.DistAttr(
            mesh=self._mesh, sharding_specs=in_shard_specs
        )
        dist_attr._set_partial_dims([0])
        out_dist_attr = dist.DistAttr(
            mesh=self._mesh, sharding_specs=out_shard_specs
        )

        input_tensor = dist.shard_tensor(a, dist_attr=dist_attr)
        output_tensor = dist.reshard(input_tensor, dist_attr=out_dist_attr)

        input_tensor = dist.shard_tensor(a, dist_attr=dist_attr)
        assert np.equal(output_tensor.shape, input_tensor.shape).all()
        np.testing.assert_equal(output_tensor._local_value().numpy(), a.numpy())

    def test_case_r_to_s(self):
        a = paddle.ones(self._shape)
        in_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs[self._shard] = "x"
        dist_attr = dist.DistAttr(
            mesh=self._mesh, sharding_specs=in_shard_specs
        )
        out_dist_attr = dist.DistAttr(
            mesh=self._mesh, sharding_specs=out_shard_specs
        )

        input_tensor = dist.shard_tensor(a, dist_attr=dist_attr)
        output_tensor = dist.reshard(input_tensor, dist_attr=out_dist_attr)

        out_shape = list(self._shape)
        if out_shape[self._shard] % 2 == 0:
            out_shape[self._shard] = out_shape[self._shard] // 2
            np.testing.assert_equal(output_tensor.numpy(), input_tensor.numpy())
        else:
            out_shape[self._shard] = (
                out_shape[self._shard] // 2
                if dist.get_rank() == 1
                else out_shape[self._shard] // 2 + 1
            )

        assert np.equal(output_tensor.shape, input_tensor.shape).all()
        assert np.equal(output_tensor._local_shape, out_shape).all()


if __name__ == '__main__':
    TestReshardAPI().run_test_cases()
