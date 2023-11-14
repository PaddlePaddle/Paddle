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
from paddle.base import core


class TestReshardPToS:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_case(self):
        if self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())
        dev_ctx = core.DeviceContext.create(place)

        paddle.seed(self._seeds)
        value = paddle.uniform(self._shape, self._dtype)

        in_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs[self._shard] = "x"

        dist_attr = dist.DistAttr(
            mesh=self._mesh, sharding_specs=in_shard_specs
        )
        dist_attr._set_partial_dims([0])
        out_dist_attr = dist.DistAttr(
            mesh=self._mesh, sharding_specs=out_shard_specs
        )

        input_tensor = dist.shard_tensor(value, dist_attr=dist_attr)

        reshard_func = core.PToSReshardFunction()
        assert reshard_func.is_suitable(input_tensor, out_dist_attr)

        out_shape = list(self._shape)
        out_shape[self._shard] = out_shape[self._shard] // 2
        out_expected_local_tensor_list = paddle.split(
            value, num_or_sections=self._mesh.shape[0], axis=self._shard
        )

        out = reshard_func.eval(dev_ctx, input_tensor, out_dist_attr)

        np.testing.assert_equal(
            out._local_value().numpy(),
            out_expected_local_tensor_list[0].numpy()
            if dist.get_rank() == 0
            else out_expected_local_tensor_list[1].numpy(),
        )

        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, out_shape).all()


if __name__ == '__main__':
    TestReshardPToS().run_test_case()
