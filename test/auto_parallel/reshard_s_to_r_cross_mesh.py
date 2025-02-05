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


class TestReshardSToRCrossMesh:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")

        self._in_mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._out_mesh = dist.ProcessMesh([1, 0], dim_names=["x"])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        dev_ctx = core.DeviceContext.create(place)
        a = paddle.randn(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._in_mesh, [dist.Shard(self._shard)]
        )
        out = dist.reshard(input_tensor, self._out_mesh, [dist.Replicate()])

        out_shape = list(self._shape)
        if out_shape[self._shard] % 2 == 0:
            split_shape = self._in_mesh.shape[0]
        else:
            split_shape = [
                out_shape[self._shard] // 2 + 1,
                out_shape[self._shard] // 2,
            ]

        in_expected_local_tensor_list = paddle.split(
            out._local_value(), num_or_sections=split_shape, axis=self._shard
        )

        np.testing.assert_equal(
            input_tensor._local_value().numpy(),
            in_expected_local_tensor_list[dist.get_rank()].numpy(),
        )

        assert np.equal(out.shape, out_shape).all()


if __name__ == '__main__':
    TestReshardSToRCrossMesh().run_test_case()
