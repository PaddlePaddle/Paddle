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
from paddle.framework import core


class TestReshardRToX:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._in_mesh = dist.ProcessMesh([0], dim_names=["x"])
        self._out_mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def _set_place(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())
        dev_ctx = core.DeviceContext.create(place)

    def test_r_to_s(self):
        self._set_place()

        a = paddle.ones(self._shape)
        input_tensor = dist.shard_tensor(a, self._in_mesh, [dist.Replicate()])
        out = dist.reshard(
            input_tensor, self._out_mesh, [dist.Shard(self._shard)]
        )

        out_shape = list(self._shape)
        if out_shape[self._shard] % 2 == 0:
            out_shape[self._shard] = out_shape[self._shard] // 2
            np.testing.assert_equal(out.numpy(), a.numpy())
        else:
            out_shape[self._shard] = (
                out_shape[self._shard] // 2
                if dist.get_rank() == 1
                else out_shape[self._shard] // 2 + 1
            )
        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, out_shape).all()

    def test_r_to_r(self):
        self._set_place()

        a = paddle.ones(self._shape)
        input_tensor = dist.shard_tensor(a, self._in_mesh, [dist.Replicate()])
        out = dist.reshard(input_tensor, self._out_mesh, [dist.Replicate()])

        if dist.get_rank() == 0:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())

    def test_r_to_p(self):
        self._set_place()

        a = paddle.ones(self._shape)
        input_tensor = dist.shard_tensor(a, self._in_mesh, [dist.Replicate()])
        out = dist.reshard(
            input_tensor,
            self._out_mesh,
            [dist.Partial(dist.ReduceType.kRedSum)],
        )

        if dist.get_rank() == 0:
            np.testing.assert_equal(
                out._local_value().numpy(), input_tensor.numpy()
            )
        else:
            zeros = paddle.zeros(self._shape)
            np.testing.assert_equal(out._local_value().numpy(), zeros.numpy())

        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, input_tensor._local_shape).all()

    def run_test_case(self):
        self.test_r_to_s()
        self.test_r_to_r()
        self.test_r_to_p()


if __name__ == '__main__':
    TestReshardRToX().run_test_case()
