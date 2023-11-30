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
from paddle import nn
from paddle.distributed import Partial, Replicate, Shard


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
        self.test_case_r_to_s()
        self.test_case_forward_and_backward()

    def test_case_p_to_r(self):
        a = paddle.ones(self._shape)
        in_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs = [None for i in range(len(self._shape))]

        input_tensor = dist.shard_tensor(a, self._mesh, [Partial()])
        output_tensor = dist.reshard(input_tensor, self._mesh, [Replicate()])

        input_tensor = dist.shard_tensor(a, self._mesh, [Replicate()])
        assert np.equal(output_tensor.shape, input_tensor.shape).all()
        np.testing.assert_equal(output_tensor._local_value().numpy(), a.numpy())

    def test_case_r_to_s(self):
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(a, self._mesh, [Replicate()])
        output_tensor = dist.reshard(input_tensor, self._mesh, [Shard(0)])

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

    def test_case_forward_and_backward(self):
        if self._backend == "cpu":
            return

        np.random.seed(1901)
        input_numpy = np.random.random(self._shape).astype("float32")
        label_numpy = np.random.random(self._shape).astype('float32')

        local_input = paddle.to_tensor(input_numpy)
        dist_input = dist.shard_tensor(
            paddle.to_tensor(input_numpy),
            dist.ProcessMesh([0, 1], dim_names=["x"]),
            [Replicate()],
        )

        local_input.stop_gradient = False
        dist_input.stop_gradient = False

        local_output = local_input + paddle.ones(self._shape)
        dist_output = dist_input + dist.shard_tensor(
            paddle.ones(self._shape),
            dist.ProcessMesh([0, 1], dim_names=["x"]),
            [Replicate()],
        )
        dist_output.stop_gradient = False

        dist_output = dist.reshard(
            dist_output, dist.ProcessMesh([0, 1], dim_names=["x"]), [Shard(0)]
        )

        local_label = paddle.to_tensor(label_numpy)
        dist_label = dist.shard_tensor(
            paddle.to_tensor(label_numpy),
            dist.ProcessMesh([0, 1], dim_names=["x"]),
            [Shard(0)],
        )

        local_loss_fn = nn.MSELoss()
        dist_loss_fn = nn.MSELoss()

        local_loss = local_loss_fn(local_output, local_label)
        dist_loss = dist_loss_fn(dist_output, dist_label)

        np.testing.assert_allclose(
            local_loss.numpy(), dist_loss.numpy(), rtol=1e-5, atol=1e-5
        )

        local_loss.backward()
        dist_loss.backward()
        np.testing.assert_allclose(
            local_input.grad.numpy(),
            dist_input.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == '__main__':
    TestReshardAPI().run_test_cases()
