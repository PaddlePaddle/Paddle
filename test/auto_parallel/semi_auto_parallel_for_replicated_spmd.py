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


class TestReplicatedSPmdApiForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.seed(self._seed)
        np.random.seed(self._seed)

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def create_local_and_dist_tensor_pair(self, np_array, placements):
        local_t = paddle.to_tensor(np_array, dtype=np_array.dtype)

        dist_t = dist.shard_tensor(np_array, self._mesh, placements)

        local_t.stop_gradient = False
        dist_t.stop_gradient = False

        return local_t, dist_t

    # input: phi::Tensor
    # output: std::vector<phi::Tensor>
    def test_unbind(self):
        x = np.random.random(size=[2, 8]).astype("float32")
        local_in, dist_in = self.create_local_and_dist_tensor_pair(
            x, [dist.Shard(0)]
        )
        local_out1, local_out2 = paddle.unbind(local_in, axis=0)
        dist_out1, dist_out2 = paddle.unbind(dist_in, axis=0)
        self.check_tensor_eq(local_out1, dist_out1)
        self.check_tensor_eq(local_out2, dist_out2)

        local_out = paddle.add(local_out1, local_out2)
        dist_out = paddle.add(dist_out1, dist_out2)

        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in.grad, dist_in.grad)

    # input: paddle::optional<phi::Tensor>
    # output: phi::Tensor
    def test_expand_as(self):
        x1 = np.random.random(size=[2, 8]).astype("float32")
        x2 = np.random.random(size=[2, 2, 8]).astype("float32")
        local_in1, dist_in1 = self.create_local_and_dist_tensor_pair(
            x1, [dist.Shard(0)]
        )
        local_in2, dist_in2 = self.create_local_and_dist_tensor_pair(
            x2, [dist.Replicate()]
        )
        local_out = paddle.expand_as(local_in1, local_in2)
        dist_out = paddle.expand_as(dist_in1, dist_in2)
        self.check_tensor_eq(local_out, dist_out)

        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in1.grad, dist_in1.grad)

    # input: phi::Tensor
    # output: inplace paddle::optional<phi::Tensor>
    def test_adamax(self):
        dtype = np.float32
        mp_dtype = np.float32
        shape = [120, 320]

        beta1 = 0.78
        beta2 = 0.899
        epsilon = 1e-5
        param = np.random.random(shape).astype(dtype)
        grad = np.random.random(shape).astype(dtype)
        moment = np.random.random(shape).astype(dtype)
        inf_norm = np.random.random(shape).astype(dtype)
        master_param = param.astype(mp_dtype)

        lr = np.array([0.002]).astype("float32")
        beta1_pow = np.array([beta1**10]).astype("float32")

        local_param, dist_param = self.create_local_and_dist_tensor_pair(
            param, [dist.Shard(0)]
        )
        local_grad, dist_grad = self.create_local_and_dist_tensor_pair(
            grad, [dist.Shard(0)]
        )
        local_lr, dist_lr = self.create_local_and_dist_tensor_pair(
            lr, [dist.Replicate()]
        )
        (
            local_beta1_pow,
            dist_beta1_pow,
        ) = self.create_local_and_dist_tensor_pair(
            beta1_pow, [dist.Replicate()]
        )
        local_moment, dist_moment = self.create_local_and_dist_tensor_pair(
            moment, [dist.Shard(0)]
        )
        local_inf_norm, dist_inf_norm = self.create_local_and_dist_tensor_pair(
            inf_norm, [dist.Shard(0)]
        )
        (
            local_master_param,
            dist_master_param,
        ) = self.create_local_and_dist_tensor_pair(
            master_param, [dist.Replicate()]
        )

        (
            local_param_out,
            local_moment_out,
            local_inf_norm_out,
            local_master_param_out,
        ) = paddle._C_ops.adamax_(
            local_param,
            local_grad,
            local_lr,
            local_moment,
            local_inf_norm,
            local_beta1_pow,
            local_master_param,
            beta1,
            beta2,
            epsilon,
            True,
        )

        (
            dist_param_out,
            dist_moment_out,
            dist_inf_norm_out,
            dist_master_param_out,
        ) = paddle._C_ops.adamax_(
            dist_param,
            dist_grad,
            dist_lr,
            dist_moment,
            dist_inf_norm,
            dist_beta1_pow,
            dist_master_param,
            beta1,
            beta2,
            epsilon,
            True,
        )

        self.check_tensor_eq(local_param_out, dist_param_out)
        self.check_tensor_eq(local_moment_out, dist_moment_out)
        self.check_tensor_eq(local_inf_norm_out, dist_inf_norm_out)
        self.check_tensor_eq(local_master_param_out, dist_master_param_out)

    # multiple operators
    def test_mse_loss(self):
        x = np.random.random(size=[4, 4]).astype(self._dtype)
        y = np.random.random(size=[4]).astype(self._dtype)
        local_in, dist_in = self.create_local_and_dist_tensor_pair(
            x, [dist.Shard(0)]
        )
        local_label, dist_label = self.create_local_and_dist_tensor_pair(
            y, [dist.Replicate()]
        )

        mse_loss = paddle.nn.loss.MSELoss()
        local_out = mse_loss(local_in, local_label)
        dist_out = mse_loss(dist_in, dist_label)
        self.check_tensor_eq(local_out, dist_out)

        # test backward
        local_out.backward()
        dist_out.backward()
        np.testing.assert_equal(dist_in.grad._local_shape, [2, 4], verbose=True)
        np.testing.assert_equal(
            dist_in.grad.placements, [dist.Shard(0)], verbose=True
        )
        self.check_tensor_eq(local_in.grad, dist_in.grad)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_unbind()
        self.test_expand_as()
        self.test_adamax()
        self.test_mse_loss()


if __name__ == '__main__':
    TestReplicatedSPmdApiForSemiAutoParallel().run_test_case()
