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


class TestReshardNdMesh:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([[0], [1]], dim_names=["x", "y"])
        self._other_mesh = dist.ProcessMesh([[1], [0]], dim_names=["x", "y"])

    def test_shard_partial_to_shard_replicated(self, dev_ctx):
        paddle.seed(self._seeds)
        value = paddle.uniform(self._shape, self._dtype)

        input_tensor = dist.shard_tensor(
            value, self._mesh, [dist.Partial(), dist.Shard(0)]
        )

        # check the shape of input tensor
        in_expected_shape = list(self._shape)
        in_expected_shape[0] = in_expected_shape[0] // self._mesh.shape[1]
        assert np.equal(input_tensor._local_shape, in_expected_shape).all()

        # check the value of input tensor
        in_expected_local_tensor_list = paddle.split(
            value, num_or_sections=self._mesh.shape[1], axis=0
        )
        index = dist.get_rank() % self._mesh.shape[1]
        if dist.get_rank() // self._mesh.shape[1] == 0:
            np.testing.assert_equal(
                input_tensor._local_value().numpy(),
                in_expected_local_tensor_list[index].numpy(),
            )
        else:
            zeros = paddle.zeros(in_expected_shape)
            np.testing.assert_equal(
                input_tensor._local_value().numpy(), zeros.numpy()
            )

        out = dist.reshard(
            input_tensor, self._mesh, [dist.Replicate(), dist.Shard(0)]
        )

        np.testing.assert_equal(
            out._local_value().numpy(),
            in_expected_local_tensor_list[index].numpy(),
        )

    def test_shard_partial_to_replicated(self, dev_ctx):
        paddle.seed(self._seeds)
        value = paddle.uniform(self._shape, self._dtype)

        input_tensor = dist.shard_tensor(
            value, self._mesh, [dist.Partial(), dist.Shard(0)]
        )

        # check the shape of input tensor
        in_expected_shape = list(self._shape)
        in_expected_shape[0] = in_expected_shape[0] // self._mesh.shape[1]
        assert np.equal(input_tensor._local_shape, in_expected_shape).all()

        # check the value of input tensor
        in_expected_local_tensor_list = paddle.split(
            value, num_or_sections=self._mesh.shape[1], axis=0
        )
        index = dist.get_rank() % self._mesh.shape[1]
        if dist.get_rank() // self._mesh.shape[1] == 0:
            np.testing.assert_equal(
                input_tensor._local_value().numpy(),
                in_expected_local_tensor_list[index].numpy(),
            )
        else:
            zeros = paddle.zeros(in_expected_shape)
            np.testing.assert_equal(
                input_tensor._local_value().numpy(), zeros.numpy()
            )

        out = dist.reshard(
            input_tensor, self._mesh, [dist.Replicate(), dist.Replicate()]
        )

        np.testing.assert_equal(out._local_value().numpy(), value.numpy())

    def test_partial_to_partial(self, dev_ctx):
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh, [dist.Partial(), dist.Replicate()]
        )

        if dist.get_rank() // self._mesh.shape[1] == 0:
            np.testing.assert_equal(
                input_tensor._local_value().numpy(), a.numpy()
            )
        else:
            zeros = paddle.zeros(self._shape)
            np.testing.assert_equal(
                input_tensor._local_value().numpy(), zeros.numpy()
            )

        out = dist.reshard(
            input_tensor, self._mesh, [dist.Replicate(), dist.Partial()]
        )

        if dist.get_rank() % self._mesh.shape[1] == 0:
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())
        else:
            zeros = paddle.zeros(self._shape)
            np.testing.assert_equal(out._local_value().numpy(), zeros.numpy())

        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, input_tensor._local_shape).all()

    def test_shard_to_shard(self, dev_ctx):
        a = paddle.ones(self._shape)

        in_shard_specs = [None for i in range(len(self._shape))]
        in_shard_specs[1] = "y"

        out_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs[0] = "x"

        input_tensor = dist.shard_tensor(
            a, self._mesh, [dist.Replicate(), dist.Shard(1)]
        )

        in_expected_shape = list(self._shape)
        in_expected_shape[1] = in_expected_shape[1] // self._mesh.shape[1]
        assert np.equal(input_tensor._local_shape, in_expected_shape).all()

        out = dist.reshard(
            input_tensor, self._mesh, [dist.Shard(0), dist.Replicate()]
        )

        out_expected_shape = list(self._shape)
        out_expected_shape[0] = out_expected_shape[0] // self._mesh.shape[0]
        assert np.equal(input_tensor._local_shape, in_expected_shape).all()

        assert np.equal(out.shape, input_tensor.shape).all()

    def test_partial_replicate_to_shard_replicated(self, dev_ctx):
        paddle.seed(self._seeds)
        a = paddle.randn(self._shape).astype(self._dtype)

        input_tensor = dist.shard_tensor(
            a, self._mesh, [dist.Partial(), dist.Replicate()]
        )
        out = dist.reshard(
            input_tensor, self._mesh, [dist.Shard(0), dist.Replicate()]
        )

        # check the value of input tensor
        out_expected_local_tensor_list = paddle.split(
            a, num_or_sections=self._mesh.shape[0], axis=0
        )
        index = dist.get_rank() % self._mesh.shape[0]
        np.testing.assert_equal(
            out._local_value().numpy(),
            out_expected_local_tensor_list[index].numpy(),
        )
        assert np.equal(out.shape, input_tensor.shape).all()

    def same_mesh_reshard(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        dev_ctx = core.DeviceContext.create(place)

        self.test_partial_to_partial(dev_ctx)
        self.test_shard_to_shard(dev_ctx)
        self.test_shard_partial_to_shard_replicated(dev_ctx)
        self.test_shard_partial_to_replicated(dev_ctx)

        if self._backend == "gpu":
            # reduce_scatter is not supported on CPU
            self.test_partial_replicate_to_shard_replicated(dev_ctx)

    def cross_mesh_reshard(self):
        a = paddle.zeros([20, 20])
        a = dist.shard_tensor(
            a,
            self._mesh,
            [
                dist.Partial(dist.ReduceType.kRedSum),
                dist.Partial(dist.ReduceType.kRedSum),
            ],
        )
        dist.reshard(a, self._other_mesh, [dist.Shard(0), dist.Shard(1)])

    def run_test_case(self):
        self.same_mesh_reshard()
        if self._backend == "gpu":
            self.cross_mesh_reshard()


if __name__ == '__main__':
    TestReshardNdMesh().run_test_case()
