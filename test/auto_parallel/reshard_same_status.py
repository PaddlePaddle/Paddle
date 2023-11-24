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


def get_coord(mesh_list, rank):
    x = 0
    y = 0
    for sub_list in mesh_list:
        if rank in sub_list:
            y = sub_list.index(rank)
            return x, y
        x += 1
    return -1, -1


class TestReshardSameStatus:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")

    def test_diff_1d_mesh_shard(self, dev_ctx):
        paddle.seed(self._seeds)

        in_mesh_list = [0]
        out_mesh_list = [1]
        in_mesh = dist.ProcessMesh(in_mesh_list, dim_names=["x"])
        value = paddle.uniform(self._shape, self._dtype)

        in_shard_specs = [None for i in range(len(self._shape))]
        in_shard_specs[0] = "x"
        dist_attr = dist.DistAttr(mesh=in_mesh, sharding_specs=in_shard_specs)

        in_expected_local_tensor_list = paddle.split(
            value, num_or_sections=in_mesh.shape[0], axis=0
        )
        if dist.get_rank() in in_mesh_list:
            index = in_mesh_list.index(dist.get_rank()) % in_mesh.shape[0]
        elif dist.get_rank() in out_mesh_list:
            index = out_mesh_list.index(dist.get_rank()) % in_mesh.shape[0]

        input_tensor = dist.shard_tensor(value, dist_attr=dist_attr)

        if dist.get_rank() in in_mesh_list:
            # check the value of input tensor
            in_expected_local_tensor_list = paddle.split(
                value, num_or_sections=in_mesh.shape[0], axis=0
            )
            np.testing.assert_equal(
                input_tensor._local_value().numpy(),
                in_expected_local_tensor_list[index].numpy(),
            )

        out_mesh = dist.ProcessMesh(out_mesh_list, dim_names=["x"])
        out_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs[0] = "x"
        out_dist_attr = dist.DistAttr(
            mesh=out_mesh, sharding_specs=out_shard_specs
        )

        reshard_func = core.SameStatusReshardFunction()
        assert reshard_func.is_suitable(input_tensor, out_dist_attr)

        out = reshard_func.eval(dev_ctx, input_tensor, out_dist_attr)

        if dist.get_rank() in out_mesh_list:
            np.testing.assert_equal(
                out._local_value().numpy(),
                in_expected_local_tensor_list[index].numpy(),
            )

    def test_diff_nd_mesh_shard_partial(self, dev_ctx):
        paddle.seed(self._seeds)

        in_mesh_list = [[0], [1]]
        out_mesh_list = [[1], [0]]
        in_mesh = dist.ProcessMesh(in_mesh_list, dim_names=["x", "y"])
        value = paddle.uniform(self._shape, self._dtype)

        in_shard_specs = [None for i in range(len(self._shape))]
        in_shard_specs[0] = "x"
        dist_attr = dist.DistAttr(mesh=in_mesh, sharding_specs=in_shard_specs)
        dist_attr._set_partial_dims([1])

        input_tensor = dist.shard_tensor(value, dist_attr=dist_attr)

        in_expected_local_tensor_list = paddle.split(
            value, num_or_sections=in_mesh.shape[0], axis=0
        )

        in_flatten_list = [
            item for sub_list in in_mesh_list for item in sub_list
        ]
        out_flatten_list = [
            item for sub_list in out_mesh_list for item in sub_list
        ]

        in_x, in_y = get_coord(in_mesh_list, dist.get_rank())
        out_x, out_y = get_coord(out_mesh_list, dist.get_rank())

        if dist.get_rank() in in_flatten_list:
            if in_y == 0:
                np.testing.assert_equal(
                    input_tensor._local_value().numpy(),
                    in_expected_local_tensor_list[in_x].numpy(),
                )
            else:
                zeros = paddle.zeros(input_tensor._local_shape)
                np.testing.assert_equal(
                    input_tensor._local_value().numpy(),
                    zeros.numpy(),
                )

        out_mesh = dist.ProcessMesh(out_mesh_list, dim_names=["x", "y"])
        out_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs[0] = "x"
        out_dist_attr = dist.DistAttr(
            mesh=out_mesh, sharding_specs=out_shard_specs
        )
        out_dist_attr._set_partial_dims([1])

        reshard_func = core.SameStatusReshardFunction()
        assert reshard_func.is_suitable(input_tensor, out_dist_attr)

        out = reshard_func.eval(dev_ctx, input_tensor, out_dist_attr)

        if dist.get_rank() in out_flatten_list:
            if out_y == 0:
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    in_expected_local_tensor_list[out_x].numpy(),
                )
            else:
                zeros = paddle.zeros(out._local_shape)
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    zeros.numpy(),
                )

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        dev_ctx = core.DeviceContext.create(place)

        self.test_diff_1d_mesh_shard(dev_ctx)
        self.test_diff_nd_mesh_shard_partial(dev_ctx)


if __name__ == '__main__':
    TestReshardSameStatus().run_test_case()
