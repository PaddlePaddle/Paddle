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

import unittest

import numpy as np

import paddle
import paddle.distributed as dist


class TestSemiAutoParallelFunctionalInSingleCard(unittest.TestCase):
    def test_tensor_use_gpudnn(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        dist_tensor._use_gpudnn(False)

    def test_tensor_data_ptr(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        prt = dist_tensor.data_ptr()

    def test_tensor_offset(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        offset = dist_tensor._offset()

    def test_tensor_copy_to(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        dist_tensor._copy_to(paddle.CUDAPlace(0), True)

    def test_tensor__share_buffer_to(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        dense_tensor2 = paddle.randn([10, 10])
        to = dist.shard_tensor(dense_tensor2, mesh, [dist.Replicate()])
        dist_tensor._share_buffer_to(to)

    def test_tensor__is_shared_buffer_with(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        dense_tensor2 = paddle.randn([10, 10])
        to = dist.shard_tensor(dense_tensor2, mesh, [dist.Replicate()])
        dist_tensor._share_buffer_to(to)
        self.assertTrue(dist_tensor._is_shared_buffer_with(to))

    def test_tensor_strides(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dense_tensor = dense_tensor.reshape([20, 10])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        strides = dist_tensor.get_strides()
        is_contiguous = dist_tensor.is_contiguous()
        dist_tensor = dist_tensor.contiguous()

    def test_tensor_uva(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        place = paddle.CPUPlace()
        np_value = np.random.random(size=[10, 30]).astype('float32')
        dense_tensor = paddle.to_tensor(np_value, place=place)
        dist_tensor = dist.shard_tensor(
            dense_tensor, place=place, mesh=mesh, placements=[dist.Replicate()]
        )
        dist_tensor._uva()

    def test_tensor_properties(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor = paddle.randn([10, 20])
        dense_tensor = dense_tensor.reshape([20, 10])
        dist_tensor = dist.shard_tensor(dense_tensor, mesh, [dist.Replicate()])
        type = dist_tensor.type
        strides = dist_tensor.strides
        offsets = dist_tensor.offset

    def test_tensor_set_data(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dense_tensor_a = paddle.randn([10, 20])
        dist_tensor_a = dist.shard_tensor(
            dense_tensor_a, mesh, [dist.Replicate()]
        )

        dense_tensor_b = paddle.randn([5, 8])
        dist_tensor_b = dist.shard_tensor(
            dense_tensor_b, mesh, [dist.Replicate()]
        )

        dist_tensor_b.data = dist_tensor_a


if __name__ == "__main__":
    unittest.main()
