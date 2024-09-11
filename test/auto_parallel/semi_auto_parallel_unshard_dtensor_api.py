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

import paddle
import paddle.distributed as dist
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.distributed import Replicate, Shard


class TestUnshardDTensor(unittest.TestCase):
    def __init__(self):
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_dynamic(self):
        ori_tensor = paddle.rand([4, 1024, 512])
        d_tensor = dist.shard_tensor(ori_tensor, self.mesh, [Shard(0)])
        dense_tensor = dist.unshard_dtensor(d_tensor)
        self.assertListEqual(dense_tensor.shape, ori_tensor.shape)
        self.assertFalse(dense_tensor.is_dist())

        ori_parameter = paddle.create_parameter([1024, 512], dtype='float32')
        d_tensor = dist.shard_tensor(ori_parameter, self.mesh, [Shard(0)])
        dense_parameter = dist.unshard_dtensor(d_tensor)
        self.assertListEqual(dense_parameter.shape, ori_parameter.shape)
        self.assertFalse(dense_parameter.is_dist())
        self.assertTrue(
            isinstance(dense_parameter, paddle.base.framework.EagerParamBase)
        )

    @switch_to_static_graph
    def run_static(self):
        ori_tensor = paddle.static.data(
            name="input",
            shape=[4, 1024, 512],
            dtype='float32',
        )
        self.assertIsNone(ori_tensor.dist_attr())
        d_tensor = dist.shard_tensor(ori_tensor, self.mesh, [Shard(0)])
        self.assertTrue(d_tensor.is_dist_dense_tensor_type())
        self.assertEqual(d_tensor.dist_attr().process_mesh, self.mesh)

        dense_tensor = dist.unshard_dtensor(d_tensor)
        self.assertListEqual(dense_tensor.shape, ori_tensor.shape)
        self.assertFalse(d_tensor.is_dist_dense_tensor_type())

    def run_dy2static(self):
        @paddle.jit.to_static(full_graph=True)
        def unshard_func():
            mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            input = paddle.rand([4, 1024, 512])
            d_tensor = dist.shard_tensor(input, mesh, [Replicate()])
            dense_tensor = dist.unshard_dtensor(d_tensor)
            return input, dense_tensor

        dy_ori_tensor, dy_dense_tensor = unshard_func()
        st_ori_tensor = unshard_func.outputs[0]
        st_dense_tensor = unshard_func.outputs[1]
        self.assertListEqual(dy_dense_tensor.shape, dy_ori_tensor.shape)
        self.assertFalse(dy_dense_tensor.is_dist())

        self.assertIsNone(st_dense_tensor.dist_attr())

    def run_test_cases(self):
        self.run_dynamic()
        self.run_static()
        # self.run_dy2static() ## not support


if __name__ == "__main__":
    TestUnshardDTensor().run_test_cases()
