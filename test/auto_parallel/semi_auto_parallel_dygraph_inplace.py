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


class TestInplaceForSemiAutoParallel(unittest.TestCase):
    def run_test_case(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        x_np = np.random.random(size=[64, 32]).astype(np.float32)
        y_np = np.random.random(size=[32, 48]).astype(np.float32)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False

        x_dist_attr = [dist.Shard(0)]
        y_dist_attr = [dist.Replicate()]

        dist_x = dist.shard_tensor(x_np, mesh, x_dist_attr)
        dist_y = dist.shard_tensor(y_np, mesh, y_dist_attr)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False
        dist_x = dist_x.add(dist_x)
        dist_y = dist_y.add(dist_y)
        dist_out = paddle.matmul(
            dist_x, dist_y, transpose_x=False, transpose_y=False
        )
        dist_x.add_(dist_x)
        dist_y.add_(dist_y)

        with self.assertRaisesRegex(
            RuntimeError,
            "received tensor_version:1 != wrapper_version_snapshot:0",
        ):
            dist_out.backward()


if __name__ == '__main__':
    TestInplaceForSemiAutoParallel().run_test_case()
