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


class TestSavedTensorHookForSemiAutoParallel(unittest.TestCase):
    def run_test_case(self):
        def pack_hook(x):
            return x.numpy()

        def unpack_hook(x):
            return paddle.to_tensor(x)

        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        x_np = np.random.random(size=[64, 32]).astype(np.float32)
        y_np = np.random.random(size=[32, 48]).astype(np.float32)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False

        dist_x = dist.shard_tensor(x_np, mesh, [dist.Replicate()])
        dist_y = dist.shard_tensor(y_np, mesh, [dist.Replicate()])
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False

        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            z = paddle.matmul(x, y, False, False)

        with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
            dist_z = paddle.matmul(dist_x, dist_y, False, False)

        np.testing.assert_allclose(
            z.numpy(), dist_z.numpy(), rtol=1e-04, verbose=True
        )

        z.backward()
        dist_z.backward()

        np.testing.assert_allclose(
            x.grad.numpy(), dist_x.grad.numpy(), rtol=1e-04, verbose=True
        )
        np.testing.assert_allclose(
            y.grad.numpy(), dist_y.grad.numpy(), rtol=1e-04, verbose=True
        )


if __name__ == '__main__':
    TestSavedTensorHookForSemiAutoParallel().run_test_case()
