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

import numpy as np

import paddle
import paddle.distributed as dist


class TestElementWiseCoShard:
    def run_unary_case_0(self):
        a = paddle.to_tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])

        placements = [
            dist.Shard(0, shard_order=0),
            dist.Shard(0, shard_order=1),
        ]
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.pow(input, 2)
        np.testing.assert_equal(out.shape, [4, 2])
        np.testing.assert_equal(
            out.placements[0], dist.Shard(dim=0, shard_order=0)
        )
        np.testing.assert_equal(
            out.placements[1], dist.Shard(dim=0, shard_order=1)
        )

    def run_unary_case_1(self):
        a = paddle.to_tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        y = paddle.to_tensor([2], dtype='float32')

        placements = [
            dist.Shard(0, shard_order=0),
            dist.Shard(0, shard_order=1),
        ]
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.pow(input, y)
        np.testing.assert_equal(out.shape, [4, 2])
        np.testing.assert_equal(
            out.placements[0], dist.Shard(dim=0, shard_order=0)
        )
        np.testing.assert_equal(
            out.placements[1], dist.Shard(dim=0, shard_order=1)
        )

    def run_unary_case_with_partial(self):
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        y = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        # TODO(ooooo): Test co_shard when matmul is supported.
        x_placements = [
            dist.Shard(1),
            dist.Replicate(),
        ]
        x = dist.shard_tensor(x, mesh, x_placements)
        y = dist.shard_tensor(
            y, mesh, [dist.Replicate() for _ in range(mesh.ndim)]
        )

        matmul_out = paddle.matmul(x, y)
        out = paddle.cast(matmul_out, 'float64')
        np.testing.assert_equal(out.shape, [2, 2])
        np.testing.assert_equal(out.placements[0], dist.Partial())
        np.testing.assert_equal(out.placements[1], dist.Replicate())

    def run_test_case_main(self):
        self.run_unary_case_0()
        self.run_unary_case_1()
        self.run_unary_case_with_partial()


if __name__ == '__main__':
    TestElementWiseCoShard().run_test_case_main()
