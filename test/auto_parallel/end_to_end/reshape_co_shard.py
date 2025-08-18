# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestReshapeCoShard:
    def run_test_flatten(self):
        a = paddle.rand([2, 12, 8], "float32")
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])

        placements = [
            dist.Shard(0),
            dist.Shard(1),
        ]
        idx = dist.get_rank()
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [-1])
        np.testing.assert_equal(out.shape, [192])
        np.testing.assert_equal(
            str(out.placements[0]), 'Shard(dim=0, shard_order=0)'
        )
        np.testing.assert_equal(str(out.placements[1]), 'Replicate()')
        new_slice = (idx // 2,)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

        a = paddle.rand([4, 6, 8], "float32")
        placements = [
            dist.Shard(0, shard_order=0),
            dist.Shard(1, shard_order=1),
        ]
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [-1])
        np.testing.assert_equal(out.shape, [192])
        np.testing.assert_equal(
            str(out.placements[0]), 'Shard(dim=0, shard_order=0)'
        )
        np.testing.assert_equal(
            str(out.placements[1]), 'Shard(dim=0, shard_order=1)'
        )
        new_slice = (idx,)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

        placements = [
            dist.Shard(1),
            dist.Shard(2),
        ]
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [-1])
        np.testing.assert_equal(out.shape, [192])
        np.testing.assert_equal(str(out.placements[0]), 'Replicate()')
        np.testing.assert_equal(str(out.placements[1]), 'Replicate()')
        new_idx = slice(None)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_idx].numpy().flatten()
        )

    def run_test_split(self):
        a = paddle.rand([192], dtype='float32')
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        placements = [
            dist.Shard(0, shard_order=0),
            dist.Shard(0, shard_order=1),
        ]
        idx = dist.get_rank()
        input = dist.shard_tensor(a, mesh, placements)

        out = paddle.reshape(input, [4, 6, -1])
        np.testing.assert_equal(out.shape, [4, 6, 8])
        np.testing.assert_equal(
            str(out.placements[0]), 'Shard(dim=0, shard_order=0)'
        )
        np.testing.assert_equal(
            str(out.placements[1]), 'Shard(dim=0, shard_order=1)'
        )
        new_slice = (idx,)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [6, -1, 8])
        np.testing.assert_equal(out.shape, [6, 4, 8])
        np.testing.assert_equal(str(out.placements[0]), 'Replicate()')
        np.testing.assert_equal(str(out.placements[1]), 'Replicate()')
        new_slice = (slice(None),)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

    def run_test_combination(self):
        a = paddle.rand([4, 6, 8], "float32")
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        placements = [
            dist.Shard(0),
            dist.Shard(1),
        ]
        idx = dist.get_rank()
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [2, 12, 8])
        np.testing.assert_equal(out.shape, [2, 12, 8])
        np.testing.assert_equal(
            str(out.placements[0]), 'Shard(dim=0, shard_order=0)'
        )
        np.testing.assert_equal(str(out.placements[1]), 'Replicate()')
        new_slice = (idx // 2,)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

        placements = [
            dist.Shard(0, shard_order=0),
            dist.Shard(1, shard_order=1),
        ]
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [2, 12, 8])
        np.testing.assert_equal(out.shape, [2, 12, 8])
        np.testing.assert_equal(str(out.placements[0]), 'Replicate()')
        np.testing.assert_equal(str(out.placements[1]), 'Replicate()')
        new_slice = (slice(None),)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [12, 2, 8])
        np.testing.assert_equal(out.shape, [12, 2, 8])
        np.testing.assert_equal(
            str(out.placements[0]), 'Shard(dim=0, shard_order=0)'
        )
        np.testing.assert_equal(
            str(out.placements[1]), 'Shard(dim=0, shard_order=1)'
        )
        new_slice = slice(idx % 4 * 3, idx % 4 * 3 + 3)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

        placements = [
            dist.Shard(1),
            dist.Shard(2),
        ]
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [8, 6, 4])
        np.testing.assert_equal(out.shape, [8, 6, 4])
        np.testing.assert_equal(str(out.placements[0]), 'Replicate()')
        np.testing.assert_equal(str(out.placements[1]), 'Replicate()')
        new_slice = (slice(None),)
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

        placements = [
            dist.Shard(2, shard_order=0),
            dist.Shard(2, shard_order=1),
        ]
        input = dist.shard_tensor(a, mesh, placements)
        out = paddle.reshape(input, [24, 4, 2])
        np.testing.assert_equal(out.shape, [24, 4, 2])
        np.testing.assert_equal(
            str(out.placements[0]), 'Shard(dim=1, shard_order=0)'
        )
        np.testing.assert_equal(
            str(out.placements[1]), 'Shard(dim=1, shard_order=1)'
        )
        new_slice = (slice(None), dist.get_rank() % 4, slice(None))
        np.testing.assert_equal(
            out._local_value().numpy().flatten(), a[new_slice].numpy().flatten()
        )

    def run_test_case_main(self):
        self.run_test_flatten()
        self.run_test_split()
        self.run_test_combination()


if __name__ == '__main__':
    TestReshapeCoShard().run_test_case_main()
