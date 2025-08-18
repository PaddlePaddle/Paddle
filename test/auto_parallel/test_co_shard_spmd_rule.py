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


class TestCoShardSPMDRule:
    """
    Unit tests for co_shard spmd rule.
    """

    def test_co_shard_for_binary_elementwise(self):
        a = paddle.randn([64, 64], dtype='float32')
        b = paddle.randn([64, 64], dtype='float32')
        # [[0],[1]],[[0,1],[]]->[[0],[1]], [[0],[1]], [[0],[1]]
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        placements1 = [dist.Shard(0), dist.Shard(1)]
        placements2 = [
            dist.Shard(dim=0, shard_order=0),
            dist.Shard(dim=0, shard_order=1),
        ]
        x = dist.shard_tensor(a, mesh, placements1)
        y = dist.shard_tensor(b, mesh, placements2)
        out = paddle.add(x, y)
        np.testing.assert_equal(str(x.placements[0]), "Shard(dim=0)")
        np.testing.assert_equal(str(x.placements[1]), "Shard(dim=1)")
        np.testing.assert_equal(str(y.placements[0]), "Shard(dim=0)")
        np.testing.assert_equal(str(y.placements[1]), "Shard(dim=1)")
        np.testing.assert_equal(out.shape, [64, 64])
        np.testing.assert_equal(str(out.placements[0]), "Shard(dim=0)")
        np.testing.assert_equal(str(out.placements[1]), "Shard(dim=1)")
        # [[0],[]], [[1],[]] ->[[0,1],[]], [[0,1],[]], [[0,1],[]]
        placements1 = [dist.Shard(0), dist.Replicate()]
        placements2 = [dist.Shard(1), dist.Replicate()]
        x = dist.shard_tensor(a, mesh, placements1)
        y = dist.shard_tensor(b, mesh, placements2)
        out = paddle.add(x, y)
        np.testing.assert_equal(
            str(x.placements[0]), dist.Shard(dim=0, shard_order=0)
        )
        np.testing.assert_equal(
            str(x.placements[1]), dist.Shard(dim=0, shard_order=1)
        )
        np.testing.assert_equal(
            str(y.placements[0]), dist.Shard(dim=0, shard_order=0)
        )
        np.testing.assert_equal(
            str(y.placements[1]), dist.Shard(dim=0, shard_order=1)
        )
        np.testing.assert_equal(out.shape, [64, 64])
        np.testing.assert_equal(
            str(out.placements[0]), dist.Shard(dim=0, shard_order=0)
        )
        np.testing.assert_equal(
            str(out.placements[1]), dist.Shard(dim=0, shard_order=1)
        )

    def test_co_shard_for_layernorm(self):
        x = paddle.rand((64, 32, 128, 128))
        layer_norm = paddle.nn.LayerNorm(x.shape[1:])

        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        # [[0],[1],[],[]] -> [[0,1],[],[]] | [[0,1],[],[]]
        placements = [
            dist.Shard(dim=0),
            dist.Shard(dim=1),
            dist.Replicate(),
            dist.Replicate(),
        ]
        input = dist.shard_tensor(x, mesh, placements)
        out = layer_norm(input)
        np.testing.assert_equal(
            str(input.placements[0]), dist.Shard(dim=0, shard_order=0)
        )
        np.testing.assert_equal(
            str(input.placements[1]), dist.Shard(dim=0, shard_order=1)
        )
        np.testing.assert_equal(
            str(out.placements[0]), dist.Shard(dim=0, shard_order=0)
        )
        np.testing.assert_equal(
            str(out.placements[1]), dist.Shard(dim=0, shard_order=1)
        )

    def run_test_case_main(self):
        self.test_co_shard_for_binary_elementwise()
        self.test_co_shard_for_layernorm()


if __name__ == "__main__":
    TestCoShardSPMDRule().run_test_case_main()
