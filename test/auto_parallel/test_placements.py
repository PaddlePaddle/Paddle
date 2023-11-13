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
from paddle.base import core


# NOTE(shenliang03): This is a temporary test to test DistTensor's addition of
# placements and process_mesh constructors, and will be modified as a whole later.
class TestShardTensorWithPlacements(unittest.TestCase):
    def setUp(self):
        self.mesh = dist.ProcessMesh(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=["x", "y"]
        )
        self.placements = [core.Replicate(), core.Replicate()]

    def test_placements(self):
        shard = core.Shard(1)
        replicate = core.Replicate()
        partial = core.Partial()

        self.assertEqual(shard.get_dim(), 1)
        self.assertEqual(shard.is_shard(), True)
        self.assertEqual(shard.is_replicated(), False)
        self.assertEqual(shard.is_partial(), False)
        self.assertEqual(str(shard), "Shard(dim=1)")

        self.assertEqual(replicate.is_shard(), False)
        self.assertEqual(replicate.is_replicated(), True)
        self.assertEqual(replicate.is_partial(), False)
        self.assertEqual(str(replicate), "Replicate()")

        self.assertEqual(partial.is_shard(), False)
        self.assertEqual(partial.is_replicated(), False)
        self.assertEqual(partial.is_partial(), True)
        self.assertEqual(str(partial), "Partial(reduce_type=SUM)")

        shard_1 = core.Shard(1)
        replicate_1 = core.Replicate()
        partial_1 = core.Partial()

        self.assertEqual(shard_1, shard)
        self.assertEqual(replicate_1, replicate)
        self.assertEqual(partial_1, partial)

        self.assertEqual(hash(shard_1), hash(shard))
        self.assertEqual(hash(replicate_1), hash(replicate))
        self.assertEqual(hash(partial_1), hash(partial))

    def test_placements_tensor_construct(self):
        tensor = paddle.rand([2, 10, 4])
        srp_tensor = paddle.Tensor(
            tensor, process_mesh=self.mesh, placements=self.placements
        )

        self.assertEqual(srp_tensor.dist_attr.process_mesh, self.mesh)
        self.assertEqual(srp_tensor.dist_attr.dims_mapping, [-1, -1, -1])
        self.assertTrue(srp_tensor.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(srp_tensor.dist_attr.is_annotated("dims_mapping"))

        dist_attr = dist.DistAttr(
            mesh=self.mesh, sharding_specs=[None, None, None]
        )

        dist_attr_tensor = paddle.Tensor(tensor, dist_attr=dist_attr)

        self.assertEqual(
            dist_attr_tensor.dist_attr.dims_mapping,
            srp_tensor.dist_attr.dims_mapping,
        )
        self.assertEqual(
            dist_attr_tensor.dist_attr.process_mesh,
            srp_tensor.dist_attr.process_mesh,
        )


if __name__ == "__main__":
    unittest.main()
