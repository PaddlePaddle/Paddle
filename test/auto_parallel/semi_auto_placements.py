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
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core


class TestDistTensorSRP(unittest.TestCase):
    def setUp(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def run_test_placements(self):
        self.placements = [core.Replicate(), core.Replicate()]
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
        self.assertNotEqual(shard_1, replicate)
        self.assertNotEqual(shard_1, partial)

        self.assertEqual(hash(shard_1), hash(shard))
        self.assertEqual(hash(replicate_1), hash(replicate))
        self.assertEqual(hash(partial_1), hash(partial))

    def run_test_dist_tensor(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
            place = paddle.CPUPlace()
        elif self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())

        tensor = paddle.rand([2, 10])
        srp_tensor = paddle.Tensor(
            tensor, process_mesh=self._mesh, placements=core.Shard(0)
        )

        self.assertEqual(srp_tensor.dist_attr.process_mesh, self._mesh)
        self.assertEqual(srp_tensor.dist_attr.dims_mapping, [0, -1])
        self.assertEqual(srp_tensor.num_shard, 2)

        self.assertTrue(srp_tensor.dist_attr.is_annotated("process_mesh"))
        self.assertTrue(srp_tensor.dist_attr.is_annotated("dims_mapping"))

        dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=["x", None])

        dist_attr_tensor = paddle.Tensor(tensor, dist_attr=dist_attr)

        self.assertEqual(
            dist_attr_tensor.dist_attr.dims_mapping,
            srp_tensor.dist_attr.dims_mapping,
        )
        self.assertEqual(
            dist_attr_tensor.dist_attr.process_mesh,
            srp_tensor.dist_attr.process_mesh,
        )
        np.testing.assert_equal(
            dist_attr_tensor._local_value().numpy(),
            srp_tensor._local_value().numpy(),
        )

    def test_case(self):
        self.run_test_placements()
        self.run_test_dist_tensor()


if __name__ == "__main__":
    unittest.main()
