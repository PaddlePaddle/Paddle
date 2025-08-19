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


import unittest

import paddle.distributed as dist


class TestPlacementTypes(unittest.TestCase):
    def test_shard_eq_with_co_shard_order_zero(self):
        """
        Tests that a Shard is equal to a CoShard with shard_order=0.
        This confirms the "semantic equality" philosophy.
        """
        s1 = dist.Shard(0)
        s2 = dist.Shard(dim=0, shard_order=0)

        # 1. Test for symmetric equality
        self.assertEqual(
            s1, s2, "Shard(0) should be equal to Shard(dim=0, shard_order=0)"
        )
        self.assertEqual(s2, s1, "Equality should be symmetric")

        # 2. Test hash consistency
        self.assertEqual(
            hash(s1), hash(s2), "Hashes must be equal for equal objects"
        )

        # 3. Test behavior in a set
        placement_set = {s1, s2}
        self.assertEqual(
            len(placement_set),
            1,
            "A set should only contain one of the two equal objects",
        )

        # 4. Test behavior in a dict
        placement_dict = {s1: "value1"}
        self.assertIn(
            s2, placement_dict, "s2 should be found in a dict keyed by s1"
        )
        self.assertEqual(placement_dict[s2], "value1")

    def test_shard_neq_with_co_shard_order_non_zero(self):
        """
        Tests that a Shard is NOT equal to a CoShard with a non-zero shard_order.
        """
        s1 = dist.Shard(0)
        s2 = dist.Shard(dim=0, shard_order=1)

        # 1. Test for symmetric inequality
        self.assertNotEqual(
            s1,
            s2,
            "Shard(0) should NOT be equal to Shard(dim=0, shard_order=1)",
        )
        self.assertNotEqual(s2, s1, "Inequality should be symmetric")

        # 2. Test hash difference
        # Note: While not a strict requirement for non-equal objects to have different hashes,
        # a good hash function should minimize collisions. We test for non-collision here.
        self.assertNotEqual(
            hash(s1), hash(s2), "Hashes should be different for unequal objects"
        )

        # 3. Test behavior in a set
        placement_set = {s1, s2}
        self.assertEqual(
            len(placement_set), 2, "A set should contain two distinct objects"
        )

    def test_co_shard_eq(self):
        """
        Tests equality for two CoShard objects.
        """
        s1 = dist.Shard(dim=0, shard_order=1)
        s2 = dist.Shard(dim=0, shard_order=1)
        s3 = dist.Shard(dim=0, shard_order=2)

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_replicate_placement(self):
        """
        Tests equality and hash for Replicate placement.
        """
        r1 = dist.Replicate()
        r2 = dist.Replicate()
        s1 = dist.Shard(0)

        # 1. Test equality
        self.assertEqual(r1, r2, "Two Replicate objects should be equal")
        self.assertNotEqual(r1, s1, "Replicate should not be equal to Shard")

        # 2. Test hash consistency
        self.assertEqual(
            hash(r1),
            hash(r2),
            "Hashes of two Replicate objects should be equal",
        )

        # 3. Test behavior in a set
        placement_set: set[dist.Placement] = {r1, r2}
        self.assertEqual(
            len(placement_set),
            1,
            "A set should only contain one Replicate object",
        )
        placement_set.add(s1)
        self.assertEqual(
            len(placement_set),
            2,
            "The set should now contain two distinct objects",
        )

    def test_partial_placement(self):
        """
        Tests equality and hash for Partial placement.
        """
        p_sum1 = dist.Partial(dist.ReduceType.kRedSum)
        p_sum2 = dist.Partial(dist.ReduceType.kRedSum)
        p_avg = dist.Partial(dist.ReduceType.kRedAvg)
        r1 = dist.Replicate()

        # 1. Test equality
        self.assertEqual(
            p_sum1, p_sum2, "Two Partial(kRedSum) objects should be equal"
        )
        self.assertNotEqual(
            p_sum1,
            p_avg,
            "Partial(kRedSum) should not be equal to Partial(kRedAvg)",
        )
        self.assertNotEqual(
            p_sum1, r1, "Partial should not be equal to Replicate"
        )

        # 2. Test hash consistency
        self.assertEqual(hash(p_sum1), hash(p_sum2))
        self.assertNotEqual(hash(p_sum1), hash(p_avg))

        # 3. Test behavior in a set
        placement_set = {p_sum1, p_sum2}
        self.assertEqual(len(placement_set), 1)
        placement_set.add(p_avg)
        self.assertEqual(len(placement_set), 2)


if __name__ == '__main__':
    unittest.main()
