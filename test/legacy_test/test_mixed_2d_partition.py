# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Single-process tests for MixedShardingOptimizer's 2D owner assignment.

``_partition_2d_parameters_machine_balanced`` reads nothing but
``self.comm_buffer_size_MB`` and ``self._global_rank`` and takes the parameters
as plain ``(name, numel, dtype_str, itemsize)`` tuples, so it can be exercised
directly -- no communication groups, no accelerators, no launcher. The
multi-process plumbing around it (the two ``all_gather_object`` calls and every
rank agreeing on the mapping) is covered by
test/collective/fleet/test_parallel_dygraph_mixed_sharding.py.
"""

import unittest

from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
    MixedShardingOptimizer,
)

MOE_SHARDING, PP, EP, CARDS_PER_MACHINE = 2, 2, 4, 8


class _StubPartitioner:
    """Carries only the state the partitioner actually reads."""

    _partition_2d_parameters_machine_balanced = (
        MixedShardingOptimizer._partition_2d_parameters_machine_balanced
    )

    def __init__(self, comm_buffer_size_MB, global_rank=1):
        self.comm_buffer_size_MB = comm_buffer_size_MB
        # Non-zero by default so the rank-0 placement summary stays out of the
        # test output.
        self._global_rank = global_rank


def _hybrid_layout():
    """pp2 x ep4 x moe_sharding2 = 16 ranks over two 8-card machines.

    Dense params shard across everything but the pipe axis, so there is one
    dense group per PP stage; expert params shard along moe_sharding only, so
    there is one group per (stage, expert). Both kinds straddle both machines.
    """

    def rank_of(sharding_idx, pp_idx, ep_idx):
        return sharding_idx * PP * EP + pp_idx * EP + ep_idx

    dense_groups = [
        tuple(
            sorted(
                rank_of(s, pp_idx, e)
                for s in range(MOE_SHARDING)
                for e in range(EP)
            )
        )
        for pp_idx in range(PP)
    ]
    moe_groups = [
        tuple(sorted(rank_of(s, pp_idx, e) for s in range(MOE_SHARDING)))
        for pp_idx in range(PP)
        for e in range(EP)
    ]
    rank_to_machine = {
        rank: f"host{rank // CARDS_PER_MACHINE}"
        for rank in range(MOE_SHARDING * PP * EP)
    }
    return dense_groups, moe_groups, rank_to_machine


def _color_group_info(dense_groups, moe_groups):
    """Six dense weights per stage, two expert weights per (stage, expert).

    Two dtypes, because a FusedCommBuffer never mixes them and the partitioner
    therefore buckets per (color, dtype).
    """
    info = {}
    for stage, group_ranks in enumerate(dense_groups):
        params = [
            (f"stage{stage}.layer{i}.w", 4096 * 4096, "bfloat16", 2)
            for i in range(5)
        ]
        params.append((f"stage{stage}.head.w", 4096 * 4096, "float32", 4))
        info[(None, group_ranks)] = params
    for idx, group_ranks in enumerate(moe_groups):
        info[("moe_expert", group_ranks)] = [
            (f"expert{idx}.w{i}", 2048 * 4096, "bfloat16", 2) for i in range(2)
        ]
    return info


class TestMachineBalancedPartition(unittest.TestCase):
    def setUp(self):
        self.dense, self.moe, self.rank_to_machine = _hybrid_layout()
        self.info = _color_group_info(self.dense, self.moe)
        self.size_of = {
            p[0]: p[1] * p[3] for ps in self.info.values() for p in ps
        }

    def _run(self, comm_buffer_size_MB, global_rank=1):
        return _StubPartitioner(
            comm_buffer_size_MB, global_rank
        )._partition_2d_parameters_machine_balanced(
            self.info, self.rank_to_machine
        )

    def test_every_param_owned_exactly_once(self):
        """One entry per group (not per color), every name placed once.

        ``None`` names one dense group per PP stage and ``moe_expert`` one per
        (stage, expert), so keying on the color alone would collapse them and
        lose every group but the last. Owners are local indices into the
        group's rank list, since that is what ``comm_group.ranks[...]`` wants.
        """
        for buffer_mb in (0, 64, 256):
            with self.subTest(buffer=buffer_mb):
                result = self._run(buffer_mb)
                self.assertEqual(set(result), set(self.info))
                owner_of = {}
                for key, ranks_map in result.items():
                    group_ranks = key[1]
                    self.assertEqual(
                        set(ranks_map), set(range(len(group_ranks)))
                    )
                    for local_rank, names in ranks_map.items():
                        for name in names:
                            self.assertNotIn(name, owner_of)
                            owner_of[name] = group_ranks[local_rank]
                self.assertEqual(set(owner_of), set(self.size_of))

    def test_owner_count_follows_bucket_volume(self):
        """A bucket bigger than one comm buffer must spread over more ranks.

        Owner count comes from ``numel * 4`` (the fp32 main_grad width that is
        actually reduced) over the comm buffer size, capped by the group. The
        dense bfloat16 bucket is 5 x 4096 x 4096 elements = 320MB of fp32 grad,
        so a 64MB buffer asks for 6 owners out of 8 while 256MB asks for 2; the
        single-parameter fp32 bucket is 64MB and needs exactly one owner at
        256MB. Without the volume term the per-rank loop would never run twice.
        """

        def active_ranks(numel, group_size, buffer_mb):
            total_mb = numel * 4 / (1024**2)
            buffer_mb = buffer_mb if buffer_mb > 0 else 256
            return min(max(1, int(total_mb / buffer_mb) + 1), group_size)

        for buffer_mb in (64, 256):
            result = self._run(buffer_mb)
            for key in [k for k in self.info if k[0] is None]:
                group_size = len(key[1])
                for dtype_str, numel, count, is_head in (
                    ("bfloat16", 5 * 4096 * 4096, 5, False),
                    ("float32", 4096 * 4096, 1, True),
                ):
                    # A bucket can never occupy more ranks than it has params.
                    expected = min(
                        active_ranks(numel, group_size, buffer_mb), count
                    )
                    owners = {
                        rank
                        for rank, names in result[key].items()
                        if any(n.endswith("head.w") == is_head for n in names)
                    }
                    with self.subTest(buffer=buffer_mb, dtype=dtype_str):
                        self.assertLessEqual(len(owners), expected)
                        if expected == 1:
                            self.assertEqual(len(owners), 1)
                        else:
                            self.assertGreater(len(owners), 1)

    def test_load_is_spread_over_every_machine(self):
        """Both machines must carry owners, and carry about the same volume.

        Taking the first ``active_ranks`` local indices of each group instead
        would pile every owner of both dense stages and all eight expert groups
        onto host0, i.e. a spread of 1.0.
        """
        for buffer_mb in (64, 256):
            with self.subTest(buffer=buffer_mb):
                machine_bytes = dict.fromkeys(
                    set(self.rank_to_machine.values()), 0
                )
                for key, ranks_map in self._run(buffer_mb).items():
                    for local_rank, names in ranks_map.items():
                        machine = self.rank_to_machine[key[1][local_rank]]
                        for name in names:
                            machine_bytes[machine] += self.size_of[name]
                loads = list(machine_bytes.values())
                self.assertTrue(all(loads), machine_bytes)
                self.assertLessEqual(
                    (max(loads) - min(loads)) / max(loads), 0.05, machine_bytes
                )

    def test_result_does_not_depend_on_merge_order(self):
        """Every rank must reach the same answer whatever order it merged in.

        Owners become the reduce dst / broadcast src, so a rank that ordered
        the gathered groups differently and derived different owners would hang
        the job.
        """
        reversed_info = dict(reversed(list(self.info.items())))
        for buffer_mb in (64, 256):
            with self.subTest(buffer=buffer_mb):
                actual = _StubPartitioner(
                    buffer_mb, 0
                )._partition_2d_parameters_machine_balanced(
                    reversed_info, self.rank_to_machine
                )
                self.assertEqual(actual, self._run(buffer_mb))

    def test_exhausted_machine_is_discarded(self):
        """A machine with no free rank left drops out of the candidates.

        host1 wins the first owner (it is the least crowded machine with two
        free ranks), and the 4MB fp32 grad wants two owners out of three, so
        the second pass has to reject a machine whose free list is empty.
        """
        group_key = (None, (0, 1, 2))
        result = _StubPartitioner(
            comm_buffer_size_MB=1, global_rank=1
        )._partition_2d_parameters_machine_balanced(
            {group_key: [("weight", 1024 * 1024, "float32", 4)]},
            {0: "host0", 1: "host1", 2: "host1"},
        )
        self.assertEqual(set(result[group_key]), {0, 1, 2})
        placed = [n for names in result[group_key].values() for n in names]
        self.assertEqual(placed, ["weight"])


if __name__ == "__main__":
    unittest.main()
