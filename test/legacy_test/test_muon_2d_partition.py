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

"""Single-process tests for MuonShardingOptimizer's 2D partitioners.

Both map every 2D (Muon) parameter to an owner rank, and which one runs is
picked by the ``machine_balanced_2d_partition`` sharding config.
``_partition_2d_parameters_machine_balanced`` reads nothing but
``self.comm_buffer_size_MB`` and ``self._global_rank``, and takes the parameters
as plain ``(name, numel, dtype_str, itemsize)`` tuples; the legacy
``_partition_2d_parameters`` reads only ``self.comm_buffer_size_MB`` and each
parameter's ``shape``/``dtype``. So both can be exercised directly -- no
communication groups, no accelerators, no launcher. Most tests here put every
rank on one machine, which makes the machine-level terms constant and leaves the
per-rank packing they are about. The multi-process behaviour they feed into is
covered by
test/collective/fleet/test_muon_sharding_mixed_dtype_partition.py.
"""

import unittest
from functools import reduce

from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)

WORLD_SIZES = (1, 2, 4, 8, 16, 32)
BUFFER_SIZES_MB = (0, 1, 64, 128, 256, 512)
ITEMSIZE = {"bfloat16": 2, "float16": 2, "float32": 4}


class _StubParam:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class _StubPartitioner:
    """Carries only the state the partitioners actually read."""

    _partition_2d_parameters = MuonShardingOptimizer._partition_2d_parameters
    _partition_2d_parameters_machine_balanced = (
        MuonShardingOptimizer._partition_2d_parameters_machine_balanced
    )

    def __init__(self, comm_buffer_size_MB, global_rank=1):
        self.comm_buffer_size_MB = comm_buffer_size_MB
        # Non-zero by default so the rank-0 placement summary stays out of the
        # output of the hundreds of subTest combinations below.
        self._global_rank = global_rank


def _numel(param):
    return reduce(lambda x, y: x * y, param.shape, 1)


def _partition(params, world_size, comm_buffer_size_MB, global_rank=1):
    color_group_key = (None, tuple(range(world_size)))
    owners = _StubPartitioner(
        comm_buffer_size_MB, global_rank
    )._partition_2d_parameters_machine_balanced(
        {
            color_group_key: [
                (p.name, _numel(p), p.dtype, ITEMSIZE[p.dtype]) for p in params
            ]
        },
        dict.fromkeys(range(world_size), "host0"),
    )[color_group_key]
    by_name = {p.name: p for p in params}
    return {
        rank: [by_name[name] for name in names]
        for rank, names in owners.items()
    }


def _partition_legacy(params, world_size, comm_buffer_size_MB):
    """The path taken when ``machine_balanced_2d_partition`` is off."""
    return _StubPartitioner(comm_buffer_size_MB)._partition_2d_parameters(
        list(params), world_size
    )


def _active_ranks(volume_numel, world_size, comm_buffer_size_MB):
    """The rank count the greedy fill is allowed to spread over."""
    total_size_mb = volume_numel * 4 / (1024**2)
    buffer_size_mb = comm_buffer_size_MB if comm_buffer_size_MB > 0 else 256
    min_active_ranks = 1
    if total_size_mb > 0:
        min_active_ranks = max(1, int(total_size_mb / buffer_size_mb) + 1)
    return min(min_active_ranks, world_size)


def _partition_whole_group(params, world_size, comm_buffer_size_MB):
    """The pre-change strategy: one bin-packing pass over all dtypes at once."""
    mapping = {rank: [] for rank in range(world_size)}
    parameters = sorted(params, key=_numel, reverse=True)
    sizes = [0] * _active_ranks(
        sum(_numel(p) for p in parameters), world_size, comm_buffer_size_MB
    )
    for param in parameters:
        rank = sizes.index(min(sizes))
        mapping[rank].append(param)
        sizes[rank] += _numel(param)
    return mapping


def _owner_of(mapping):
    return {p.name: rank for rank, plist in mapping.items() for p in plist}


def _ranks_holding(mapping, dtype):
    return {
        rank
        for rank, plist in mapping.items()
        if any(p.dtype == dtype for p in plist)
    }


def _bf16(count, dim=512):
    return [
        _StubParam(f"bf16_{i}", [dim, dim], "bfloat16") for i in range(count)
    ]


def _fp32(count, dim=256):
    return [
        _StubParam(f"fp32_{i}", [dim, dim], "float32") for i in range(count)
    ]


PARAM_SETS = {
    "empty": [],
    "single": _bf16(1),
    "bf16_only": _bf16(10),
    "fp32_only": _fp32(8),
    "mixed": _bf16(10) + _fp32(8),
    "three_dtypes": _bf16(6)
    + _fp32(4)
    + [_StubParam(f"fp16_{i}", [128, 128], "float16") for i in range(3)],
    "ragged": [
        _StubParam(f"bf16_r{i}", [64 * (i + 1), 512], "bfloat16")
        for i in range(7)
    ]
    + [
        _StubParam(f"fp32_r{i}", [32 * (i + 1), 256], "float32")
        for i in range(5)
    ],
}


class TestPartition2DParameters(unittest.TestCase):
    def test_every_param_owned_exactly_once(self):
        for label, params in PARAM_SETS.items():
            for world_size in WORLD_SIZES:
                for buffer_mb in BUFFER_SIZES_MB:
                    with self.subTest(
                        params=label, world_size=world_size, buffer=buffer_mb
                    ):
                        mapping = _partition(params, world_size, buffer_mb)
                        self.assertEqual(
                            set(mapping),
                            set(range(world_size)),
                            "every rank must be present as a key, even if empty",
                        )
                        owners = _owner_of(mapping)
                        self.assertEqual(
                            sorted(owners),
                            sorted(p.name for p in params),
                            "params must be neither dropped nor duplicated",
                        )

    def test_single_dtype_matches_whole_group_packing(self):
        """A single-dtype list must reproduce the pre-change mapping exactly.

        With one machine and one dtype the machine-level terms tie for every
        candidate, so owner choice falls through to the least loaded rank --
        which is what the original greedy did.
        """
        for label in ("empty", "single", "bf16_only", "fp32_only"):
            params = PARAM_SETS[label]
            for world_size in WORLD_SIZES:
                for buffer_mb in BUFFER_SIZES_MB:
                    with self.subTest(
                        params=label, world_size=world_size, buffer=buffer_mb
                    ):
                        self.assertEqual(
                            _owner_of(
                                _partition(params, world_size, buffer_mb)
                            ),
                            _owner_of(
                                _partition_whole_group(
                                    params, world_size, buffer_mb
                                )
                            ),
                        )

    def test_rank_count_follows_own_dtype_volume(self):
        """Each dtype spreads only as wide as its own volume requires."""
        for label, params in PARAM_SETS.items():
            dtypes = {p.dtype for p in params}
            for world_size in WORLD_SIZES:
                for buffer_mb in BUFFER_SIZES_MB:
                    mapping = _partition(params, world_size, buffer_mb)
                    for dtype in dtypes:
                        own = [p for p in params if p.dtype == dtype]
                        expected = min(
                            _active_ranks(
                                sum(_numel(p) for p in own),
                                world_size,
                                buffer_mb,
                            ),
                            len(own),
                        )
                        with self.subTest(
                            params=label,
                            world_size=world_size,
                            buffer=buffer_mb,
                            dtype=dtype,
                        ):
                            self.assertEqual(
                                len(_ranks_holding(mapping, dtype)),
                                expected,
                            )

    def test_minority_dtype_is_not_scattered(self):
        """The regression this partitioning exists for.

        A few fp32 params among many bf16 ones. When the bf16 volume alone needs
        every rank, packing the whole group at once fills all ranks with bf16 and
        then keeps going with the fp32 params, scattering them; each rank they
        land on gets its own small fp32 comm buffer, because AssignGroupBySize
        keys its groups on dtype. Per-dtype packing sizes the fp32 spread from
        the fp32 volume alone.
        """

        def buffer_count(mapping):
            # One FusedCommBuffer per (owner rank, dtype).
            return sum(len({p.dtype for p in pl}) for pl in mapping.values())

        # world_size, bf16 param count, fp32 param count, bucket MB
        configs = ((2, 4, 2, 1), (4, 8, 2, 1), (8, 16, 3, 1))
        for world_size, n_bf16, n_fp32, buffer_mb in configs:
            params = _bf16(n_bf16) + _fp32(n_fp32)
            with self.subTest(world_size=world_size, buffer=buffer_mb):
                new = _partition(params, world_size, buffer_mb)
                old = _partition_whole_group(params, world_size, buffer_mb)
                self.assertLess(
                    len(_ranks_holding(new, "float32")),
                    len(_ranks_holding(old, "float32")),
                    "per-dtype packing should concentrate the minority dtype",
                )
                self.assertLess(buffer_count(new), buffer_count(old))

    def test_dtype_iteration_order_does_not_change_owners(self):
        """Owner assignment must not depend on dtype discovery order.

        Buckets are sorted by (volume, color, dtype, group ranks) before any
        owner is picked, so the order the dtypes appear in the input cannot
        move a param. Feeding the same params with the dtypes grouped in the
        opposite order (relative order within each dtype preserved) must give
        the same owners.
        """
        bf16, fp32 = _bf16(5), _fp32(3)
        for world_size in (2, 4, 8):
            for buffer_mb in (1, 64, 256):
                with self.subTest(world_size=world_size, buffer=buffer_mb):
                    self.assertEqual(
                        _owner_of(
                            _partition(bf16 + fp32, world_size, buffer_mb)
                        ),
                        _owner_of(
                            _partition(fp32 + bf16, world_size, buffer_mb)
                        ),
                    )

    def test_input_list_is_not_reordered(self):
        """The caller's list must survive the call unchanged."""
        params = PARAM_SETS["mixed"]
        before = [p.name for p in params]
        _partition(params, 8, 1)
        self.assertEqual([p.name for p in params], before)


class TestLegacyPartition2DParameters(unittest.TestCase):
    """The machine_balanced_2d_partition=False path, kept as it was.

    It packs each color group from its own rank 0 and knows nothing about
    machines, so only the per-dtype packing it exists for is asserted here.
    """

    def test_every_param_owned_exactly_once(self):
        for label, params in PARAM_SETS.items():
            for world_size in WORLD_SIZES:
                for buffer_mb in BUFFER_SIZES_MB:
                    with self.subTest(
                        params=label, world_size=world_size, buffer=buffer_mb
                    ):
                        mapping = _partition_legacy(
                            params, world_size, buffer_mb
                        )
                        self.assertEqual(
                            set(mapping),
                            set(range(world_size)),
                            "every rank must be present as a key, even if empty",
                        )
                        self.assertEqual(
                            sorted(_owner_of(mapping)),
                            sorted(p.name for p in params),
                            "params must be neither dropped nor duplicated",
                        )

    def test_rank_count_follows_own_dtype_volume(self):
        """Each dtype spreads only as wide as its own volume requires."""
        for label, params in PARAM_SETS.items():
            dtypes = {p.dtype for p in params}
            for world_size in WORLD_SIZES:
                for buffer_mb in BUFFER_SIZES_MB:
                    mapping = _partition_legacy(params, world_size, buffer_mb)
                    for dtype in dtypes:
                        own = [p for p in params if p.dtype == dtype]
                        expected = min(
                            _active_ranks(
                                sum(_numel(p) for p in own),
                                world_size,
                                buffer_mb,
                            ),
                            len(own),
                        )
                        with self.subTest(
                            params=label,
                            world_size=world_size,
                            buffer=buffer_mb,
                            dtype=dtype,
                        ):
                            self.assertEqual(
                                len(_ranks_holding(mapping, dtype)),
                                expected,
                            )


# ---------------------------------------------------------------------------
# PP + EP + sharding: several groups per color, spread over several machines
# ---------------------------------------------------------------------------

MOE_SHARDING, PP, EP, CARDS_PER_MACHINE = 2, 2, 4, 8


def _hybrid_layout():
    """Rank layout of the MoE topology order ['moe_sharding', 'pipe', 'expert'].

    Mirrors the rank formula the optimizer relies on for group_call_opt,
    ``moe_sharding_idx * pp * ep + pp_idx * ep + ep_idx``. Dense params shard
    across everything but the pipe axis, so there is one dense group per PP
    stage; expert params shard along moe_sharding only, so there is one group
    per (stage, expert). Both kinds of group straddle the two machines.
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


def _hybrid_color_group_info(dense_groups, moe_groups):
    """Six dense weights per stage, two expert weights per (stage, expert)."""
    info = {}
    for stage, group_ranks in enumerate(dense_groups):
        info[(None, group_ranks)] = [
            (f"stage{stage}.layer{i}.w", 4096 * 4096, "bfloat16", 2)
            for i in range(6)
        ]
    for idx, group_ranks in enumerate(moe_groups):
        info[("moe_expert", group_ranks)] = [
            (f"expert{idx}.w{i}", 2048 * 4096, "bfloat16", 2) for i in range(2)
        ]
    return info


class TestHybridParallelPartition(unittest.TestCase):
    """The partitioner sees every color group in the job, not just its own."""

    def setUp(self):
        self.dense, self.moe, self.rank_to_machine = _hybrid_layout()
        self.info = _hybrid_color_group_info(self.dense, self.moe)

    def _run(self, comm_buffer_size_MB, global_rank=0):
        return _StubPartitioner(
            comm_buffer_size_MB, global_rank
        )._partition_2d_parameters_machine_balanced(
            self.info, self.rank_to_machine
        )

    def test_one_entry_per_group_not_per_color(self):
        """color_key alone is not an identity once PP or EP is on.

        ``None`` names one dense group per PP stage and ``moe_expert`` one per
        (stage, expert), so keying on the color alone would collapse them and
        lose every group but the last.
        """
        for buffer_mb in (64, 256):
            with self.subTest(buffer=buffer_mb):
                result = self._run(buffer_mb)
                self.assertEqual(set(result), set(self.info))
                self.assertEqual(
                    len([k for k in result if k[0] is None]), len(self.dense)
                )
                self.assertEqual(
                    len([k for k in result if k[0] == "moe_expert"]),
                    len(self.moe),
                )
                for key, params in self.info.items():
                    self.assertEqual(set(result[key]), set(range(len(key[1]))))
                    self.assertEqual(
                        sorted(
                            n for names in result[key].values() for n in names
                        ),
                        sorted(p[0] for p in params),
                    )

    def test_owner_is_a_member_of_its_own_group(self):
        """Every param has exactly one owner, and it is inside its own group."""
        for buffer_mb in (64, 256):
            with self.subTest(buffer=buffer_mb):
                owner_of = {}
                for key, ranks_map in self._run(buffer_mb).items():
                    group_ranks = key[1]
                    for local_rank, names in ranks_map.items():
                        for name in names:
                            self.assertNotIn(name, owner_of)
                            owner_of[name] = group_ranks[local_rank]
                self.assertEqual(
                    set(owner_of),
                    {p[0] for ps in self.info.values() for p in ps},
                )

    def test_result_does_not_depend_on_merge_order(self):
        """Every rank must reach the same answer whatever order it merged in.

        Owners become the reduce dst, so a rank that ordered the gathered
        groups differently and derived different owners would hang the job.
        """
        reversed_info = dict(reversed(list(self.info.items())))
        for buffer_mb in (64, 256):
            with self.subTest(buffer=buffer_mb):
                expected = self._run(buffer_mb)
                actual = _StubPartitioner(
                    buffer_mb, 0
                )._partition_2d_parameters_machine_balanced(
                    reversed_info, self.rank_to_machine
                )
                self.assertEqual(actual, expected)

    def test_load_is_spread_over_every_machine(self):
        """Both machines must carry owners, and carry about the same volume.

        Taking the first ``active_ranks`` local indices of each group instead
        would put every owner of both dense stages and all eight expert groups
        on host0 at 256MB buckets, i.e. a spread of 1.0.
        """
        for buffer_mb in (64, 256):
            with self.subTest(buffer=buffer_mb):
                size_of = {
                    p[0]: p[1] * p[3] for ps in self.info.values() for p in ps
                }
                machine_bytes = dict.fromkeys(
                    set(self.rank_to_machine.values()), 0
                )
                for key, ranks_map in self._run(buffer_mb).items():
                    for local_rank, names in ranks_map.items():
                        machine = self.rank_to_machine[key[1][local_rank]]
                        for name in names:
                            machine_bytes[machine] += size_of[name]

                loads = list(machine_bytes.values())
                self.assertTrue(all(loads), machine_bytes)
                self.assertLessEqual(
                    (max(loads) - min(loads)) / max(loads), 0.05, machine_bytes
                )


if __name__ == "__main__":
    unittest.main()
