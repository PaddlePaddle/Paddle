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

"""Single-process tests for MuonShardingOptimizer._partition_2d_parameters.

The method maps every 2D (Muon) parameter to an owner rank. It reads nothing
but ``self.comm_buffer_size_MB`` and each parameter's ``shape``/``dtype``, so it
can be exercised directly against stub parameters -- no communication groups, no
accelerators, no launcher. The multi-process behaviour it feeds into is covered
by test/collective/fleet/test_muon_sharding_mixed_dtype_partition.py.
"""

import unittest
from functools import reduce

from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)

WORLD_SIZES = (1, 2, 4, 8, 16, 32)
BUFFER_SIZES_MB = (0, 1, 64, 128, 256, 512)


class _StubParam:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class _StubPartitioner:
    """Carries only the state ``_partition_2d_parameters`` actually reads."""

    _partition_2d_parameters = MuonShardingOptimizer._partition_2d_parameters

    def __init__(self, comm_buffer_size_MB):
        self.comm_buffer_size_MB = comm_buffer_size_MB


def _numel(param):
    return reduce(lambda x, y: x * y, param.shape, 1)


def _partition(params, world_size, comm_buffer_size_MB):
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
        """A single-dtype list must reproduce the pre-change mapping exactly."""
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

        Each dtype is packed from rank 0 with a fresh size vector, so the order
        the dtypes are visited in cannot move a param to another rank. Feeding
        the same params with the dtypes grouped in the opposite order (relative
        order within each dtype preserved) must give the same owners.
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


if __name__ == "__main__":
    unittest.main()
