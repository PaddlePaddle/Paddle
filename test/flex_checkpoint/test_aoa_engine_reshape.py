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

import re
import unittest

import numpy as np

import paddle
from paddle.distributed.flex_checkpoint.aoa.aoa_engine import (
    AOAEngine,
    ShardedWeightDesc,
)
from paddle.distributed.flex_checkpoint.aoa.postprocess import make_reshape_op
from paddle.distributed.flex_checkpoint.aoa.reshape import MAX_RESHAPE_PIECES
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.distributed.flex_checkpoint.dcp.utils import assign_sharded_slice

# canonical [E, H, 2, I] <-> flattened [E * H, 2 * I]
E, H, TWO, I = 2, 3, 2, 2
CANONICAL = (E, H, TWO, I)
FLATTENED = (E * H, TWO * I)
CANONICAL_STR = "[2, 3, 2, 2]"

# Scaled-down version of the only production layout that stores the canonical
# form: grouped-GEMM expert weight1 is global [512, 2048, 2, 2048] with EP=8.
# The structure that matters is preserved: leading expert axis, an ``io`` axis,
# the gate/up axis of length 2, and a trailing intermediate axis.
BIZ_E, BIZ_IO, BIZ_I = 4, 4, 4
BIZ_CANONICAL = (BIZ_E, BIZ_IO, TWO, BIZ_I)
BIZ_FLATTENED = (BIZ_E * BIZ_IO, TWO * BIZ_I)
BIZ_CANONICAL_STR = str(list(BIZ_CANONICAL))


def _build_engine(source_desc, dest_desc, statements, aoa_config=None):
    config = {"aoa_statements": statements}
    if aoa_config:
        config.update(aoa_config)
    return AOAEngine(
        aoa_config=config,
        source_state_shard_info=source_desc,
        destination_state_shard_info=dest_desc,
    )


def _fill_target(engine, source_tensors, tgt_desc, dtype="float32"):
    """Reproduce the local reshard loop of load_state_dict for one target shard."""
    out = paddle.zeros(list(tgt_desc.local_shape), dtype=dtype)
    dst_shard = ShardedWeight(
        key=tgt_desc.key,
        local_tensor=out,
        local_shape=tgt_desc.local_shape,
        global_shape=tgt_desc.global_shape,
        global_offset=tgt_desc.global_offset,
    )
    for mapping in engine.find_shard_sources(tgt_desc):
        src_desc = mapping.source_slice
        dst_desc = mapping.target_slice
        full = source_tensors[src_desc.key]
        region_index = tuple(
            slice(offset, offset + size)
            for offset, size in zip(
                src_desc.global_offset, src_desc.local_shape
            )
        )
        region = full[region_index].clone()
        src_shard = ShardedWeight(
            key=src_desc.key,
            local_tensor=region,
            local_shape=src_desc.local_shape,
            global_shape=src_desc.global_shape,
            global_offset=src_desc.global_offset,
        )
        assign_sharded_slice(
            src_desc,
            src_shard,
            dst_desc,
            dst_shard,
            mapping.postprocess_list,
        )
    return out


def _descs(key, global_shape, shards, dtype="float32"):
    """shards = ((local_shape, global_offset), ...)"""
    return [
        ShardedWeightDesc(
            key, tuple(shape), tuple(global_shape), tuple(offset), dtype
        )
        for shape, offset in shards
    ]


def _whole(key, shape, dtype="float32"):
    return _descs(key, shape, ((shape, (0,) * len(shape)),), dtype)


def _flattened(canonical):
    """The 2-D form ``[d0 * d1, prod(rest)]`` of a canonical grouped-GEMM shape."""
    return [canonical[0] * canonical[1], int(np.prod(canonical[2:]))]


def _arange(shape, dtype="float32"):
    return paddle.arange(int(np.prod(shape)), dtype=dtype).reshape(list(shape))


class TestAOAEngineReshape(unittest.TestCase):
    def test_full_and_sharded_reshape(self):
        cases = (
            (
                "forward_full",
                FLATTENED,
                CANONICAL,
                ((CANONICAL, (0, 0, 0, 0)),),
            ),
            (
                "forward_sharded",
                FLATTENED,
                CANONICAL,
                (
                    ((1, H, TWO, I), (0, 0, 0, 0)),
                    ((1, H, TWO, I), (1, 0, 0, 0)),
                ),
            ),
            ("reverse_full", CANONICAL, FLATTENED, ((FLATTENED, (0, 0)),)),
            (
                "reverse_sharded",
                CANONICAL,
                FLATTENED,
                (((H, TWO * I), (0, 0)), ((H, TWO * I), (H, 0))),
            ),
        )
        for name, source_shape, destination_shape, shards in cases:
            with self.subTest(name=name):
                source = paddle.arange(
                    np.prod(source_shape), dtype="float32"
                ).reshape(list(source_shape))
                source_offset = (0,) * len(source_shape)
                source_desc = {
                    "s0": [
                        ShardedWeightDesc(
                            "s0",
                            source_shape,
                            source_shape,
                            source_offset,
                            "float32",
                        )
                    ]
                }
                dest_desc = {
                    "d": [
                        ShardedWeightDesc(
                            "d", shape, destination_shape, offset, "float32"
                        )
                        for shape, offset in shards
                    ]
                }
                engine = _build_engine(
                    source_desc,
                    dest_desc,
                    [f"s0 -> d, reshape = '{list(destination_shape)}'"],
                )
                expected = source.reshape(list(destination_shape))
                for shape, offset in shards:
                    target = ShardedWeightDesc(
                        "d", shape, destination_shape, offset, "float32"
                    )
                    out = _fill_target(engine, {"s0": source}, target)
                    index = tuple(
                        slice(start, start + size)
                        for start, size in zip(offset, shape)
                    )
                    np.testing.assert_allclose(
                        out.numpy(), expected[index].numpy()
                    )

    def test_identity_when_source_equals_destination(self):
        source_desc = {
            "s0": [
                ShardedWeightDesc(
                    "s0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        engine = _build_engine(
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )
        tgt = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
        )
        mappings = engine.find_shard_sources(tgt)
        for mapping in mappings:
            reshape_entries = [
                item
                for item in (mapping.postprocess_list or [])
                if isinstance(item, str) and item.startswith("reshape:")
            ]
            self.assertEqual(reshape_entries, [])


class TestAOAEngineReshapeTrailingDimShards(unittest.TestCase):
    """Destinations that are sharded on dimensions other than the leading one.

    A destination shard whose region is a box of both shapes is copied in one
    piece; one that is a box of the destination only is cut into the pieces that
    are boxes of both. The two geometries have to produce the same values, so the
    cases below cover a sharding of the first trailing dim (one piece), of the
    last one (one piece per index of the axes in front of it), of both at once,
    and the mirror image on the flattened side.
    """

    def _check(self, source_shape, destination_shape, shards):
        source = paddle.arange(np.prod(source_shape), dtype="float32").reshape(
            list(source_shape)
        )
        source_desc = {
            "s0": [
                ShardedWeightDesc(
                    "s0",
                    source_shape,
                    source_shape,
                    (0,) * len(source_shape),
                    "float32",
                )
            ]
        }
        targets = [
            ShardedWeightDesc("d", shape, destination_shape, offset, "float32")
            for shape, offset in shards
        ]
        engine = _build_engine(
            source_desc,
            {"d": targets},
            [f"s0 -> d, reshape = '{list(destination_shape)}'"],
        )
        expected = source.reshape(list(destination_shape))
        for target in targets:
            out = _fill_target(engine, {"s0": source}, target)
            index = tuple(
                slice(start, start + size)
                for start, size in zip(target.global_offset, target.local_shape)
            )
            np.testing.assert_allclose(out.numpy(), expected[index].numpy())

    def test_canonical_destination_sharded_on_a_trailing_dim(self):
        cases = (
            (
                "first_trailing_dim",
                tuple(((E, H, 1, I), (0, 0, k, 0)) for k in range(TWO)),
            ),
            (
                "last_dim",
                tuple(((E, H, TWO, 1), (0, 0, 0, k)) for k in range(I)),
            ),
            (
                "both_trailing_dims",
                tuple(
                    ((E, H, 1, 1), (0, 0, a, b))
                    for a in range(TWO)
                    for b in range(I)
                ),
            ),
            (
                "leading_and_trailing_dims",
                tuple(
                    ((1, H, 1, I), (e, 0, k, 0))
                    for e in range(E)
                    for k in range(TWO)
                ),
            ),
        )
        for name, shards in cases:
            with self.subTest(name=name):
                self._check(FLATTENED, CANONICAL, shards)

    def test_flattened_destination_sharded_on_columns(self):
        cases = (
            (
                "column_blocks",
                tuple(((E * H, I), (0, k * I)) for k in range(TWO)),
            ),
            (
                "single_columns",
                tuple(((E * H, 1), (0, k)) for k in range(TWO * I)),
            ),
            (
                "rows_and_columns",
                tuple(
                    ((H, I), (r * H, k * I))
                    for r in range(E)
                    for k in range(TWO)
                ),
            ),
        )
        for name, shards in cases:
            with self.subTest(name=name):
                self._check(CANONICAL, FLATTENED, shards)

    def test_flattened_destination_sharded_across_block_boundaries(self):
        # Row shards that do not line up with the 3-row blocks the canonical
        # form groups the rows into. Rows [0:2] are part of one block, rows
        # [2:6] are the rest of it plus the next one, so neither shard is a box
        # of the canonical form and both are copied in the pieces that are.
        self._check(
            CANONICAL,
            FLATTENED,
            (((2, TWO * I), (0, 0)), ((E * H - 2, TWO * I), (2, 0))),
        )


class TestGetDestinationGlobalShape(unittest.TestCase):
    def _engine(self, dest_desc):
        source_desc = {
            "s0": [
                ShardedWeightDesc("s0", FLATTENED, FLATTENED, (0, 0), "float32")
            ]
        }
        return _build_engine(
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )

    def test_returns_none_without_destination_info(self):
        engine = self._engine(
            {
                "d": [
                    ShardedWeightDesc(
                        "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                    )
                ]
            }
        )
        engine.destination_state_shard_info = None
        self.assertIsNone(engine.get_destination_global_shape("d"))

    def test_direct_key_hit(self):
        engine = self._engine(
            {
                "d": [
                    ShardedWeightDesc(
                        "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                    )
                ]
            }
        )
        self.assertEqual(engine.get_destination_global_shape("d"), CANONICAL)

    def test_resolves_unique_shape_via_optimizer_state_keys(self):
        dest_desc = {
            "d.w_0": [
                ShardedWeightDesc(
                    "d.w_0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ],
            "d.moment1_0": [
                ShardedWeightDesc(
                    "d.moment1_0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ],
        }
        engine = self._engine(dest_desc)
        self.assertEqual(engine.get_destination_global_shape("d"), CANONICAL)

    def test_ambiguous_shape_raises(self):
        # two keys strip to "d" but carry conflicting global shapes. Inject the
        # ambiguous metadata after a valid build so the failure is isolated to
        # get_destination_global_shape rather than shape propagation at build.
        engine = self._engine(
            {
                "d": [
                    ShardedWeightDesc(
                        "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                    )
                ]
            }
        )
        engine.destination_state_shard_info = {
            "d.w_0": [
                ShardedWeightDesc(
                    "d.w_0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ],
            "d.moment1_0": [
                ShardedWeightDesc(
                    "d.moment1_0", FLATTENED, FLATTENED, (0, 0), "float32"
                )
            ],
        }
        with self.assertRaises(ValueError):
            engine.get_destination_global_shape("d")


class TestAOAEngineReshapeMultiSource(unittest.TestCase):
    def test_forward_query_skips_non_intersecting_slice(self):
        block = (H, TWO * I)
        source_desc = {
            "s0": [ShardedWeightDesc("s0", block, block, (0, 0), "float32")],
            "s1": [ShardedWeightDesc("s1", block, block, (0, 0), "float32")],
        }
        shard_shape = (1, H, TWO, I)
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", shard_shape, CANONICAL, (0, 0, 0, 0), "float32"
                ),
                ShardedWeightDesc(
                    "d", shard_shape, CANONICAL, (1, 0, 0, 0), "float32"
                ),
            ]
        }
        engine = _build_engine(
            source_desc,
            dest_desc,
            ["s0, s1 -> s, axis = 0", f"s -> d, reshape = '{CANONICAL_STR}'"],
        )

        expected_source_key = {0: "s0", 1: "s1"}
        for expert in range(E):
            tgt = ShardedWeightDesc(
                "d", shard_shape, CANONICAL, (expert, 0, 0, 0), "float32"
            )
            mappings = engine.find_shard_sources(tgt)
            self.assertEqual(len(mappings), 1)
            self.assertEqual(
                mappings[0].source_slice.key, expected_source_key[expert]
            )
            self.assertEqual(
                tuple(mappings[0].target_slice.global_offset),
                (expert, 0, 0, 0),
            )

    def test_reverse_concat_reads_each_source(self):
        block_shape = (1, H, TWO, I)
        sources = {
            f"s{expert}": paddle.arange(
                expert * H * TWO * I,
                (expert + 1) * H * TWO * I,
                dtype="float32",
            ).reshape(block_shape)
            for expert in range(E)
        }
        source_desc = {
            key: [
                ShardedWeightDesc(
                    key, block_shape, block_shape, (0, 0, 0, 0), "float32"
                )
            ]
            for key in sources
        }
        shard_shape = (H, TWO * I)
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", shard_shape, FLATTENED, (expert * H, 0), "float32"
                )
                for expert in range(E)
            ]
        }
        engine = _build_engine(
            source_desc,
            dest_desc,
            ["s0, s1 -> s, axis = 0", f"s -> d, reshape = '{list(FLATTENED)}'"],
        )
        for expert in range(E):
            target = dest_desc["d"][expert]
            out = _fill_target(engine, sources, target)
            np.testing.assert_allclose(
                out.numpy(), sources[f"s{expert}"].reshape(shard_shape).numpy()
            )


class TestAOAEngineReshapeColumnWiseSources(unittest.TestCase):
    """Sources that own part of a flattened row rather than whole rows.

    A flattened row ``(e, h)`` is ``[gate(e, h, :) | up(e, h, :)]``, so a
    checkpoint that keeps ``gate_proj`` and ``up_proj`` apart -- or splits the
    ``2 * I`` axis across tensor-parallel ranks -- contributes only part of every
    row. Each contribution is still a canonical box, and the conversion is
    expected to find it.
    """

    def _check(self, column_splits):
        """Assemble the flattened form from column slices, then reshape it."""
        whole = paddle.arange(np.prod(FLATTENED), dtype="float32").reshape(
            list(FLATTENED)
        )
        bounds = [0, *np.cumsum(column_splits).tolist()]
        keys = [f"s{index}" for index in range(len(column_splits))]
        sources = {
            key: whole[:, start:stop].clone()
            for key, start, stop in zip(keys, bounds, bounds[1:])
        }
        source_desc = {
            key: [
                ShardedWeightDesc(
                    key,
                    tuple(tensor.shape),
                    tuple(tensor.shape),
                    (0, 0),
                    "float32",
                )
            ]
            for key, tensor in sources.items()
        }
        target = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
        )
        engine = _build_engine(
            source_desc,
            {"d": [target]},
            [
                f"{', '.join(keys)} -> s, axis = 1",
                f"s -> d, reshape = '{CANONICAL_STR}'",
            ],
        )
        mappings = engine.find_shard_sources(target)
        self.assertEqual(
            {mapping.source_slice.key for mapping in mappings}, set(keys)
        )
        out = _fill_target(engine, sources, target)
        np.testing.assert_allclose(
            out.numpy(), whole.reshape(list(CANONICAL)).numpy()
        )

    def test_gate_and_up_halves(self):
        # Each half is the row-major image of one index of the length-2 axis.
        self._check((I, I))

    def test_one_column_per_source(self):
        # The finest split: every column is its own box.
        self._check((1,) * (TWO * I))

    def test_uneven_column_splits(self):
        # A split that crosses a box boundary: the 3-wide source covers one whole
        # index of the length-2 axis plus half of the next, so it is mapped as
        # two canonical boxes.
        self._check((3, 1))


class TestAOAEngineReshapeTrailingDimSources(unittest.TestCase):
    """Canonical sources that own part of the trailing dims.

    The mirror image of ``TestAOAEngineReshapeColumnWiseSources``: a dispatcher
    that shards the canonical weight along a trailing axis contributes, after
    flattening, a strided set of columns of every row. Such a box is a stack of
    the boxes that do map onto one contiguous column range, and the conversion is
    expected to cut it into them.
    """

    def _sources(self, canonical, axis, splits):
        whole = paddle.arange(int(np.prod(canonical)), dtype="float32").reshape(
            list(canonical)
        )
        bounds = [0, *np.cumsum(splits).tolist()]
        sources, source_desc = {}, {}
        for index, (start, stop) in enumerate(zip(bounds, bounds[1:])):
            key = f"s{index}"
            sources[key] = whole[
                tuple(
                    slice(start, stop) if dim == axis else slice(None)
                    for dim in range(len(canonical))
                )
            ].clone()
            shape = tuple(sources[key].shape)
            source_desc[key] = [
                ShardedWeightDesc(
                    key, shape, shape, (0,) * len(shape), "float32"
                )
            ]
        return whole, sources, source_desc

    def _engine(self, canonical, axis, source_desc, dest_desc):
        return _build_engine(
            source_desc,
            dest_desc,
            [
                f"{', '.join(source_desc)} -> s, axis = {axis}",
                f"s -> d, reshape = '{_flattened(canonical)}'",
            ],
        )

    def _check(self, canonical, axis, splits, pieces_per_source):
        flattened = (canonical[0] * canonical[1], int(np.prod(canonical[2:])))
        whole, sources, source_desc = self._sources(canonical, axis, splits)
        target = ShardedWeightDesc("d", flattened, flattened, (0, 0), "float32")
        engine = self._engine(canonical, axis, source_desc, {"d": [target]})
        mappings = engine.find_shard_sources(target)
        # One slice per contiguous run of columns, so the count is the visible
        # symptom of the box being cut up rather than taken whole.
        self.assertEqual(len(mappings), pieces_per_source * len(splits))
        out = _fill_target(engine, sources, target)
        np.testing.assert_allclose(
            out.numpy(), whole.reshape(list(flattened)).numpy()
        )

    def test_concat_on_the_gate_up_axis(self):
        # Each source is one index of the length-2 axis, which is a whole row of
        # the trailing box: one contiguous column range, one slice.
        self._check(CANONICAL, 2, (1, 1), 1)

    def test_concat_on_the_intermediate_axis(self):
        # Each source is one column of every gate/up half, so its flattened image
        # is {0, I} strided: two runs.
        self._check(CANONICAL, 3, (1, 1), 2)

    def test_expert_parallel_shard_of_the_intermediate_axis(self):
        # The production layout: grouped-GEMM expert weight1 sharded on the
        # trailing intermediate axis, scaled down to [4, 4, 2, 4].
        self._check((4, 4, TWO, 4), 3, (2, 2), 2)

    def test_uneven_shards_of_the_intermediate_axis(self):
        # A split that is not a divisor: the 3-wide source covers columns
        # [0, 3) and [4, 7) of the flattened row.
        self._check((E, H, TWO, 4), 3, (3, 1), 2)

    def test_destination_sharded_on_columns(self):
        # The destination shards the flattened form by columns while the sources
        # shard the canonical one by intermediate index, so neither side's slices
        # line up with the other's and every mapping is a real intersection.
        flattened = (E * H, TWO * I)
        whole, sources, source_desc = self._sources(CANONICAL, 3, (1, 1))
        targets = [
            ShardedWeightDesc("d", (E * H, I), flattened, (0, start), "float32")
            for start in range(0, TWO * I, I)
        ]
        engine = self._engine(CANONICAL, 3, source_desc, {"d": targets})
        expected = whole.reshape(list(flattened)).numpy()
        for target in targets:
            out = _fill_target(engine, sources, target)
            start = target.global_offset[1]
            np.testing.assert_allclose(
                out.numpy(), expected[:, start : start + I]
            )


class TestAOAEngineReshapeErrors(unittest.TestCase):
    def _build(self, source_desc, dest_desc, statements):
        return _build_engine(source_desc, dest_desc, statements)

    def test_stated_shape_must_be_the_destination(self):
        # ``reshape=`` names the shape the statement produces. Naming the
        # source shape or an unrelated one is refused, and the message gives
        # the fix.
        source_desc = {
            "s0": [
                ShardedWeightDesc("s0", FLATTENED, FLATTENED, (0, 0), "float32")
            ]
        }
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        for stated in ("[3, 8]", str(list(FLATTENED))):
            with (
                self.subTest(stated=stated),
                self.assertRaisesRegex(
                    ValueError,
                    "must name the shape the statement produces.*"
                    + re.escape(f"Write reshape='{list(CANONICAL)}'"),
                ),
            ):
                self._build(
                    source_desc, dest_desc, [f"s0 -> d, reshape = '{stated}'"]
                )

    def test_element_counts_must_match(self):
        # source (5, 4) holds 20 elements while the canonical form holds 24, so
        # no reshape relates them at all
        source_desc = {
            "s0": [ShardedWeightDesc("s0", (5, 4), (5, 4), (0, 0), "float32")]
        }
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        with self.assertRaisesRegex(ValueError, "hold 20 and 24 elements"):
            self._build(
                source_desc,
                dest_desc,
                [f"s0 -> d, reshape = '{CANONICAL_STR}'"],
            )


class TestAOAEngineReshapeConcatCrossingBoxBoundaries(unittest.TestCase):
    """Statement-level slices that are no box of the shape on the other side.

    A ``concat`` feeding the reshape fixes which region of the reshape's input
    each source contributes, and nothing makes that region a box of the output
    shape. Such a slice is cut into the pieces that are boxes of both, so the
    cases below assemble the whole tensor out of contributions that each need
    more than one piece.
    """

    def test_forward_concat_source_crosses_a_block_boundary(self):
        # concat produces flattened row slices [0:2] and [2:6]; the canonical
        # form groups the rows into blocks of H = 3, so the second slice covers
        # the tail of one block plus all of the next: two pieces.
        whole = paddle.arange(int(np.prod(FLATTENED)), dtype="float32").reshape(
            list(FLATTENED)
        )
        splits = (2, E * H - 2)
        bounds = [0, *np.cumsum(splits).tolist()]
        keys = [f"s{index}" for index in range(len(splits))]
        sources = {
            key: whole[start:stop].clone()
            for key, start, stop in zip(keys, bounds, bounds[1:])
        }
        source_desc = {
            key: [
                ShardedWeightDesc(
                    key,
                    tuple(tensor.shape),
                    tuple(tensor.shape),
                    (0, 0),
                    "float32",
                )
            ]
            for key, tensor in sources.items()
        }
        target = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
        )
        engine = _build_engine(
            source_desc,
            {"d": [target]},
            [
                f"{', '.join(keys)} -> s, axis = 0",
                f"s -> d, reshape = '{CANONICAL_STR}'",
            ],
        )
        # s0 is one piece, s1 two, and the count is the visible symptom of the
        # slice being cut up rather than taken whole.
        self.assertEqual(len(engine.find_shard_sources(target)), 3)
        out = _fill_target(engine, sources, target)
        np.testing.assert_allclose(
            out.numpy(), whole.reshape(list(CANONICAL)).numpy()
        )

    def test_reverse_concat_owns_part_of_the_block_axis(self):
        # concat along the block axis leaves canonical slices that own only part
        # of it, so each covers a strided set of flattened rows: one piece per
        # index of the axis in front of it.
        whole = paddle.arange(int(np.prod(CANONICAL)), dtype="float32").reshape(
            list(CANONICAL)
        )
        splits = (1, H - 1)
        bounds = [0, *np.cumsum(splits).tolist()]
        keys = [f"s{index}" for index in range(len(splits))]
        sources = {
            key: whole[:, start:stop].clone()
            for key, start, stop in zip(keys, bounds, bounds[1:])
        }
        source_desc = {
            key: [
                ShardedWeightDesc(
                    key,
                    tuple(tensor.shape),
                    tuple(tensor.shape),
                    (0, 0, 0, 0),
                    "float32",
                )
            ]
            for key, tensor in sources.items()
        }
        target = ShardedWeightDesc("d", FLATTENED, FLATTENED, (0, 0), "float32")
        engine = _build_engine(
            source_desc,
            {"d": [target]},
            [
                f"{', '.join(keys)} -> s, axis = 1",
                f"s -> d, reshape = '{list(FLATTENED)}'",
            ],
        )
        self.assertEqual(
            len(engine.find_shard_sources(target)), E * len(splits)
        )
        out = _fill_target(engine, sources, target)
        np.testing.assert_allclose(
            out.numpy(), whole.reshape(list(FLATTENED)).numpy()
        )


class TestOptimizerStatesWithoutTheParameter(unittest.TestCase):
    """Optimizer states are loaded in a load of their own, whose destination
    holds ``<param>.moment1_0`` and the like but not ``<param>``; the scalar
    ``beta*_pow_acc_0`` states there do not have the parameter's shape."""

    def _states(self, name, shape):
        return {
            f"{name}.moment1_0": _whole(f"{name}.moment1_0", shape),
            f"{name}.beta1_pow_acc_0": _whole(f"{name}.beta1_pow_acc_0", (1,)),
        }

    def _load(self, statements, source, destination, reverse=False):
        engine = AOAEngine(
            aoa_config={
                "aoa_statements": statements,
                "aoa_config_reverse": reverse,
            },
            source_state_shard_info=self._states(*source),
            destination_state_shard_info=self._states(*destination),
        )
        name, shape = source
        moment = _arange(shape)
        target = _whole(f"{destination[0]}.moment1_0", destination[1])[0]
        return moment, _fill_target(
            engine, {f"{name}.moment1_0": moment}, target
        )

    def test_reshape(self):
        moment, out = self._load(
            ["s0 -> d, reshape = '[2, 3, 4]'"], ("s0", (6, 4)), ("d", (2, 3, 4))
        )
        np.testing.assert_array_equal(
            out.numpy(), moment.numpy().reshape(2, 3, 4)
        )

    def test_reshape_under_reverse(self):
        moment, out = self._load(
            ["s0 -> d, reshape = '[2, 3, 4]'"],
            ("d", (2, 3, 4)),
            ("s0", (6, 4)),
            reverse=True,
        )
        np.testing.assert_array_equal(out.numpy(), moment.numpy().reshape(6, 4))

    def test_permute_under_reverse(self):
        moment, out = self._load(
            ["s0 -> d, permute = '[1, 0]'"],
            ("d", (6, 4)),
            ("s0", (4, 6)),
            reverse=True,
        )
        np.testing.assert_array_equal(out.numpy(), moment.numpy().T)


class TestAOAEngineReshapeThenPermute(unittest.TestCase):
    """A reshape followed by a permute: the destination lives in the permuted
    layout, so the expert axis is no longer axis 0. Each destination shard has
    to be traced back through the permute, then through the reshape, to the
    source rows it comes from."""

    # flattened [6, 4] -> canonical [2, 3, 4] --permute[1,0,2]--> [3, 2, 4]
    RESHAPE_STR = "[2, 3, 4]"
    PERMUTE = "[1, 0, 2]"

    def _engine(self, dest_desc):
        source_desc = {
            "s0": [ShardedWeightDesc("s0", (6, 4), (6, 4), (0, 0), "float32")]
        }
        return _build_engine(
            source_desc,
            dest_desc,
            [
                f"s0 -> tmp, reshape = '{self.RESHAPE_STR}'",
                f"tmp -> d, permute = '{self.PERMUTE}'",
            ],
        )

    def test_find_sources_maps_permuted_block_axis(self):
        # d has shape [3, 2, 4]; axis 1 is the block/expert axis after permute.
        # Sharding along axis 1 must route each destination block to the matching
        # source block (rows [0:3] vs [3:6]), not always block 0.
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", (3, 1, 4), (3, 2, 4), (0, 0, 0), "float32"
                ),
                ShardedWeightDesc(
                    "d", (3, 1, 4), (3, 2, 4), (0, 1, 0), "float32"
                ),
            ]
        }
        engine = self._engine(dest_desc)
        expected_rows = {0: (0, 3), 1: (3, 6)}
        for block in range(2):
            tgt = ShardedWeightDesc(
                "d", (3, 1, 4), (3, 2, 4), (0, block, 0), "float32"
            )
            mappings = engine.find_shard_sources(tgt)
            self.assertEqual(len(mappings), 1)
            src = mappings[0].source_slice
            self.assertEqual(src.key, "s0")
            self.assertEqual(
                (
                    src.global_offset[0],
                    src.global_offset[0] + src.local_shape[0],
                ),
                expected_rows[block],
            )

    def test_assign_sharded_slice_writes_correct_block(self):
        # End-to-end check through assign_sharded_slice: each permuted block must
        # contain the data of the corresponding flattened source block.
        s0 = paddle.arange(24, dtype="float32").reshape([6, 4])
        expected = paddle.transpose(
            s0.reshape([2, 3, 4]), [1, 0, 2]
        )  # [3, 2, 4]
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", (3, 1, 4), (3, 2, 4), (0, 0, 0), "float32"
                ),
                ShardedWeightDesc(
                    "d", (3, 1, 4), (3, 2, 4), (0, 1, 0), "float32"
                ),
            ]
        }
        engine = self._engine(dest_desc)
        for block in range(2):
            tgt = ShardedWeightDesc(
                "d", (3, 1, 4), (3, 2, 4), (0, block, 0), "float32"
            )
            out = _fill_target(engine, {"s0": s0}, tgt)
            np.testing.assert_allclose(
                out.numpy(), expected.numpy()[:, block : block + 1, :]
            )


class TestAOAEnginePermuteThenReshape(unittest.TestCase):
    """A permute *in front of* the reshape: the mirror image of the class above.

    The source slices recorded by the engine live in the source tensor's own axis
    order, while the reshape reads the permuted order. Each slice ref carries
    the permute in its postprocess list, so cutting it for the reshape traces
    every piece back through the permute, and ``assign_sharded_slice`` replays
    the permute before the reshape. This is the layout Hugging Face stores the
    per-expert matrices in.
    """

    TRANSPOSED = (FLATTENED[1], FLATTENED[0])  # (4, 6), i.e. flattened^T

    def _forward_engine(self, dest_desc, statements=None):
        return _build_engine(
            {
                "s0": [
                    ShardedWeightDesc(
                        "s0",
                        self.TRANSPOSED,
                        self.TRANSPOSED,
                        (0, 0),
                        "float32",
                    )
                ]
            },
            dest_desc,
            statements or [f"s0^T -> d, reshape = '{CANONICAL_STR}'"],
        )

    def _transposed_source(self):
        flat = paddle.arange(int(np.prod(FLATTENED)), dtype="float32").reshape(
            list(FLATTENED)
        )
        return paddle.transpose(flat, [1, 0]).contiguous(), flat.reshape(
            list(CANONICAL)
        )

    def test_macro_and_spelled_out_permute_agree(self):
        # ^T is rewritten into a permute statement, so the two spellings have to
        # produce the same data; anything else would be an artifact of the
        # rewrite rather than of the reshape.
        source, expected = self._transposed_source()
        target = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0,) * 4, "float32"
        )
        for name, statements in (
            ("macro", [f"s0^T -> d, reshape = '{CANONICAL_STR}'"]),
            (
                "permute",
                [
                    "s0 -> t, permute = '[1, 0]'",
                    f"t -> d, reshape = '{CANONICAL_STR}'",
                ],
            ),
        ):
            with self.subTest(name=name):
                engine = self._forward_engine({"d": [target]}, statements)
                out = _fill_target(engine, {"s0": source}, target)
                np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_destination_sharded_on_the_expert_axis(self):
        # find_source_slices has to map the destination's block range onto rows
        # of the *flattened* form, which are columns of the source tensor.
        source, expected = self._transposed_source()
        shard = (1, H, TWO, I)
        targets = [
            ShardedWeightDesc("d", shard, CANONICAL, (e, 0, 0, 0), "float32")
            for e in range(E)
        ]
        engine = self._forward_engine({"d": targets})
        for e, target in enumerate(targets):
            mappings = engine.find_shard_sources(target)
            self.assertEqual(len(mappings), 1)
            src = mappings[0].source_slice
            # columns [e * H, (e + 1) * H) of the transposed source
            self.assertEqual(src.global_offset, (0, e * H))
            self.assertEqual(src.local_shape, (self.TRANSPOSED[0], H))
            out = _fill_target(engine, {"s0": source}, target)
            np.testing.assert_allclose(out.numpy(), expected.numpy()[e : e + 1])

    def test_destination_sharded_on_the_gate_up_axis(self):
        # This destination is a contiguous column range of the flattened form,
        # so the plan narrows the source to it. The narrowing is computed in the
        # reshape's axis order and has to be mapped back through the permute to
        # address the source tensor.
        source, expected = self._transposed_source()
        shard = (E, H, 1, I)
        targets = [
            ShardedWeightDesc("d", shard, CANONICAL, (0, 0, g, 0), "float32")
            for g in range(TWO)
        ]
        engine = self._forward_engine({"d": targets})
        for g, target in enumerate(targets):
            out = _fill_target(engine, {"s0": source}, target)
            np.testing.assert_allclose(
                out.numpy(), expected.numpy()[:, :, g : g + 1]
            )

    def _reverse_engine(self, dest_desc):
        permuted = (H, E, TWO, I)
        return _build_engine(
            {
                "s0": [
                    ShardedWeightDesc(
                        "s0", permuted, permuted, (0,) * 4, "float32"
                    )
                ]
            },
            dest_desc,
            [
                "s0 -> t, permute = '[1, 0, 2, 3]'",
                f"t -> d, reshape = '{list(FLATTENED)}'",
            ],
        )

    def _permuted_source(self):
        canonical = paddle.arange(
            int(np.prod(CANONICAL)), dtype="float32"
        ).reshape(list(CANONICAL))
        return paddle.transpose(
            canonical, [1, 0, 2, 3]
        ).contiguous(), canonical.reshape(list(FLATTENED))

    def test_flatten_a_permuted_canonical_source(self):
        source, expected = self._permuted_source()
        target = ShardedWeightDesc("d", FLATTENED, FLATTENED, (0, 0), "float32")
        engine = self._reverse_engine({"d": [target]})
        out = _fill_target(engine, {"s0": source}, target)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_flatten_into_row_shards_of_a_permuted_source(self):
        # Row shards of the flattened destination select blocks of the canonical
        # form, which after the permute is axis 1 of the source tensor.
        source, expected = self._permuted_source()
        targets = [
            ShardedWeightDesc(
                "d", (H, FLATTENED[1]), FLATTENED, (e * H, 0), "float32"
            )
            for e in range(E)
        ]
        engine = self._reverse_engine({"d": targets})
        for e, target in enumerate(targets):
            mappings = engine.find_shard_sources(target)
            self.assertEqual(len(mappings), 1)
            src = mappings[0].source_slice
            self.assertEqual(src.global_offset, (0, e, 0, 0))
            self.assertEqual(src.local_shape, (H, 1, TWO, I))
            out = _fill_target(engine, {"s0": source}, target)
            np.testing.assert_allclose(
                out.numpy(), expected.numpy()[e * H : (e + 1) * H]
            )


class TestAnyShapesSharingFactorsAreReshaped(unittest.TestCase):
    """Shapes that group the same factors differently.

    The reshape is not limited to ``[d0, d1, *rest]`` <-> ``[d0 * d1,
    prod(rest)]``: any pair of shapes that are contiguous axis groupings of a
    common finer shape -- the shared factors, one digit each on both sides --
    works with whole regions on both sides. Pairs whose factors the two sides
    write differently are in the class after this one.
    """

    def _assert_reshaped(self, source_shape, destination_shape):
        engine = _build_engine(
            {"s0": _whole("s0", source_shape)},
            {"d": _whole("d", destination_shape)},
            [f"s0 -> d, reshape = '{list(destination_shape)}'"],
        )
        source = _arange(source_shape)
        out = _fill_target(
            engine, {"s0": source}, _whole("d", destination_shape)[0]
        )
        np.testing.assert_allclose(
            out.numpy(), source.reshape(list(destination_shape)).numpy()
        )

    def test_merging_only_the_trailing_dims(self):
        # [2, 3, 2, 2] -> [2, 3, 4]: keep the leading dims, merge the trailing
        # ones. The merged side need not have d0 * d1 rows.
        self._assert_reshaped(CANONICAL, (E, H, TWO * I))

    def test_merging_all_but_the_last_dim(self):
        # [2, 3, 2, 2] -> [12, 2]: the split point is wherever the shapes put it.
        self._assert_reshaped(CANONICAL, (E * H * TWO, I))

    def test_merging_everything_after_the_first_dim(self):
        # [2, 3, 2, 2] -> [2, 12]
        self._assert_reshaped(CANONICAL, (E, H * TWO * I))

    def test_inserting_a_size_one_dim(self):
        # [2, 3, 2, 2] -> [2, 3, 1, 2, 2] moves no data at all. A size-one axis
        # owns no digit, which the geometry treats as an empty range.
        self._assert_reshaped(CANONICAL, (E, H, 1, TWO, I))

    def test_neither_shape_groups_the_other(self):
        # [2, 6] and [4, 3] share the factors 2, 2 and 3 and nothing coarser,
        # so the factors the geometry works in are neither of the two sides.
        self._assert_reshaped((2, 6), (4, 3))


class TestShapesThatWriteAFactorDifferently(unittest.TestCase):
    """Row-major order agrees even when no axis boundary is shared.

    ``[6, 4]`` and ``[4, 6]`` share no finer shape, yet hold the same elements
    in the same order: the whole index is one factor the two shapes write in
    different radixes. A shard is cut into the boxes of both that make it up.
    """

    SOURCE, DESTINATION = (6, 4), (4, 6)

    def _assert_shard(self, local, offset):
        engine = _build_engine(
            {"s0": _whole("s0", self.SOURCE)},
            {"d": _descs("d", self.DESTINATION, ((local, offset),))},
            [f"s0 -> d, reshape = '{list(self.DESTINATION)}'"],
        )
        tgt = ShardedWeightDesc("d", local, self.DESTINATION, offset, "float32")
        source = _arange(self.SOURCE)
        out = _fill_target(engine, {"s0": source}, tgt)
        index = tuple(slice(o, o + n) for o, n in zip(offset, local))
        np.testing.assert_allclose(
            out.numpy(),
            source.reshape(list(self.DESTINATION)).numpy()[index],
        )
        return engine.find_shard_sources(tgt)

    def test_whole_tensor_and_row_shards_are_one_piece(self):
        self.assertEqual(len(self._assert_shard((4, 6), (0, 0))), 1)
        self.assertEqual(len(self._assert_shard((2, 6), (2, 0))), 1)

    def test_column_shard_falls_apart_into_its_rows(self):
        # each row of [4, 6] keeps 3 elements: rows 0 and 2 straddle two rows
        # of [6, 4], rows 1 and 3 lie in one
        self.assertEqual(len(self._assert_shard((4, 3), (0, 3))), 6)


class TestUnalignedRowShards(unittest.TestCase):
    """Row shards that do not land on multiples of ``canonical[1]``.

    Expert-parallel shards are aligned to experts, but sharding/DP splits the
    flattened weight by element count, so a shard can start or end inside an
    expert. Such a region is a box of the flattened shape only and is cut into
    the pieces that are boxes of both.
    """

    def test_flatten_destination_row_not_block_aligned(self):
        source = _arange(CANONICAL)
        engine = _build_engine(
            {"s0": _whole("s0", CANONICAL)},
            {"d": _whole("d", FLATTENED)},
            [f"s0 -> d, reshape = '{list(FLATTENED)}'"],
        )
        # rows [0:2] are not a multiple of the former block_width H = 3
        tgt = ShardedWeightDesc("d", (2, TWO * I), FLATTENED, (0, 0), "float32")
        out = _fill_target(engine, {"s0": source}, tgt)
        np.testing.assert_allclose(
            out.numpy(), source.reshape(list(FLATTENED)).numpy()[0:2, :]
        )

    def test_forward_source_row_not_block_aligned(self):
        # A sharding split of the flattened business weight: 16 rows cut into
        # 6 + 10 instead of multiples of the former block_width = 4.
        rows = (6, 10)
        engine = _build_engine(
            {
                name: _whole(name, (count, BIZ_FLATTENED[1]))
                for name, count in zip(("a", "b"), rows)
            },
            {"d": _whole("d", BIZ_CANONICAL)},
            [
                "a, b -> s, axis = 0",
                f"s -> d, reshape = '{BIZ_CANONICAL_STR}'",
            ],
        )
        flattened = _arange(BIZ_FLATTENED)
        sources = {
            "a": flattened[0 : rows[0]].clone(),
            "b": flattened[rows[0] :].clone(),
        }
        out = _fill_target(engine, sources, _whole("d", BIZ_CANONICAL)[0])
        np.testing.assert_allclose(
            out.numpy(), flattened.reshape(list(BIZ_CANONICAL)).numpy()
        )

    def test_both_sides_unaligned_at_once(self):
        # The source is split 6 + 10 rows and the destination shard owns a
        # single index of the last canonical axis, so neither side's region is a
        # box of the other shape.
        rows = (6, 10)
        shard = (BIZ_E, BIZ_IO, TWO, 1)
        engine = _build_engine(
            {
                name: _whole(name, (count, BIZ_FLATTENED[1]))
                for name, count in zip(("a", "b"), rows)
            },
            {"d": _descs("d", BIZ_CANONICAL, ((shard, (0, 0, 0, 1)),))},
            [
                "a, b -> s, axis = 0",
                f"s -> d, reshape = '{BIZ_CANONICAL_STR}'",
            ],
        )
        flattened = _arange(BIZ_FLATTENED)
        sources = {
            "a": flattened[0 : rows[0]].clone(),
            "b": flattened[rows[0] :].clone(),
        }
        tgt = ShardedWeightDesc(
            "d", shard, BIZ_CANONICAL, (0, 0, 0, 1), "float32"
        )
        out = _fill_target(engine, sources, tgt)
        np.testing.assert_allclose(
            out.numpy(),
            flattened.reshape(list(BIZ_CANONICAL)).numpy()[:, :, :, 1:2],
        )


class TestFragmentationIsRejectedNotEnumerated(unittest.TestCase):
    """Cutting an axis the other shape merged costs one piece per element.

    A shard that owns a single index of the last canonical axis is a stack of
    ``prod`` of the axes in front of it, one row each, on the flattened side.
    That is correct but unbounded, so past ``MAX_RESHAPE_PIECES`` pieces the
    engine refuses instead of enumerating them, and names the shapes and the
    shard at fault.
    """

    WIDE = (E, H, MAX_RESHAPE_PIECES * 2, I)
    WIDE_FLATTENED = (E * H, MAX_RESHAPE_PIECES * 2 * I)

    def test_single_index_of_the_last_axis_is_too_fragmented(self):
        shard = (E, H, self.WIDE[2], 1)
        engine = _build_engine(
            {"s0": _whole("s0", self.WIDE_FLATTENED)},
            {"d": _descs("d", self.WIDE, ((shard, (0, 0, 0, 0)),))},
            [f"s0 -> d, reshape = '{list(self.WIDE)}'"],
        )
        tgt = ShardedWeightDesc("d", shard, self.WIDE, (0, 0, 0, 0), "float32")
        with self.assertRaises(ValueError) as ctx:
            engine.find_shard_sources(tgt)
        message = str(ctx.exception)
        self.assertIn("fall into few boxes of both shapes", message)
        self.assertIn(f"more than {MAX_RESHAPE_PIECES} pieces", message)
        self.assertIn(
            f"no fewer boxes common to {self.WIDE} and {self.WIDE_FLATTENED}",
            message,
        )
        self.assertIn("destination_shard=", message)


class TestMaxReshapePiecesFromAOAConfig(unittest.TestCase):
    """``aoa_config["max_reshape_pieces"]`` overrides the default piece limit.

    A shard that owns a single index of the last canonical axis is scattered
    across the flattened side, so the reshape needs several pieces. The limit
    that refuses an over-fragmented reshape is read from the config: lowering
    it below the need refuses a reshape that otherwise fits, and a value above
    the need loads it as usual.
    """

    # flattened (2, 8) <-> canonical (2, 1, 4, 2); the i=0 shard scatters into
    # one column per w in every row -- several pieces, far under the default.
    FLATTENED = (2, 8)
    CANONICAL = (2, 1, 4, 2)
    SHARD = (2, 1, 4, 1)
    STATEMENT = [f"s0 -> d, reshape = '{list(CANONICAL)}'"]

    def _engine(self, max_reshape_pieces=None):
        config = {}
        if max_reshape_pieces is not None:
            config["max_reshape_pieces"] = max_reshape_pieces
        return _build_engine(
            {"s0": _whole("s0", self.FLATTENED)},
            {"d": _descs("d", self.CANONICAL, ((self.SHARD, (0, 0, 0, 0)),))},
            self.STATEMENT,
            aoa_config=config,
        )

    def _target(self):
        return ShardedWeightDesc(
            "d", self.SHARD, self.CANONICAL, (0, 0, 0, 0), "float32"
        )

    def test_a_low_limit_refuses_a_reshape_that_fits_by_default(self):
        self._engine().find_shard_sources(self._target())  # default: fine
        engine = self._engine(max_reshape_pieces=1)
        with self.assertRaises(ValueError) as ctx:
            engine.find_shard_sources(self._target())
        self.assertIn("more than 1 pieces", str(ctx.exception))

    def test_a_limit_above_the_need_loads_correctly(self):
        engine = self._engine(max_reshape_pieces=4096)
        source = _arange(self.FLATTENED)
        out = _fill_target(engine, {"s0": source}, self._target())
        expected = source.numpy().reshape(self.CANONICAL)[:, :, :, 0:1]
        np.testing.assert_array_equal(out.numpy(), expected)


class TestSeveralReshapesInOneChain(unittest.TestCase):
    """A statement chain may hold any number of reshapes.

    Every reshape, permute and cast runs on the data of a slice ref, so a
    piece has to be a box at every stage; the engine cuts each reshape with
    its own mapping and re-cuts what another reshape cuts, until all agree.
    Two reshapes in a row compose into one, so the shape between them cuts
    nothing.
    """

    def _assert_loaded(self, source_shape, statements, expected, shards):
        """Load every shard of ``d`` and compare with ``expected`` (numpy)."""
        destination_shape = expected.shape
        engine = _build_engine(
            {"s0": _whole("s0", source_shape)},
            {"d": _descs("d", destination_shape, shards)},
            statements,
        )
        source = _arange(source_shape)
        mappings = []
        for local, offset in shards:
            tgt = ShardedWeightDesc(
                "d", local, destination_shape, offset, "float32"
            )
            out = _fill_target(engine, {"s0": source}, tgt)
            index = tuple(slice(o, o + n) for o, n in zip(offset, local))
            np.testing.assert_allclose(out.numpy(), expected[index])
            mappings.append(engine.find_shard_sources(tgt))
        return mappings

    def test_reshape_permute_reshape_changing_the_rank(self):
        x = np.arange(16, dtype="float32").reshape(4, 4)
        expected = x.reshape(2, 2, 2, 2).transpose(1, 0, 2, 3).reshape(4, 4)
        statements = [
            "s0 -> t, reshape = '[2, 2, 2, 2]'",
            "t -> u, permute = '[1, 0, 2, 3]'",
            "u -> d, reshape = '[4, 4]'",
        ]
        for shards in (
            (((4, 4), (0, 0)),),
            (((2, 4), (0, 0)), ((2, 4), (2, 0))),
            (((4, 2), (0, 0)), ((4, 2), (0, 2))),
        ):
            with self.subTest(shards=shards):
                self._assert_loaded((4, 4), statements, expected, shards)

    def test_reshape_permute_reshape_keeping_the_rank(self):
        x = np.arange(24, dtype="float32").reshape(4, 6)
        expected = x.reshape(2, 12).T.reshape(6, 4)
        statements = [
            "s0 -> t, reshape = '[2, 12]'",
            "t -> u, permute = '[1, 0]'",
            "u -> d, reshape = '[6, 4]'",
        ]
        for shards in (
            (((6, 4), (0, 0)),),
            (((3, 4), (0, 0)), ((3, 4), (3, 0))),
            (((6, 2), (0, 0)), ((6, 2), (0, 2))),
        ):
            with self.subTest(shards=shards):
                self._assert_loaded((4, 6), statements, expected, shards)

    def test_two_reshapes_in_a_row_compose(self):
        # (4, 6) -> (2, 12) -> (8, 3) is (4, 6) -> (8, 3): every half of the
        # destination is one piece, although (2, 12) alone would cut it.
        x = np.arange(24, dtype="float32").reshape(4, 6)
        mappings = self._assert_loaded(
            (4, 6),
            ["s0 -> t, reshape = '[2, 12]'", "t -> d, reshape = '[8, 3]'"],
            x.reshape(8, 3),
            (((4, 3), (0, 0)), ((4, 3), (4, 0))),
        )
        for shard_mappings in mappings:
            (mapping,) = shard_mappings
            self.assertEqual(
                mapping.postprocess_list, [make_reshape_op((4, 3))]
            )

    def test_reshaping_back_restores_the_slices(self):
        x = np.arange(24, dtype="float32").reshape(4, 6)
        (mappings,) = self._assert_loaded(
            (4, 6),
            ["s0 -> t, reshape = '[2, 3, 4]'", "t -> d, reshape = '[4, 6]'"],
            x,
            (((4, 6), (0, 0)),),
        )
        self.assertEqual([m.postprocess_list for m in mappings], [None])

    def test_reshape_split_reshape(self):
        # The split stands between the reshapes, so they do not compose and
        # the second one is cut through the stages of the first.
        x = np.arange(48, dtype="float32").reshape(6, 8)
        expected = x.reshape(2, 3, 8)[:, :, 4:].reshape(4, 3, 2)
        self._assert_loaded(
            (6, 8),
            [
                "s0 -> t, reshape = '[2, 3, 8]'",
                "t -> a, b, axis = 2",
                "b -> d, reshape = '[4, 3, 2]'",
            ],
            expected,
            (((2, 3, 2), (0, 0, 0)), ((2, 3, 2), (2, 0, 0))),
        )


class TestEveryMappingReadsExactlyItsPiece(unittest.TestCase):
    """A destination shard maps to one slice per piece, and reads nothing else.

    A shard of ``[E, H, 2, I]`` that owns one index of the last axis is, in
    ``[E * H, 2 * I]``, two full-height columns ``I`` apart. Those are two
    pieces, so the shard gets two mappings of one column each: the read is
    exactly the shard's own data rather than the box enclosing both columns.
    """

    def _engine_and_targets(self):
        shard = (E, H, TWO, 1)
        dest = _descs(
            "d", CANONICAL, tuple((shard, (0, 0, 0, k)) for k in range(I))
        )
        engine = _build_engine(
            {"s0": _whole("s0", FLATTENED)},
            {"d": dest},
            [f"s0 -> d, reshape = '{CANONICAL_STR}'"],
        )
        return engine, dest

    def test_one_mapping_per_piece(self):
        engine, targets = self._engine_and_targets()
        for k, tgt in enumerate(targets):
            mappings = engine.find_shard_sources(tgt)
            self.assertEqual(len(mappings), TWO)
            for g, mapping in enumerate(mappings):
                src = mapping.source_slice
                self.assertEqual(tuple(src.local_shape), (E * H, 1))
                self.assertEqual(tuple(src.global_offset), (0, g * I + k))
                self.assertEqual(
                    tuple(mapping.target_slice.local_shape), (E, H, 1, 1)
                )
                self.assertEqual(
                    mapping.postprocess_list,
                    [make_reshape_op((E, H, 1, 1))],
                )
            read = sum(
                int(np.prod(m.source_slice.local_shape)) for m in mappings
            )
            self.assertEqual(read, int(np.prod(tgt.local_shape)))

    def test_results_are_still_correct(self):
        source = _arange(FLATTENED)
        expected = source.reshape(list(CANONICAL))
        engine, targets = self._engine_and_targets()
        for k, tgt in enumerate(targets):
            out = _fill_target(engine, {"s0": source}, tgt)
            np.testing.assert_allclose(
                out.numpy(), expected.numpy()[:, :, :, k : k + 1]
            )


class TestReshapeFollowedByOtherStatements(unittest.TestCase):
    """Statements after a reshape move its slices; the reshape has to follow.

    A ``split`` or ``concat`` behind the reshape re-bases the destination
    slices, and a ``split`` also narrows them. The reshape entry only describes
    the slice it travels with, so narrowing happens relative to that slice and
    stays right wherever the slice ends up.
    """

    def _check(self, sources, destination_shape, statements, expected):
        engine = _build_engine(
            {key: _whole(key, tuple(t.shape)) for key, t in sources.items()},
            {"d": _whole("d", destination_shape)},
            statements,
        )
        out = _fill_target(engine, sources, _whole("d", destination_shape)[0])
        np.testing.assert_allclose(out.numpy(), expected)

    def test_split_on_an_axis_the_reshape_created(self):
        s0 = _arange((6, 4))
        self._check(
            {"s0": s0},
            (1, 3, 4),
            ["s0 -> t, reshape = '[2, 3, 4]'", "t -> a, d, axis = 0"],
            s0.numpy().reshape(2, 3, 4)[1:2],
        )

    def test_split_on_an_axis_the_reshape_kept(self):
        s0 = _arange((6, 4))
        self._check(
            {"s0": s0},
            (2, 3, 2),
            ["s0 -> t, reshape = '[2, 3, 4]'", "t -> a, d, axis = 2"],
            s0.numpy().reshape(2, 3, 4)[:, :, 2:4],
        )

    def test_concat_behind_another_tensor(self):
        # Every axis: the reshape's slice lands at a non-zero offset of ``d``.
        s0 = _arange((6, 4))
        for axis, s1_shape in ((0, (1, 3, 4)), (1, (2, 1, 4)), (2, (2, 3, 1))):
            with self.subTest(axis=axis):
                s1 = _arange(s1_shape) + 100
                expected = np.concatenate(
                    [s1.numpy(), s0.numpy().reshape(2, 3, 4)], axis
                )
                self._check(
                    {"s0": s0, "s1": s1},
                    expected.shape,
                    [
                        "s0 -> t, reshape = '[2, 3, 4]'",
                        f"s1, t -> d, axis = {axis}",
                    ],
                    expected,
                )

    def test_permute_then_split(self):
        s0 = _arange((6, 4))
        self._check(
            {"s0": s0},
            (4, 1, 3),
            [
                "s0 -> t, reshape = '[2, 3, 4]'",
                "t -> u, permute = '[2, 0, 1]'",
                "u -> a, d, axis = 1",
            ],
            s0.numpy().reshape(2, 3, 4).transpose(2, 0, 1)[:, 1:2, :],
        )


class TestReshapeUnderAoaConfigReverse(unittest.TestCase):
    """``aoa_config_reverse`` runs every statement right to left.

    ``reshape=`` still names the right-hand side as written, which is then the
    shape going in; the shape coming out is the left-hand side's. It comes from
    the destination's shard info, or for an intermediate variable from the
    statements run as written, and the two must agree when both are known.
    """

    def _engine(self, statements, destination):
        return AOAEngine(
            aoa_config={
                "aoa_statements": statements,
                "aoa_config_reverse": True,
            },
            source_state_shard_info={"d": _whole("d", (2, 3, 4))},
            destination_state_shard_info=destination,
        )

    def test_the_forward_statement_reads_back(self):
        engine = self._engine(
            ["s0 -> d, reshape = '[2, 3, 4]'"], {"s0": _whole("s0", (6, 4))}
        )
        d = _arange((2, 3, 4))
        out = _fill_target(engine, {"d": d}, _whole("s0", (6, 4))[0])
        np.testing.assert_allclose(out.numpy(), d.numpy().reshape(6, 4))

    def test_stated_shape_must_be_what_is_read(self):
        with self.assertRaisesRegex(ValueError, "names the shape it reads"):
            self._engine(
                ["s0 -> d, reshape = '[6, 4]'"], {"s0": _whole("s0", (6, 4))}
            )

    def _read_back(self, statements, written_input_shape, expected):
        engine = self._engine(
            statements, {"s0": _whole("s0", written_input_shape)}
        )
        d = _arange((2, 3, 4))
        out = _fill_target(
            engine, {"d": d}, _whole("s0", written_input_shape)[0]
        )
        np.testing.assert_allclose(out.numpy(), expected(d.numpy()))

    def test_an_intermediate_takes_its_written_shape(self):
        # Reversed, ``t -> d, reshape`` writes ``t``, which has no shard info;
        # run as written, ``t`` is ``s0`` permuted, (6, 4).
        self._read_back(
            [
                "s0 -> t, permute = '[1, 0]'",
                "t -> d, reshape = '[2, 3, 4]'",
            ],
            (4, 6),
            lambda d: d.reshape(6, 4).T,
        )

    def test_two_reshapes_in_a_row_read_back(self):
        self._read_back(
            [
                "s0 -> t, reshape = '[2, 12]'",
                "t -> d, reshape = '[2, 3, 4]'",
            ],
            (6, 4),
            lambda d: d.reshape(6, 4),
        )

    def test_reshape_permute_reshape_reads_back(self):
        self._read_back(
            [
                "s0 -> t, reshape = '[2, 3, 4]'",
                "t -> u, permute = '[0, 2, 1]'",
                "u -> d, reshape = '[2, 3, 4]'",
            ],
            (6, 4),
            lambda d: d.reshape(2, 4, 3).transpose(0, 2, 1).reshape(6, 4),
        )

    def test_shard_info_that_disagrees_with_the_statements_is_refused(self):
        # ``t`` has shard info (4, 6), but as written it is (6, 4).
        with self.assertRaisesRegex(ValueError, "when run as written"):
            self._engine(
                [
                    "s0 -> t, permute = '[1, 0]'",
                    "t -> d, reshape = '[2, 3, 4]'",
                ],
                {"s0": _whole("s0", (4, 6)), "t": _whole("t", (4, 6))},
            )

    def test_a_written_input_without_shard_info_is_refused(self):
        # Without destination shard info (the full_param path) nothing gives
        # the shape of a written input.
        with self.assertRaisesRegex(ValueError, "has neither"):
            self._engine(["s0 -> d, reshape = '[2, 3, 4]'"], None)


if __name__ == '__main__':
    unittest.main()
