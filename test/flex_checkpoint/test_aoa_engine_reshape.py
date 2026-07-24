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

import numpy as np

import paddle
from paddle.distributed.flex_checkpoint.aoa.aoa_engine import (
    AOAEngine,
    ShardedWeightDesc,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.distributed.flex_checkpoint.dcp.utils import assign_sharded_slice

# canonical [E, H, 2, I] <-> flattened [E * H, 2 * I]
E, H, TWO, I = 2, 3, 2, 2
CANONICAL = (E, H, TWO, I)
FLATTENED = (E * H, TWO * I)
CANONICAL_STR = "[2, 3, 2, 2]"


def _build_engine(source_desc, dest_desc, statements):
    return AOAEngine(
        aoa_config={"aoa_statements": statements},
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


class TestAOAEngineReshape(unittest.TestCase):
    def test_forward_full(self):
        # flattened [6, 4] -> canonical [2, 3, 2, 2]
        source = paddle.arange(np.prod(FLATTENED), dtype="float32").reshape(
            list(FLATTENED)
        )
        expected = source.reshape(list(CANONICAL))

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
        engine = _build_engine(
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )

        tgt = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
        )
        out = _fill_target(engine, {"s0": source}, tgt)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_forward_sharded_along_experts(self):
        # destination canonical split along the leading (expert) dim
        source = paddle.arange(np.prod(FLATTENED), dtype="float32").reshape(
            list(FLATTENED)
        )
        expected = source.reshape(list(CANONICAL))

        source_desc = {
            "s0": [
                ShardedWeightDesc("s0", FLATTENED, FLATTENED, (0, 0), "float32")
            ]
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
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )

        for expert in range(E):
            tgt = ShardedWeightDesc(
                "d", shard_shape, CANONICAL, (expert, 0, 0, 0), "float32"
            )
            out = _fill_target(engine, {"s0": source}, tgt)
            np.testing.assert_allclose(
                out.numpy(), expected.numpy()[expert : expert + 1]
            )

    def test_reverse_full(self):
        # canonical [2, 3, 2, 2] -> flattened [6, 4]
        source = paddle.arange(np.prod(CANONICAL), dtype="float32").reshape(
            list(CANONICAL)
        )
        expected = source.reshape(list(FLATTENED))

        source_desc = {
            "s0": [
                ShardedWeightDesc(
                    "s0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        dest_desc = {
            "d": [ShardedWeightDesc("d", FLATTENED, FLATTENED, (0, 0), "float32")]
        }
        engine = _build_engine(
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )

        tgt = ShardedWeightDesc("d", FLATTENED, FLATTENED, (0, 0), "float32")
        out = _fill_target(engine, {"s0": source}, tgt)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_reverse_sharded_block_aligned(self):
        # destination flattened split into block-aligned row chunks
        source = paddle.arange(np.prod(CANONICAL), dtype="float32").reshape(
            list(CANONICAL)
        )
        expected = source.reshape(list(FLATTENED))

        source_desc = {
            "s0": [
                ShardedWeightDesc(
                    "s0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        shard_shape = (H, TWO * I)
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", shard_shape, FLATTENED, (0, 0), "float32"
                ),
                ShardedWeightDesc(
                    "d", shard_shape, FLATTENED, (H, 0), "float32"
                ),
            ]
        }
        engine = _build_engine(
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )

        for block in range(E):
            offset = block * H
            tgt = ShardedWeightDesc(
                "d", shard_shape, FLATTENED, (offset, 0), "float32"
            )
            out = _fill_target(engine, {"s0": source}, tgt)
            np.testing.assert_allclose(
                out.numpy(), expected.numpy()[offset : offset + H]
            )

    def test_identity_when_source_equals_destination(self):
        # source already canonical and destination canonical -> no reshape marker
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
            markers = [
                item
                for item in (mapping.postprocess_list or [])
                if isinstance(item, str)
                and (item.startswith("reshape:") or item.startswith("flatten:"))
            ]
            self.assertEqual(markers, [])


class TestGetDestinationGlobalShape(unittest.TestCase):
    def _engine(self, dest_desc):
        source_desc = {
            "s0": [ShardedWeightDesc("s0", FLATTENED, FLATTENED, (0, 0), "float32")]
        }
        return _build_engine(
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )

    def test_returns_none_without_destination_info(self):
        engine = self._engine(
            {"d": [ShardedWeightDesc("d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32")]}
        )
        engine.destination_state_shard_info = None
        self.assertIsNone(engine.get_destination_global_shape("d"))

    def test_direct_key_hit(self):
        engine = self._engine(
            {"d": [ShardedWeightDesc("d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32")]}
        )
        self.assertEqual(
            engine.get_destination_global_shape("d"), CANONICAL
        )

    def test_resolves_unique_shape_via_optimizer_state_keys(self):
        # "d" is not a direct key, but both optimizer-state keys strip to "d"
        # and share a single global shape.
        dest_desc = {
            "d.w_0": [
                ShardedWeightDesc("d.w_0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32")
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
            {"d": [ShardedWeightDesc("d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32")]}
        )
        engine.destination_state_shard_info = {
            "d.w_0": [
                ShardedWeightDesc("d.w_0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32")
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
        # Two block-aligned flattened sources concat along rows, then reshape to
        # canonical. The reshaped tensor carries two slices (one per expert), so
        # a per-expert query intersects exactly one of them; the other slice is
        # skipped, exercising the non-intersecting branch in find_shard_sources.
        block = (H, TWO * I)  # (3, 4) == one expert block in flattened form
        source_desc = {
            "s0": [ShardedWeightDesc("s0", block, block, (0, 0), "float32")],
            "s1": [ShardedWeightDesc("s1", block, block, (0, 0), "float32")],
        }
        shard_shape = (1, H, TWO, I)
        dest_desc = {
            "d": [
                ShardedWeightDesc("d", shard_shape, CANONICAL, (0, 0, 0, 0), "float32"),
                ShardedWeightDesc("d", shard_shape, CANONICAL, (1, 0, 0, 0), "float32"),
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
            # Only the intersecting slice for this expert is returned; the other
            # reshaped slice is skipped via the non-intersecting branch.
            self.assertEqual(len(mappings), 1)
            self.assertEqual(
                mappings[0].source_slice.key, expected_source_key[expert]
            )
            self.assertEqual(
                tuple(mappings[0].target_slice.global_offset),
                (expert, 0, 0, 0),
            )

class TestAssignShardedSliceReshapeMultiPostprocess(unittest.TestCase):
    def test_reshape_then_transpose_then_cast(self):
        # A reshape marker followed by two more operations exercises the
        # postprocess loop across multiple iterations (transpose then cast).
        source = paddle.arange(
            int(np.prod(FLATTENED)), dtype="float32"
        ).reshape(list(FLATTENED))
        src_desc = ShardedWeightDesc("s", FLATTENED, FLATTENED, (0, 0), "float32")
        src_shard = ShardedWeight("s", source, FLATTENED, FLATTENED, (0, 0))
        out = paddle.zeros(list(CANONICAL), dtype="float16")
        dst_desc = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float16"
        )
        dst_shard = ShardedWeight("d", out, CANONICAL, CANONICAL, (0, 0, 0, 0))

        assign_sharded_slice(
            src_desc,
            src_shard,
            dst_desc,
            dst_shard,
            [f"reshape:{CANONICAL_STR}", "[2, 1, 0, 3]", "float16"],
        )
        expected = paddle.transpose(
            source.reshape(list(CANONICAL)), [2, 1, 0, 3]
        ).astype("float16")
        np.testing.assert_allclose(out.numpy(), expected.numpy())


class TestAOAEngineReshapeErrors(unittest.TestCase):
    def _build(self, source_desc, dest_desc, statements):
        return _build_engine(source_desc, dest_desc, statements)

    def test_canonical_shape_too_few_dims(self):
        source_desc = {
            "s0": [ShardedWeightDesc("s0", (6, 4), (6, 4), (0, 0), "float32")]
        }
        dest_desc = {
            "d": [ShardedWeightDesc("d", (6, 4), (6, 4), (0, 0), "float32")]
        }
        with self.assertRaises(ValueError):
            self._build(source_desc, dest_desc, ["s0 -> d, reshape = '[6, 4]'"])

    def test_unsupported_conversion(self):
        # source (5, 4) is neither the flattened (6, 4) nor canonical form
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
        with self.assertRaises(ValueError):
            self._build(
                source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
            )

    def test_flatten_query_not_block_aligned(self):
        source = paddle.arange(np.prod(CANONICAL), dtype="float32").reshape(
            list(CANONICAL)
        )
        source_desc = {
            "s0": [
                ShardedWeightDesc(
                    "s0", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        dest_desc = {
            "d": [ShardedWeightDesc("d", FLATTENED, FLATTENED, (0, 0), "float32")]
        }
        engine = _build_engine(
            source_desc, dest_desc, [f"s0 -> d, reshape = '{CANONICAL_STR}'"]
        )
        # rows [0:2] are not aligned to block_width H=3
        tgt = ShardedWeightDesc("d", (2, TWO * I), FLATTENED, (0, 0), "float32")
        with self.assertRaises(ValueError):
            _fill_target(engine, {"s0": source}, tgt)


class TestAssignShardedSliceReshapeErrors(unittest.TestCase):
    def test_forward_source_not_block_aligned(self):
        src_desc = ShardedWeightDesc("s", (2, 4), FLATTENED, (0, 0), "float32")
        src_shard = ShardedWeight(
            "s", paddle.zeros([2, 4], dtype="float32"), (2, 4), FLATTENED, (0, 0)
        )
        dst_desc = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
        )
        dst_shard = ShardedWeight(
            "d",
            paddle.zeros(list(CANONICAL), dtype="float32"),
            CANONICAL,
            CANONICAL,
            (0, 0, 0, 0),
        )
        with self.assertRaises(ValueError):
            assign_sharded_slice(
                src_desc,
                src_shard,
                dst_desc,
                dst_shard,
                [f"reshape:{CANONICAL_STR}"],
            )

    def test_flatten_source_trailing_dims_incomplete(self):
        # trailing dims (2, 2, 2) do not match canonical trailing dims (3, 2, 2)
        bad_canonical = (1, 2, 2, 2)
        src_desc = ShardedWeightDesc(
            "s", bad_canonical, CANONICAL, (0, 0, 0, 0), "float32"
        )
        src_shard = ShardedWeight(
            "s",
            paddle.zeros(list(bad_canonical), dtype="float32"),
            bad_canonical,
            CANONICAL,
            (0, 0, 0, 0),
        )
        dst_desc = ShardedWeightDesc(
            "d", (H, TWO * I), FLATTENED, (0, 0), "float32"
        )
        dst_shard = ShardedWeight(
            "d",
            paddle.zeros([H, TWO * I], dtype="float32"),
            (H, TWO * I),
            FLATTENED,
            (0, 0),
        )
        with self.assertRaises(ValueError):
            assign_sharded_slice(
                src_desc,
                src_shard,
                dst_desc,
                dst_shard,
                [f"flatten:{CANONICAL_STR}"],
            )


class TestAOAEngineReshapeMisalignedSlices(unittest.TestCase):
    """Cover the reshape() alignment checks reached via a preceding concat."""

    def test_forward_concat_source_not_block_aligned(self):
        # concat produces flattened row slices [0:2] and [2:6]; the [0:2] slice
        # is not aligned to block_width H=3.
        source_desc = {
            "s0": [ShardedWeightDesc("s0", (2, 4), (2, 4), (0, 0), "float32")],
            "s1": [ShardedWeightDesc("s1", (4, 4), (4, 4), (0, 0), "float32")],
        }
        dest_desc = {
            "d": [
                ShardedWeightDesc(
                    "d", CANONICAL, CANONICAL, (0, 0, 0, 0), "float32"
                )
            ]
        }
        with self.assertRaises(ValueError):
            _build_engine(
                source_desc,
                dest_desc,
                [
                    "s0, s1 -> s, axis = 0",
                    f"s -> d, reshape = '{CANONICAL_STR}'",
                ],
            )

    def test_reverse_concat_trailing_dims_incomplete(self):
        # concat along a trailing dim leaves canonical slices that are not
        # complete on that dim, so the flatten direction must reject them.
        source_desc = {
            "a": [ShardedWeightDesc("a", (2, 3, 1, 2), (2, 3, 1, 2), (0, 0, 0, 0), "float32")],
            "b": [ShardedWeightDesc("b", (2, 3, 1, 2), (2, 3, 1, 2), (0, 0, 0, 0), "float32")],
        }
        dest_desc = {
            "d": [ShardedWeightDesc("d", FLATTENED, FLATTENED, (0, 0), "float32")]
        }
        with self.assertRaises(ValueError):
            _build_engine(
                source_desc,
                dest_desc,
                [
                    "a, b -> s, axis = 2",
                    f"s -> d, reshape = '{CANONICAL_STR}'",
                ],
            )


class TestAssignShardedSliceReshapePostprocess(unittest.TestCase):
    """Cover the transpose/cast postprocess loop that follows a reshape."""

    def _forward_src_dst(self, dtype):
        source = paddle.arange(
            int(np.prod(FLATTENED)), dtype="float32"
        ).reshape(list(FLATTENED))
        src_desc = ShardedWeightDesc("s", FLATTENED, FLATTENED, (0, 0), "float32")
        src_shard = ShardedWeight(
            "s", source, FLATTENED, FLATTENED, (0, 0)
        )
        out = paddle.zeros(list(CANONICAL), dtype=dtype)
        dst_desc = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), dtype
        )
        dst_shard = ShardedWeight(
            "d", out, CANONICAL, CANONICAL, (0, 0, 0, 0)
        )
        return source, src_desc, src_shard, dst_desc, dst_shard, out

    def test_reshape_then_cast(self):
        source, src_desc, src_shard, dst_desc, dst_shard, out = (
            self._forward_src_dst("float16")
        )
        assign_sharded_slice(
            src_desc,
            src_shard,
            dst_desc,
            dst_shard,
            [f"reshape:{CANONICAL_STR}", "float16"],
        )
        expected = source.reshape(list(CANONICAL)).astype("float16")
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_reshape_then_transpose(self):
        # perm [2, 1, 0, 3] keeps the shape (2, 3, 2, 2) unchanged.
        source, src_desc, src_shard, dst_desc, dst_shard, out = (
            self._forward_src_dst("float32")
        )
        assign_sharded_slice(
            src_desc,
            src_shard,
            dst_desc,
            dst_shard,
            [f"reshape:{CANONICAL_STR}", "[2, 1, 0, 3]"],
        )
        expected = paddle.transpose(
            source.reshape(list(CANONICAL)), [2, 1, 0, 3]
        )
        np.testing.assert_allclose(out.numpy(), expected.numpy())


if __name__ == '__main__':
    unittest.main()
