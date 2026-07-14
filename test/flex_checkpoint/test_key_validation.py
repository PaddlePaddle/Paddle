# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import MagicMock, patch

import paddle
from paddle.distributed.flex_checkpoint.dcp.key_validation import (
    AOAMappingEntry,
    AOASliceMapping,
    KeyValidationResult,
    ShapeMismatchInfo,
    _append_src_lines,
    _build_aoa_mappings,
    _classify_mappings,
    _describe_ops,
    _emit,
    _format_key_list,
    _format_pattern_groups,
    _format_slice_range,
    _get_signature,
    _group_by_signature,
    _group_keys_adaptive,
    _print_aoa_report,
    _print_standard_report,
    _slice_covers_full,
    _try_fold_src_keys,
    validate_and_report_keys_aoa,
    validate_and_report_keys_standard,
)
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorIndex,
    LocalTensorMetadata,
    Metadata,
)


class TestSliceCoversFull(unittest.TestCase):
    def test_covers_full(self):
        sl = (slice(0, 4), slice(0, 8))
        self.assertTrue(_slice_covers_full(sl, (4, 8)))

    def test_not_covers_partial(self):
        sl = (slice(0, 2), slice(0, 8))
        self.assertFalse(_slice_covers_full(sl, (4, 8)))

    def test_not_covers_mismatched_dims(self):
        sl = (slice(0, 4),)
        self.assertFalse(_slice_covers_full(sl, (4, 8)))

    def test_non_zero_start(self):
        sl = (slice(1, 4), slice(0, 8))
        self.assertFalse(_slice_covers_full(sl, (4, 8)))


class TestFormatSliceRange(unittest.TestCase):
    def test_basic(self):
        src_sl = (slice(0, 4), slice(0, 8))
        dst_sl = (slice(0, 4), slice(0, 8))
        result = _format_slice_range(src_sl, dst_sl)
        self.assertIn("0:4", result)
        self.assertIn("0:8", result)
        self.assertIn("->", result)

    def test_partial_slices(self):
        src_sl = (slice(2, 6),)
        dst_sl = (slice(0, 4),)
        result = _format_slice_range(src_sl, dst_sl)
        self.assertIn("2:6", result)
        self.assertIn("0:4", result)


class TestTryFoldSrcKeys(unittest.TestCase):
    def test_fold_consecutive(self):
        keys = [f"model.experts.{i}.weight" for i in range(8)]
        result = _try_fold_src_keys(keys)
        self.assertIsNotNone(result)
        self.assertIn("{0..7}", result)

    def test_no_fold_different_patterns(self):
        keys = ["model.a.weight", "model.b.weight"]
        result = _try_fold_src_keys(keys)
        self.assertIsNone(result)

    def test_no_fold_multiple_varying_positions(self):
        keys = ["layer.0.expert.0.w", "layer.1.expert.1.w"]
        result = _try_fold_src_keys(keys)
        self.assertIsNone(result)

    def test_single_key(self):
        result = _try_fold_src_keys(["a.0.b"])
        self.assertIsNone(result)

    def test_empty(self):
        result = _try_fold_src_keys([])
        self.assertIsNone(result)


class TestDescribeOps(unittest.TestCase):
    def test_single_with_permute(self):
        entry = AOAMappingEntry(
            dst_key="a.weight",
            dst_global_shape=(4, 8),
            slice_mappings=[
                AOASliceMapping(
                    "b.weight",
                    (slice(0, 4), slice(0, 8)),
                    (slice(0, 4), slice(0, 8)),
                    ["[1, 0]"],
                )
            ],
        )
        result = _describe_ops(entry)
        self.assertIn("permute([1, 0])", result)

    def test_concat_with_cast(self):
        entry = AOAMappingEntry(
            dst_key="a.weight",
            dst_global_shape=(8, 4),
            slice_mappings=[
                AOASliceMapping(
                    "b.weight",
                    (slice(0, 4), slice(0, 4)),
                    (slice(0, 4), slice(0, 4)),
                    ["bfloat16"],
                ),
                AOASliceMapping(
                    "c.weight",
                    (slice(0, 4), slice(0, 4)),
                    (slice(4, 8), slice(0, 4)),
                    ["bfloat16"],
                ),
            ],
        )
        result = _describe_ops(entry)
        self.assertIn("concat", result)
        self.assertIn("cast(bfloat16)", result)

    def test_no_ops(self):
        entry = AOAMappingEntry(
            dst_key="a.weight",
            dst_global_shape=(4, 8),
            slice_mappings=[
                AOASliceMapping(
                    "b.weight",
                    (slice(0, 4), slice(0, 8)),
                    (slice(0, 4), slice(0, 8)),
                    None,
                )
            ],
        )
        result = _describe_ops(entry)
        self.assertEqual(result, "")

    def test_empty_slice_mappings(self):
        entry = AOAMappingEntry(
            dst_key="a.weight", dst_global_shape=(4,), slice_mappings=[]
        )
        result = _describe_ops(entry)
        self.assertEqual(result, "")


class TestClassifyMappings(unittest.TestCase):
    def _make_entry(self, dst_key, src_key, pp=None, multi_src=False):
        if multi_src:
            sms = [
                AOASliceMapping(src_key, (slice(0, 4),), (slice(0, 4),), pp),
                AOASliceMapping(
                    src_key + ".2", (slice(0, 4),), (slice(4, 8),), pp
                ),
            ]
        else:
            sms = [AOASliceMapping(src_key, (slice(0, 4),), (slice(0, 4),), pp)]
        return AOAMappingEntry(
            dst_key=dst_key, dst_global_shape=(8,), slice_mappings=sms
        )

    def test_rename_only(self):
        entry = self._make_entry("model.layers.2.w", "model.layers.0.w")
        rename, transform, struct = _classify_mappings([entry])
        self.assertEqual(len(rename), 1)
        self.assertEqual(len(transform), 0)
        self.assertEqual(len(struct), 0)

    def test_with_transform(self):
        entry = self._make_entry(
            "model.layers.2.w", "model.layers.0.w", ["[1, 0]"]
        )
        rename, transform, struct = _classify_mappings([entry])
        self.assertEqual(len(rename), 0)
        self.assertEqual(len(transform), 1)
        self.assertEqual(len(struct), 0)

    def test_structural_multi_src(self):
        entry = self._make_entry(
            "model.layers.2.qkv", "model.layers.0.q", multi_src=True
        )
        rename, transform, struct = _classify_mappings([entry])
        self.assertEqual(len(rename), 0)
        self.assertEqual(len(transform), 0)
        self.assertEqual(len(struct), 1)

    def test_structural_different_pattern(self):
        entry = self._make_entry("model.decoder.0.w", "model.encoder.0.w")
        rename, transform, struct = _classify_mappings([entry])
        self.assertEqual(len(struct), 1)


class TestGroupBySignature(unittest.TestCase):
    def test_same_signature_grouped(self):
        entries = []
        for i in range(4):
            entries.append(
                AOAMappingEntry(
                    dst_key=f"model.layers.{i}.w",
                    dst_global_shape=(4,),
                    slice_mappings=[
                        AOASliceMapping(
                            f"src.layers.{i}.w",
                            (slice(0, 4),),
                            (slice(0, 4),),
                            ["[1, 0]"],
                        )
                    ],
                )
            )
        groups = _group_by_signature(entries)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(next(iter(groups.values()))), 4)

    def test_different_signatures(self):
        e1 = AOAMappingEntry(
            dst_key="model.layers.0.w",
            dst_global_shape=(4,),
            slice_mappings=[
                AOASliceMapping(
                    "src.layers.0.w", (slice(0, 4),), (slice(0, 4),), None
                )
            ],
        )
        e2 = AOAMappingEntry(
            dst_key="model.layers.0.qkv",
            dst_global_shape=(12,),
            slice_mappings=[
                AOASliceMapping(
                    "src.layers.0.q", (slice(0, 4),), (slice(0, 4),), None
                ),
                AOASliceMapping(
                    "src.layers.0.k", (slice(0, 4),), (slice(4, 8),), None
                ),
            ],
        )
        groups = _group_by_signature([e1, e2])
        self.assertEqual(len(groups), 2)


class TestGetSignature(unittest.TestCase):
    def test_digits_normalized(self):
        entry = AOAMappingEntry(
            dst_key="model.layers.5.weight",
            dst_global_shape=(4,),
            slice_mappings=[
                AOASliceMapping(
                    "src.layers.5.weight", (slice(0, 4),), (slice(0, 4),), None
                )
            ],
        )
        sig = _get_signature(entry)
        self.assertIn("{N}", sig)
        self.assertNotIn("5", sig)


class TestFormatPatternGroups(unittest.TestCase):
    def test_basic_output(self):
        entries = [
            AOAMappingEntry(
                dst_key="model.layers.0.w",
                dst_global_shape=(4, 8),
                slice_mappings=[
                    AOASliceMapping(
                        "src.layers.0.w",
                        (slice(0, 4), slice(0, 8)),
                        (slice(0, 4), slice(0, 8)),
                        ["[1, 0]"],
                    )
                ],
            )
        ]
        groups = {"sig1": entries}
        lines, next_idx = _format_pattern_groups(groups, "test", 1)
        self.assertTrue(any("Pattern #1" in l for l in lines))
        self.assertEqual(next_idx, 2)

    def test_numbering_continues(self):
        e1 = [
            AOAMappingEntry(
                dst_key="a.0.w",
                dst_global_shape=(4,),
                slice_mappings=[
                    AOASliceMapping(
                        "b.0.w", (slice(0, 4),), (slice(0, 4),), None
                    )
                ],
            )
        ]
        e2 = [
            AOAMappingEntry(
                dst_key="c.0.w",
                dst_global_shape=(4,),
                slice_mappings=[
                    AOASliceMapping(
                        "d.0.w", (slice(0, 4),), (slice(0, 4),), None
                    )
                ],
            )
        ]
        groups = {"sig1": e1, "sig2": e2}
        lines, next_idx = _format_pattern_groups(groups, "test", 5)
        self.assertEqual(next_idx, 7)

    def test_max_patterns_truncation(self):
        import paddle.distributed.flex_checkpoint.dcp.key_validation as kv

        old = kv._MAX_PATTERNS_SHOWN
        kv._MAX_PATTERNS_SHOWN = 2
        try:
            groups = {}
            for i in range(5):
                groups[f"sig{i}"] = [
                    AOAMappingEntry(
                        dst_key=f"x.{i}.w",
                        dst_global_shape=(4,),
                        slice_mappings=[
                            AOASliceMapping(
                                f"y.{i}.w", (slice(0, 4),), (slice(0, 4),), None
                            )
                        ],
                    )
                ]
            lines, _ = _format_pattern_groups(groups, "test", 1)
            self.assertTrue(any("more" in l for l in lines))
        finally:
            kv._MAX_PATTERNS_SHOWN = old


class TestAppendSrcLines(unittest.TestCase):
    def test_few_srcs(self):
        sms = [
            AOASliceMapping("a.w", (slice(0, 4),), (slice(0, 4),), None),
            AOASliceMapping("b.w", (slice(0, 4),), (slice(4, 8),), None),
        ]
        lines = []
        _append_src_lines(lines, sms)
        self.assertEqual(len(lines), 2)
        self.assertIn("SRC:", lines[0])
        self.assertIn("+", lines[1])

    def test_many_srcs_foldable(self):
        sms = [
            AOASliceMapping(
                f"experts.{i}.w",
                (slice(0, 4),),
                (slice(i * 4, (i + 1) * 4),),
                None,
            )
            for i in range(10)
        ]
        lines = []
        _append_src_lines(lines, sms)
        # Should fold into single line with ×N
        self.assertTrue(any("\u00d7" in l for l in lines))

    def test_many_srcs_not_foldable(self):
        sms = [
            AOASliceMapping(
                "src_alpha.w", (slice(0, 4),), (slice(0, 4),), None
            ),
            AOASliceMapping("src_beta.w", (slice(0, 4),), (slice(4, 8),), None),
            AOASliceMapping(
                "src_gamma.w", (slice(0, 4),), (slice(8, 12),), None
            ),
            AOASliceMapping(
                "src_delta.w", (slice(0, 4),), (slice(12, 16),), None
            ),
            AOASliceMapping(
                "src_epsilon.w", (slice(0, 4),), (slice(16, 20),), None
            ),
            AOASliceMapping(
                "src_zeta.w", (slice(0, 4),), (slice(20, 24),), None
            ),
        ]
        lines = []
        _append_src_lines(lines, sms)
        # Should show first 2, ..., last 1
        self.assertTrue(any("more" in l for l in lines))


class TestGroupKeysAdaptive(unittest.TestCase):
    def test_basic_grouping(self):
        keys = [
            "model.layers.0.weight",
            "model.layers.1.weight",
            "model.layers.2.weight",
            "model.embed.weight",
        ]
        groups = _group_keys_adaptive(keys)
        self.assertEqual(len(groups), 2)
        # layers.* grouped together
        layer_group = [g for g in groups.values() if len(g) == 3]
        self.assertEqual(len(layer_group), 1)

    def test_no_digits(self):
        keys = ["model.weight", "model.bias"]
        groups = _group_keys_adaptive(keys)
        self.assertEqual(len(groups), 2)


class TestFormatKeyList(unittest.TestCase):
    def test_few_keys(self):
        keys = {"a.w", "b.w", "c.w"}
        lines = _format_key_list(keys)
        self.assertEqual(len(lines), 3)

    def test_many_keys_grouped(self):
        keys = {f"model.layers.{i}.weight" for i in range(100)}
        lines = _format_key_list(keys)
        # Should be grouped and folded
        self.assertTrue(len(lines) < 100)
        self.assertTrue(any("[" in l for l in lines))

    def test_empty(self):
        lines = _format_key_list(set())
        self.assertEqual(lines, [])


class TestEmit(unittest.TestCase):
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation.logger")
    def test_normal_output(self, mock_logger):
        lines = ["line1", "line2", "line3"]
        _emit(lines)
        self.assertEqual(mock_logger.info.call_count, 3)

    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation.logger")
    def test_truncation(self, mock_logger):
        import paddle.distributed.flex_checkpoint.dcp.key_validation as kv

        old_max = kv._MAX_TOTAL_LINES
        kv._MAX_TOTAL_LINES = 5
        try:
            lines = ["x"] * 20
            _emit(lines)
            # 5 lines + 1 truncation msg = 6
            self.assertEqual(mock_logger.info.call_count, 6)
        finally:
            kv._MAX_TOTAL_LINES = old_max


class TestPrintStandardReport(unittest.TestCase):
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_all_matched(self, mock_emit):
        result = KeyValidationResult()
        _print_standard_report(result, "/tmp/ckpt", 100)
        lines = mock_emit.call_args[0][0]
        self.assertTrue(any("[OK]" in l for l in lines))

    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_with_missing_and_unexpected(self, mock_emit):
        result = KeyValidationResult(
            missing_keys={"a.w", "b.w"},
            unexpected_keys={"c.w"},
            shape_mismatches=[ShapeMismatchInfo("d.w", (4, 8), (4, 16))],
        )
        _print_standard_report(result, "/tmp/ckpt", 100)
        lines = mock_emit.call_args[0][0]
        self.assertTrue(any("Missing" in l for l in lines))
        self.assertTrue(any("Unexpected" in l for l in lines))
        self.assertTrue(any("Shape" in l for l in lines))
        self.assertTrue(any("Matched: 98/100" in l for l in lines))

    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_shape_mismatch_truncation(self, mock_emit):
        import paddle.distributed.flex_checkpoint.dcp.key_validation as kv

        old = kv._MAX_SHAPE_MISMATCHES
        kv._MAX_SHAPE_MISMATCHES = 2
        try:
            mismatches = [
                ShapeMismatchInfo(f"k{i}", (4,), (8,)) for i in range(5)
            ]
            result = KeyValidationResult(
                missing_keys={"x"}, shape_mismatches=mismatches
            )
            _print_standard_report(result, "/tmp/ckpt", 10)
            lines = mock_emit.call_args[0][0]
            self.assertTrue(any("and 3 more" in l for l in lines))
        finally:
            kv._MAX_SHAPE_MISMATCHES = old


class TestPrintAoaReport(unittest.TestCase):
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_all_resolved(self, mock_emit):
        mappings = [
            AOAMappingEntry(
                dst_key="a.w",
                dst_global_shape=(4,),
                slice_mappings=[
                    AOASliceMapping("b.w", (slice(0, 4),), (slice(0, 4),), None)
                ],
                is_identity=False,
            ),
        ]
        result = KeyValidationResult()
        _print_aoa_report(result, mappings, set(), "/tmp/ckpt")
        lines = mock_emit.call_args[0][0]
        self.assertTrue(any("[OK]" in l for l in lines))

    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_with_missing(self, mock_emit):
        mappings = [
            AOAMappingEntry(
                "a.w",
                (4,),
                [AOASliceMapping("b.w", (slice(0, 4),), (slice(0, 4),), None)],
            )
        ]
        result = KeyValidationResult(
            missing_keys={"c.w"}, unexpected_keys={"d.w"}
        )
        _print_aoa_report(result, mappings, {"removed.w"}, "/tmp/ckpt")
        lines = mock_emit.call_args[0][0]
        self.assertTrue(any("Missing" in l for l in lines))
        self.assertTrue(any("Unexpected" in l for l in lines))
        self.assertTrue(any("Removed" in l for l in lines))

    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_randomly_initialized_keys(self, mock_emit):
        mappings = []
        result = KeyValidationResult(
            randomly_initialized_keys={"init.w", "init.b"}
        )
        _print_aoa_report(result, mappings, set(), "/tmp/ckpt")
        lines = mock_emit.call_args[0][0]
        self.assertTrue(any("Initialized (2)" in l for l in lines))

    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_removed_keys_truncation(self, mock_emit):
        mappings = []
        removed = {f"removed.key.{i}" for i in range(10)}
        result = KeyValidationResult()
        _print_aoa_report(result, mappings, removed, "/tmp/ckpt")
        lines = mock_emit.call_args[0][0]
        self.assertTrue(any("more" in l for l in lines))


class TestBuildAoaMappings(unittest.TestCase):
    def test_basic(self):
        engine = MagicMock()
        td1 = MagicMock()
        td1.shape = [4, 8]
        td1.slices = [
            (
                "src.w",
                (slice(0, 4), slice(0, 8)),
                (slice(0, 4), slice(0, 8)),
                None,
            )
        ]
        td2 = MagicMock()
        td2.shape = [8, 8]
        td2.slices = [
            (
                "src.q",
                (slice(0, 4), slice(0, 8)),
                (slice(0, 4), slice(0, 8)),
                ["[1, 0]"],
            ),
            (
                "src.k",
                (slice(0, 4), slice(0, 8)),
                (slice(4, 8), slice(0, 8)),
                ["[1, 0]"],
            ),
        ]
        ov = MagicMock()
        ov.items.return_value = sorted({"dst.qkv": td2, "dst.w": td1}.items())
        engine.output_vars = ov

        results = _build_aoa_mappings(engine)
        self.assertEqual(len(results), 2)
        qkv = next(r for r in results if r.dst_key == "dst.qkv")
        self.assertFalse(qkv.is_identity)
        self.assertEqual(len(qkv.slice_mappings), 2)

    def test_identity_detection(self):
        engine = MagicMock()
        td = MagicMock()
        td.shape = [4, 8]
        td.slices = [
            (
                "same.key",
                (slice(0, 4), slice(0, 8)),
                (slice(0, 4), slice(0, 8)),
                None,
            )
        ]
        ov = MagicMock()
        ov.items.return_value = [("same.key", td)]
        engine.output_vars = ov

        results = _build_aoa_mappings(engine)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_identity)

    def test_none_tensor_desc_skipped(self):
        engine = MagicMock()
        ov = MagicMock()
        ov.items.return_value = [("a", None), ("b", None)]
        engine.output_vars = ov
        results = _build_aoa_mappings(engine)
        self.assertEqual(len(results), 0)


class TestValidateAndReportKeysStandard(unittest.TestCase):
    def _make_metadata(self, keys_shapes):
        """keys_shapes: dict of {key: shape_tuple}"""
        storage_metadata = {}
        state_dict_metadata = {}
        for key, shape in keys_shapes.items():
            idx = LocalTensorIndex(
                tensor_key=key,
                global_offset=tuple([0] * len(shape)),
                replica_id=0,
            )
            storage_metadata[idx] = f"{key}.distcp"
            state_dict_metadata[key] = [
                LocalTensorMetadata(
                    global_offset=tuple([0] * len(shape)),
                    local_shape=shape,
                    dtype="float32",
                    global_shape=shape,
                )
            ]
        return Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_metadata=storage_metadata,
        )

    @patch("paddle.distributed.get_rank", return_value=0)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_all_match(self, mock_emit, mock_rank):
        metadata = self._make_metadata({"w1": (4, 8), "w2": (4, 8)})
        state_dict = {
            "w1": paddle.zeros([4, 8]),
            "w2": paddle.zeros([4, 8]),
        }
        result = validate_and_report_keys_standard(
            [metadata], {"w1", "w2"}, None, False, "/tmp/ckpt", state_dict
        )
        self.assertEqual(len(result.missing_keys), 0)
        self.assertEqual(len(result.unexpected_keys), 0)

    @patch("paddle.distributed.get_rank", return_value=0)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_missing_keys(self, mock_emit, mock_rank):
        metadata = self._make_metadata({"w1": (4,)})
        state_dict = {
            "w1": paddle.zeros([4]),
            "w2": paddle.zeros([4]),
        }
        result = validate_and_report_keys_standard(
            [metadata], {"w1", "w2"}, None, False, "/tmp/ckpt", state_dict
        )
        self.assertIn("w2", result.missing_keys)

    @patch("paddle.distributed.get_rank", return_value=0)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_unexpected_keys(self, mock_emit, mock_rank):
        metadata = self._make_metadata({"w1": (4,), "w2": (4,), "w3": (4,)})
        state_dict = {"w1": paddle.zeros([4])}
        result = validate_and_report_keys_standard(
            [metadata], {"w1"}, None, False, "/tmp/ckpt", state_dict
        )
        self.assertIn("w2", result.unexpected_keys)
        self.assertIn("w3", result.unexpected_keys)

    @patch("paddle.distributed.get_rank", return_value=0)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_shape_mismatch(self, mock_emit, mock_rank):
        metadata = self._make_metadata({"w1": (4, 8)})
        state_dict = {"w1": paddle.zeros([4, 16])}
        result = validate_and_report_keys_standard(
            [metadata], {"w1"}, None, False, "/tmp/ckpt", state_dict
        )
        self.assertEqual(len(result.shape_mismatches), 1)
        self.assertEqual(result.shape_mismatches[0].src_global_shape, (4, 8))
        self.assertEqual(result.shape_mismatches[0].dst_global_shape, (4, 16))

    @patch("paddle.distributed.get_rank", return_value=0)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_replica_id_filtered(self, mock_emit, mock_rank):
        """Keys with replica_id != 0 should be filtered out."""
        storage_metadata = {
            LocalTensorIndex(
                tensor_key="w1", global_offset=(0,), replica_id=0
            ): "f1",
            LocalTensorIndex(
                tensor_key="w2", global_offset=(0,), replica_id=1
            ): "f2",
        }
        metadata = Metadata(
            state_dict_metadata={
                "w1": [LocalTensorMetadata((0,), (4,), "float32", (4,))]
            },
            storage_metadata=storage_metadata,
        )
        state_dict = {"w1": paddle.zeros([4])}
        result = validate_and_report_keys_standard(
            [metadata], {"w1"}, None, False, "/tmp/ckpt", state_dict
        )
        self.assertEqual(len(result.unexpected_keys), 0)

    @patch(
        "paddle.distributed.flex_checkpoint.dcp.key_validation._get_rank",
        return_value=1,
    )
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    @patch("paddle.distributed.all_gather_object")
    def test_non_rank0_no_print(self, mock_gather, mock_emit, mock_rank):
        metadata = self._make_metadata({"w1": (4,)})
        state_dict = {"w1": paddle.zeros([4])}

        def gather_side_effect(out_list, obj, group=None):
            out_list.clear()
            out_list.append(obj)

        mock_gather.side_effect = gather_side_effect
        validate_and_report_keys_standard(
            [metadata], {"w1"}, None, True, "/tmp/ckpt", state_dict
        )
        mock_emit.assert_not_called()


class TestValidateAndReportKeysAoa(unittest.TestCase):
    def _make_mock_engine(self):
        engine = MagicMock()
        td1 = MagicMock()
        td1.shape = [4, 8]
        td1.slices = [
            (
                "src.w1",
                (slice(0, 4), slice(0, 8)),
                (slice(0, 4), slice(0, 8)),
                None,
            )
        ]
        td2 = MagicMock()
        td2.shape = [8, 8]
        td2.slices = [
            (
                "src.q",
                (slice(0, 4), slice(0, 8)),
                (slice(0, 4), slice(0, 8)),
                ["[1, 0]"],
            ),
            (
                "src.k",
                (slice(0, 4), slice(0, 8)),
                (slice(4, 8), slice(0, 8)),
                ["[1, 0]"],
            ),
        ]
        # output_vars: need .items() for _build_aoa_mappings and iteration for values()
        ov = MagicMock()
        ov.items.return_value = sorted({"dst.w1": td1, "dst.qkv": td2}.items())
        ov.values.return_value = [td1, td2]
        ov.__iter__ = lambda self: iter({"dst.w1": td1, "dst.qkv": td2})
        ov.__getitem__ = lambda self, k: {"dst.w1": td1, "dst.qkv": td2}[k]
        engine.output_vars = ov
        engine.need_add_output_vars = ["dst.init"]
        engine.need_remove_input_vars = ["src.removed"]
        engine.input_vars = MagicMock()
        engine.input_vars.keys.return_value = [
            "src.w1",
            "src.q",
            "src.k",
            "src.removed",
            "src.leftover",
        ]
        engine.context = MagicMock()
        engine.context.get_all_dst_state_keys.return_value = {
            "dst.w1",
            "dst.qkv",
            "dst.init",
        }
        return engine

    @patch("paddle.distributed.get_rank", return_value=0)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_all_resolved(self, mock_emit, mock_rank):
        engine = self._make_mock_engine()
        metadata = MagicMock()
        result = validate_and_report_keys_aoa(engine, metadata, "/tmp/ckpt")
        # dst.w1 and dst.qkv are covered; dst.init is randomly initialized
        self.assertEqual(len(result.missing_keys), 0)
        # src.leftover not consumed and not removed
        self.assertIn("src.leftover", result.unexpected_keys)
        self.assertIn("dst.init", result.randomly_initialized_keys)

    @patch("paddle.distributed.get_rank", return_value=0)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_truly_missing(self, mock_emit, mock_rank):
        engine = self._make_mock_engine()
        # Add a dst key that is NOT covered
        engine.context.get_all_dst_state_keys = lambda: {
            "dst.w1",
            "dst.qkv",
            "dst.init",
            "dst.missing",
        }
        metadata = MagicMock()
        result = validate_and_report_keys_aoa(engine, metadata, "/tmp/ckpt")
        self.assertIn("dst.missing", result.missing_keys)

    @patch("paddle.distributed.get_rank", return_value=1)
    @patch("paddle.distributed.flex_checkpoint.dcp.key_validation._emit")
    def test_non_rank0_no_print(self, mock_emit, mock_rank):
        engine = self._make_mock_engine()
        metadata = MagicMock()
        validate_and_report_keys_aoa(engine, metadata, "/tmp/ckpt")
        mock_emit.assert_not_called()


class TestColorHelpers(unittest.TestCase):
    def test_no_color(self):
        from paddle.distributed.flex_checkpoint.dcp.key_validation import _C

        self.assertEqual(_C.green("test"), "test")
        self.assertEqual(_C.yellow("test"), "test")
        self.assertEqual(_C.red("test"), "test")
        self.assertEqual(_C.cyan("test"), "test")


if __name__ == "__main__":
    unittest.main()
