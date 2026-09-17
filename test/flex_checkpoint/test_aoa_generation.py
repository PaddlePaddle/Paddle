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
#
# Scope: direct unit tests for the stateless helpers in
# ``paddle.distributed.flex_checkpoint.aoa.generation``. The recursion that
# consumes them lives on ``Layer`` and is pinned by
# ``test_gen_aoa_statements.py``, which only ever drives their happy paths.
# This file covers the name-space algebra (root stripping, placeholder
# templates, subtree re-rooting), the dtype-cast rule lookup and its two
# formatters, and every documented failure mode: those raises are what stops a
# mis-declared model config from silently producing wrong checkpoint keys
# instead of failing at conversion time.

import unittest
from dataclasses import FrozenInstanceError

from paddle.distributed.flex_checkpoint.aoa.generation import (
    AOAContext,
    AOANameScope,
    format_dtype_cast_attr,
    format_inv_dtype_cast_attr,
    join_name,
    resolve_checkpoint_name_from_anchor,
    resolve_dtype_cast_rule,
    resolve_names,
    resolve_single_name,
    should_skip,
    strip_name_suffix,
    validate_checkpoint_name_mapping,
)

_MODEL = "model"

_RULE = {"checkpoint_dtype": "float32", "model_dtype": "bfloat16"}

# The MTP layout: the checkpoint keeps the subtree under its own root, while
# the leaf mapping rules are written against a normal layer ("layers.1").
_MTP_SCOPE = AOANameScope(
    checkpoint_prefix="mtp.0",
    logical_model_prefix="model.layers.1",
    actual_model_prefix="model.mtp.0",
)


def _resolve(
    local_name,
    structured_name_prefix,
    *,
    checkpoint_name_prefix="hf",
    pp_mapping=None,
    name_mapping=None,
    model_name_prefix=_MODEL,
    aoa_name_scope=None,
):
    return resolve_names(
        local_name,
        checkpoint_name_prefix,
        structured_name_prefix,
        pp_mapping or {},
        name_mapping or {},
        model_name_prefix=model_name_prefix,
        aoa_name_scope=aoa_name_scope,
    )


def _ctx():
    return AOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        checkpoint_name_mapping={},
        dtype_cast_rules={},
        pp_to_single_mapping={},
        model_name_prefix=_MODEL,
    )


class TestJoinName(unittest.TestCase):
    def test_both_operands_present(self):
        self.assertEqual(
            join_name("model.layers.0", "weight"), "model.layers.0.weight"
        )

    def test_empty_operand_is_dropped(self):
        # An empty checkpoint prefix (a checkpoint rooted at the top level)
        # must not leave a leading dot behind.
        self.assertEqual(join_name("", "stem.weight"), "stem.weight")
        self.assertEqual(join_name("hf", ""), "hf")
        self.assertEqual(join_name("", ""), "")


class TestStripNameSuffix(unittest.TestCase):
    def test_strips_a_dotted_tail(self):
        self.assertEqual(
            strip_name_suffix("model.layers.0.q_proj.weight", "q_proj.weight"),
            "model.layers.0",
        )

    def test_whole_name_becomes_empty(self):
        self.assertEqual(strip_name_suffix("weight", "weight"), "")

    def test_substring_is_not_a_suffix(self):
        # "roj.weight" ends the name as a string but not on a segment
        # boundary, so it is a miss rather than a silent bad split.
        with self.assertRaisesRegex(ValueError, "does not end with suffix"):
            strip_name_suffix("model.q_proj.weight", "roj.weight")


class TestResolveSingleName(unittest.TestCase):
    _PP = {"pipe.block.weight": "model.layers.0.weight"}

    def test_non_empty_mapping_hits_the_pre_mapping_name(self):
        self.assertEqual(
            resolve_single_name("weight", "pipe.block.", self._PP, _MODEL),
            "model.layers.0.weight",
        )

    def test_non_empty_mapping_miss_raises(self):
        # Deliberately no identity fallback: under PP a miss means the mapping
        # and the live tree disagree, and falling back would mis-key silently.
        with self.assertRaisesRegex(
            KeyError, "missing from pp_to_single_mapping"
        ):
            resolve_single_name("bias", "pipe.block.", self._PP, _MODEL)

    def test_empty_mapping_allows_root_identity(self):
        self.assertEqual(
            resolve_single_name("weight", "model.stem.", {}, _MODEL),
            "model.stem.weight",
        )
        self.assertEqual(resolve_single_name("model", "", {}, _MODEL), "model")

    def test_empty_mapping_rejects_a_foreign_root(self):
        with self.assertRaisesRegex(
            KeyError, "empty pp_to_single_mapping only allows"
        ):
            resolve_single_name("weight", "layers.0.", {}, _MODEL)

    def test_root_must_match_on_a_segment_boundary(self):
        # "modelx" has "model" as a string prefix but is a different root.
        with self.assertRaisesRegex(
            KeyError, "empty pp_to_single_mapping only allows"
        ):
            resolve_single_name("weight", "modelx.", {}, _MODEL)


class TestResolveNames(unittest.TestCase):
    def test_identity_pair_positional_signature(self):
        # Called positionally on purpose: the argument order is part of the
        # contract every component generator is written against.
        self.assertEqual(
            resolve_names(
                "weight", "hf", "model.stem.", {}, {}, model_name_prefix=_MODEL
            ),
            ("hf.stem.weight", "model.stem.weight"),
        )

    def test_empty_checkpoint_prefix_keeps_names_at_the_root(self):
        self.assertEqual(
            _resolve("weight", "model.stem.", checkpoint_name_prefix=""),
            ("stem.weight", "model.stem.weight"),
        )

    def test_tower_root_is_stripped_before_template_matching(self):
        # A multi-tower model roots a tower below the model root; the
        # checkpoint side is relative to that tower root, not to "model".
        self.assertEqual(
            _resolve(
                "weight",
                "model.language_model.stem.",
                model_name_prefix="model.language_model",
            ),
            ("hf.stem.weight", "model.language_model.stem.weight"),
        )

    def test_placeholders_are_captured_and_rendered(self):
        mapping = {
            "layers.$LAYER_ID.experts.$EXPERT_ID.weight": (
                "blocks.$LAYER_ID.e.$EXPERT_ID.w"
            )
        }
        self.assertEqual(
            _resolve(
                "weight", "model.layers.2.experts.5.", name_mapping=mapping
            )[0],
            "hf.blocks.2.e.5.w",
        )

    def test_placeholder_only_matches_a_decimal_segment(self):
        mapping = {"layers.$LAYER_ID.weight": "blocks.$LAYER_ID.w"}
        # A non-numeric segment misses, so the name passes through unchanged
        # rather than rendering "blocks.shared.w" from an uncaptured template.
        self.assertEqual(
            _resolve("weight", "model.layers.shared.", name_mapping=mapping)[0],
            "hf.layers.shared.weight",
        )

    def test_repeated_placeholder_must_capture_one_value(self):
        mapping = {"layers.$LAYER_ID.mtp.$LAYER_ID.weight": "x.$LAYER_ID.w"}
        self.assertEqual(
            _resolve("weight", "model.layers.0.mtp.0.", name_mapping=mapping)[
                0
            ],
            "hf.x.0.w",
        )
        self.assertEqual(
            _resolve("weight", "model.layers.0.mtp.1.", name_mapping=mapping)[
                0
            ],
            "hf.layers.0.mtp.1.weight",
        )

    def test_segment_count_mismatch_is_a_miss(self):
        mapping = {"layers.$LAYER_ID.weight": "blocks.$LAYER_ID.w"}
        self.assertEqual(
            _resolve("weight", "model.layers.0.attn.", name_mapping=mapping)[0],
            "hf.layers.0.attn.weight",
        )

    def test_ambiguous_mapping_raises_at_resolution_time(self):
        mapping = {
            "layers.$LAYER_ID.weight": "a.$LAYER_ID.w",
            "layers.0.weight": "b.w",
        }
        with self.assertRaisesRegex(
            ValueError, "ambiguous checkpoint name mapping"
        ):
            _resolve("weight", "model.layers.0.", name_mapping=mapping)


class TestResolveNamesScoped(unittest.TestCase):
    def test_subtree_is_rerooted_under_the_checkpoint_prefix(self):
        self.assertEqual(
            _resolve("weight", "model.mtp.0.", aoa_name_scope=_MTP_SCOPE),
            ("hf.mtp.0.weight", "model.mtp.0.weight"),
        )

    def test_leaf_mapping_applies_through_the_logical_root(self):
        # The rule is written for a normal layer; scoped resolution routes the
        # subtree name through the logical root so the same rule still hits.
        mapping = {"layers.$LAYER_ID.weight": "layers.$LAYER_ID.linear.weight"}
        self.assertEqual(
            _resolve(
                "weight",
                "model.mtp.0.",
                name_mapping=mapping,
                aoa_name_scope=_MTP_SCOPE,
            )[0],
            "hf.mtp.0.linear.weight",
        )

    def test_absolute_prefix_skips_the_shared_checkpoint_prefix(self):
        # The ForCausalLM layout keeps the head at the checkpoint root, a
        # sibling of the backbone rather than under it.
        head = AOANameScope(
            checkpoint_prefix="lm_head",
            logical_model_prefix="model.output_layer",
            actual_model_prefix="model.output_layer",
            is_checkpoint_prefix_absolute=True,
        )
        self.assertEqual(
            _resolve("weight", "model.output_layer.", aoa_name_scope=head)[0],
            "lm_head.weight",
        )

    def test_relative_prefix_is_the_default(self):
        head = AOANameScope(
            checkpoint_prefix="lm_head",
            logical_model_prefix="model.output_layer",
            actual_model_prefix="model.output_layer",
        )
        self.assertFalse(head.is_checkpoint_prefix_absolute)
        self.assertEqual(
            _resolve("weight", "model.output_layer.", aoa_name_scope=head)[0],
            "hf.lm_head.weight",
        )

    def test_name_outside_the_subtree_raises(self):
        scope = AOANameScope(
            checkpoint_prefix="mtp.0",
            logical_model_prefix="model.layers.1",
            actual_model_prefix="model.mtp.9",
        )
        with self.assertRaisesRegex(ValueError, "is not under prefix"):
            _resolve("weight", "model.mtp.0.", aoa_name_scope=scope)

    def test_mapping_that_rewrites_the_logical_root_raises(self):
        # Scoped resolution strips the logical root back off the mapped name
        # before re-anchoring it, so a value template that drops that root
        # cannot be re-anchored and must not be guessed at.
        mapping = {"layers.$LAYER_ID.weight": "embeddings.weight"}
        with self.assertRaisesRegex(
            ValueError, "rewrites the logical root prefix"
        ):
            _resolve(
                "weight",
                "model.mtp.0.",
                name_mapping=mapping,
                aoa_name_scope=_MTP_SCOPE,
            )


class TestResolveCheckpointNameFromAnchor(unittest.TestCase):
    def _anchor(self, anchor_local="q_proj.weight", **kwargs):
        return resolve_checkpoint_name_from_anchor(
            kwargs.pop("anchor_single", "model.layers.0.q_proj.weight"),
            anchor_local,
            "qkv_proj.weight",
            "hf",
            kwargs.pop("name_mapping", None) or {},
            model_name_prefix=_MODEL,
            **kwargs,
        )

    def test_synthetic_name_lands_in_the_anchor_scope(self):
        # qkv_proj exists only in the checkpoint, so it is never sent through
        # pp_to_single_mapping; its scope comes from a real sibling.
        self.assertEqual(self._anchor(), "hf.layers.0.qkv_proj.weight")

    def test_mapping_applies_to_the_synthetic_name(self):
        mapping = {
            "layers.$LAYER_ID.qkv_proj.weight": "layers.$LAYER_ID.attn.qkv.w"
        }
        self.assertEqual(
            self._anchor(name_mapping=mapping), "hf.layers.0.attn.qkv.w"
        )

    def test_wrong_anchor_local_name_raises(self):
        # The anchor's local name must really end the resolved single name,
        # otherwise the derived scope would be silently wrong.
        with self.assertRaisesRegex(ValueError, "does not end with suffix"):
            self._anchor(anchor_local="k_proj.weight")

    def test_scoped_anchor_is_rerooted_like_a_real_param(self):
        self.assertEqual(
            self._anchor(
                anchor_single="model.mtp.0.q_proj.weight",
                aoa_name_scope=_MTP_SCOPE,
            ),
            "hf.mtp.0.qkv_proj.weight",
        )


class TestResolveDtypeCastRule(unittest.TestCase):
    _TEMPLATE = {"layers.$LAYER_ID.weight": _RULE}

    def test_empty_rules_short_circuit_before_root_stripping(self):
        # The early return keeps the overwhelmingly common no-rule case free of
        # any name work at all -- so a foreign root is not even looked at.
        self.assertIsNone(resolve_dtype_cast_rule("elsewhere", {}, _MODEL))

    def test_template_matches_through_the_model_root(self):
        self.assertEqual(
            resolve_dtype_cast_rule(
                "model.layers.7.weight", self._TEMPLATE, _MODEL
            ),
            _RULE,
        )

    def test_miss_returns_none(self):
        self.assertIsNone(
            resolve_dtype_cast_rule(
                "model.layers.7.bias", self._TEMPLATE, _MODEL
            )
        )

    def test_name_outside_the_model_root_raises(self):
        with self.assertRaisesRegex(ValueError, "is not under prefix"):
            resolve_dtype_cast_rule("other.stem.weight", self._TEMPLATE, _MODEL)

    def test_ambiguous_templates_raise(self):
        rules = {
            "layers.$LAYER_ID.weight": _RULE,
            "layers.0.weight": {
                "checkpoint_dtype": "float32",
                "model_dtype": "float16",
            },
        }
        with self.assertRaisesRegex(
            ValueError, "ambiguous dtype cast rule for 'layers.0.weight'"
        ):
            resolve_dtype_cast_rule("model.layers.0.weight", rules, _MODEL)

    def test_half_declared_rule_raises(self):
        # Refusing at lookup names the offending template; letting it through
        # would surface as a KeyError inside a formatter instead.
        rules = {"stem.weight": {"checkpoint_dtype": "float32"}}
        with self.assertRaisesRegex(
            ValueError, r"is missing \['model_dtype'\]"
        ):
            resolve_dtype_cast_rule("model.stem.weight", rules, _MODEL)


class TestFormatDtypeCastAttr(unittest.TestCase):
    def test_no_rule_is_no_suffix(self):
        self.assertEqual(format_dtype_cast_attr(None), "")
        self.assertEqual(format_inv_dtype_cast_attr(None), "")

    def test_equal_endpoints_are_a_noop(self):
        rule = {"checkpoint_dtype": "bfloat16", "model_dtype": "bfloat16"}
        self.assertEqual(format_dtype_cast_attr(rule), "")
        self.assertEqual(format_inv_dtype_cast_attr(rule), "")

    def test_endpoints_are_mirrored_between_directions(self):
        # One rule drives both directions and the endpoint order is the only
        # difference between the two functions, so pin both literals together:
        # swapping them silently casts the wrong way.
        self.assertEqual(
            format_dtype_cast_attr(_RULE),
            ", src_dtype='float32', dst_dtype='bfloat16'",
        )
        self.assertEqual(
            format_inv_dtype_cast_attr(_RULE),
            ", src_dtype='bfloat16', dst_dtype='float32'",
        )


class TestShouldSkip(unittest.TestCase):
    def test_untransformed_identity_is_skipped(self):
        # AOAEngine fills an unproduced destination from the same-named source,
        # so emitting this line would be redundant.
        self.assertTrue(should_skip("model.weight", "model.weight", ""))

    def test_a_cast_keeps_an_identity_alive(self):
        self.assertFalse(should_skip("model.weight", "model.weight", ", cast"))

    def test_differing_names_are_never_skipped(self):
        self.assertFalse(should_skip("hf.weight", "model.weight", ""))


class TestValidateCheckpointNameMapping(unittest.TestCase):
    def test_empty_mapping_is_valid(self):
        validate_checkpoint_name_mapping({})

    def test_wellformed_mapping_is_valid(self):
        validate_checkpoint_name_mapping(
            {
                "layers.$LAYER_ID.weight": "layers.$LAYER_ID.linear.weight",
                "layers.$LAYER_ID.experts.$EXPERT_ID.w": (
                    "layers.$LAYER_ID.e.$EXPERT_ID.w"
                ),
            }
        )

    def test_empty_key_or_value_raises(self):
        for mapping in ({"": "a.w"}, {"a.w": ""}):
            with self.assertRaisesRegex(ValueError, "empty-string"):
                validate_checkpoint_name_mapping(mapping)

    def test_placeholder_must_be_a_whole_known_segment(self):
        # Either form would survive rendering into a bogus checkpoint name.
        for mapping in (
            {"layers.l$LAYER_ID.weight": "x.w"},
            {"layers.$LAYERID.weight": "x.w"},
        ):
            with self.assertRaisesRegex(ValueError, "has key segment"):
                validate_checkpoint_name_mapping(mapping)

    def test_unknown_placeholder_in_a_value_raises(self):
        with self.assertRaisesRegex(ValueError, "has value segment"):
            validate_checkpoint_name_mapping({"a.w": "x.$FOO.w"})

    def test_value_placeholder_must_be_captured_by_the_key(self):
        with self.assertRaisesRegex(ValueError, "not captured by the key"):
            validate_checkpoint_name_mapping(
                {"layers.0.w": "blocks.$LAYER_ID.w"}
            )

    def test_duplicate_target_raises(self):
        # Two model names landing on one checkpoint key is a tie, which only
        # the whole-model alias handler may express.
        with self.assertRaisesRegex(ValueError, "duplicate checkpoint target"):
            validate_checkpoint_name_mapping({"a.w": "z.w", "b.w": "z.w"})

    def test_concrete_key_shadowed_by_a_template_raises(self):
        # Same config that TestResolveNames pins as a runtime ambiguity; the
        # config-time check catches it first and names both keys.
        with self.assertRaisesRegex(
            ValueError, "overlapping checkpoint name mapping"
        ):
            validate_checkpoint_name_mapping(
                {
                    "layers.$LAYER_ID.weight": "a.$LAYER_ID.w",
                    "layers.0.weight": "b.w",
                }
            )


class TestContainerDefaults(unittest.TestCase):
    def test_excluded_names_defaults_to_empty(self):
        self.assertEqual(_ctx().excluded_names, frozenset())

    def test_context_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            _ctx().checkpoint_name_prefix = "mut"

    def test_name_scope_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            _MTP_SCOPE.checkpoint_prefix = "mut"


if __name__ == "__main__":
    unittest.main()
