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
# templates, subtree segment dropping) and every documented failure mode: those
# raises are what stops a mis-declared model config from silently producing
# wrong checkpoint keys instead of failing at conversion time.

import unittest
from dataclasses import FrozenInstanceError

from paddle.distributed.flex_checkpoint.aoa.generation import (
    AOAContext,
    join_name,
    resolve_checkpoint_name_from_anchor,
    resolve_names,
    resolve_single_name,
    strip_name_suffix,
    validate_checkpoint_name_mapping,
)

_MODEL = "model"

# The MTP layout: the live tree holds the transformer layer one module deeper
# than the checkpoint, which keeps its tensors directly under the layer.
_MTP_DROP = "transformer_layer"


def _resolve(
    local_name,
    structured_name_prefix,
    *,
    checkpoint_name_prefix="hf",
    pp_mapping=None,
    name_mapping=None,
    model_name_prefix=_MODEL,
    checkpoint_lookup_drop_segment=None,
):
    return resolve_names(
        local_name,
        checkpoint_name_prefix,
        structured_name_prefix,
        pp_mapping or {},
        name_mapping or {},
        model_name_prefix=model_name_prefix,
        checkpoint_lookup_drop_segment=checkpoint_lookup_drop_segment,
    )


def _ctx():
    return AOAContext(
        config=None,
        checkpoint_name_prefix="hf",
        checkpoint_name_mapping={},
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
            "model.layers.$LAYER_ID.experts.$EXPERT_ID.weight": (
                "hf.blocks.$LAYER_ID.e.$EXPERT_ID.w"
            )
        }
        # Both sides are absolute, so the rendered value is already the final
        # checkpoint name: the checkpoint prefix is not prepended a second time.
        self.assertEqual(
            _resolve(
                "weight", "model.layers.2.experts.5.", name_mapping=mapping
            )[0],
            "hf.blocks.2.e.5.w",
        )

    def test_value_outside_the_checkpoint_root_is_used_verbatim(self):
        # A mapped value is the final checkpoint name, so the shared checkpoint
        # prefix is not prepended to it. This is what lets the ``ForCausalLM``
        # layout keep its output head a top-level sibling of the backbone
        # instead of forcing it under the backbone root.
        mapping = {"model.lm_head.weight": "lm_head.weight"}
        self.assertEqual(
            _resolve("weight", "model.lm_head.", name_mapping=mapping),
            ("lm_head.weight", "model.lm_head.weight"),
        )

    def test_placeholder_only_matches_a_decimal_segment(self):
        mapping = {"model.layers.$LAYER_ID.weight": "hf.blocks.$LAYER_ID.w"}
        # A non-numeric segment misses, so the name passes through unchanged
        # rather than rendering "blocks.shared.w" from an uncaptured template.
        self.assertEqual(
            _resolve("weight", "model.layers.shared.", name_mapping=mapping)[0],
            "hf.layers.shared.weight",
        )

    def test_repeated_placeholder_must_capture_one_value(self):
        mapping = {
            "model.layers.$LAYER_ID.mtp.$LAYER_ID.weight": "hf.x.$LAYER_ID.w"
        }
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
        mapping = {"model.layers.$LAYER_ID.weight": "hf.blocks.$LAYER_ID.w"}
        self.assertEqual(
            _resolve("weight", "model.layers.0.attn.", name_mapping=mapping)[0],
            "hf.layers.0.attn.weight",
        )

    def test_key_missing_the_model_root_never_matches(self):
        # A key written root-relative cannot match an absolute single name; the
        # miss degrades to the identity fallback. Rejecting it up front is
        # validate_checkpoint_name_mapping's job.
        mapping = {"layers.$LAYER_ID.weight": "hf.blocks.$LAYER_ID.w"}
        self.assertEqual(
            _resolve("weight", "model.layers.0.", name_mapping=mapping)[0],
            "hf.layers.0.weight",
        )

    def test_ambiguous_mapping_raises_at_resolution_time(self):
        mapping = {
            "model.layers.$LAYER_ID.weight": "hf.a.$LAYER_ID.w",
            "model.layers.0.weight": "hf.b.w",
        }
        with self.assertRaisesRegex(
            ValueError, "ambiguous checkpoint name mapping"
        ):
            _resolve("weight", "model.layers.0.", name_mapping=mapping)


class TestResolveNamesDropSegment(unittest.TestCase):
    def test_segment_is_dropped_from_the_checkpoint_side_only(self):
        self.assertEqual(
            _resolve(
                "weight",
                "model.layers.1.transformer_layer.",
                checkpoint_lookup_drop_segment=_MTP_DROP,
            ),
            ("hf.layers.1.weight", "model.layers.1.transformer_layer.weight"),
        )

    def test_layer_mapping_hits_after_the_drop(self):
        # The rule is written for an ordinary layer; dropping the extra live
        # segment is what lets the same rule hit inside the subtree.
        mapping = {
            "model.layers.$LAYER_ID.weight": (
                "hf.layers.$LAYER_ID.linear.weight"
            )
        }
        self.assertEqual(
            _resolve(
                "weight",
                "model.layers.1.transformer_layer.",
                name_mapping=mapping,
                checkpoint_lookup_drop_segment=_MTP_DROP,
            )[0],
            "hf.layers.1.linear.weight",
        )

    def test_absent_segment_is_a_noop(self):
        # A subtree owner passes one value to every child, including the ones
        # that do not carry the segment.
        self.assertEqual(
            _resolve(
                "weight",
                "model.layers.1.",
                checkpoint_lookup_drop_segment=_MTP_DROP,
            ),
            ("hf.layers.1.weight", "model.layers.1.weight"),
        )

    def test_none_leaves_the_name_untouched(self):
        self.assertEqual(
            _resolve(
                "weight",
                "model.layers.1.transformer_layer.",
                checkpoint_lookup_drop_segment=None,
            )[0],
            "hf.layers.1.transformer_layer.weight",
        )

    def test_repeated_segment_raises(self):
        # Which occurrence to drop is unknowable, and guessing would mis-key
        # silently.
        with self.assertRaisesRegex(ValueError, "appears 2 times"):
            _resolve(
                "weight",
                "model.transformer_layer.1.transformer_layer.",
                checkpoint_lookup_drop_segment=_MTP_DROP,
            )

    def test_segment_matches_a_whole_segment_only(self):
        # A substring of a segment is not a segment: dropping on substring
        # match would corrupt unrelated names.
        self.assertEqual(
            _resolve(
                "weight",
                "model.layers.1.transformer_layer_ext.",
                checkpoint_lookup_drop_segment=_MTP_DROP,
            )[0],
            "hf.layers.1.transformer_layer_ext.weight",
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
            "model.layers.$LAYER_ID.qkv_proj.weight": (
                "hf.layers.$LAYER_ID.attn.qkv.w"
            )
        }
        self.assertEqual(
            self._anchor(name_mapping=mapping), "hf.layers.0.attn.qkv.w"
        )

    def test_wrong_anchor_local_name_raises(self):
        # The anchor's local name must really end the resolved single name,
        # otherwise the derived scope would be silently wrong.
        with self.assertRaisesRegex(ValueError, "does not end with suffix"):
            self._anchor(anchor_local="k_proj.weight")

    def test_drop_segment_applies_to_the_synthetic_name(self):
        self.assertEqual(
            self._anchor(
                anchor_single="model.layers.0.transformer_layer.q_proj.weight",
                checkpoint_lookup_drop_segment=_MTP_DROP,
            ),
            "hf.layers.0.qkv_proj.weight",
        )


class TestValidateCheckpointNameMapping(unittest.TestCase):
    def _validate(self, mapping):
        validate_checkpoint_name_mapping(
            mapping,
            model_name_prefix=_MODEL,
        )

    def test_empty_mapping_is_valid(self):
        self._validate({})

    def test_wellformed_mapping_is_valid(self):
        self._validate(
            {
                "model.layers.$LAYER_ID.weight": (
                    "hf.layers.$LAYER_ID.linear.weight"
                ),
                "model.layers.$LAYER_ID.experts.$EXPERT_ID.w": (
                    "hf.layers.$LAYER_ID.e.$EXPERT_ID.w"
                ),
            }
        )

    def test_empty_key_or_value_raises(self):
        for mapping in ({"": "hf.a.w"}, {"model.a.w": ""}):
            with self.assertRaisesRegex(ValueError, "empty-string"):
                self._validate(mapping)

    def test_key_outside_the_model_root_raises(self):
        # A root-relative key can never match an absolute single name, and the
        # miss would silently degrade to the identity fallback instead of
        # failing, so it has to be rejected up front.
        with self.assertRaisesRegex(
            ValueError, "has key template .* outside its root prefix"
        ):
            self._validate({"layers.0.w": "hf.blocks.0.w"})

    def test_key_root_must_match_on_a_segment_boundary(self):
        with self.assertRaisesRegex(ValueError, "outside its root prefix"):
            self._validate({"modelx.layers.0.w": "hf.blocks.0.w"})

    def test_value_outside_any_shared_root_is_accepted(self):
        # A value is the final checkpoint name, used exactly as written, so one
        # that sits outside the shared checkpoint root is honoured rather than
        # mis-keyed -- which is how the ``ForCausalLM`` layout keeps its output
        # head a top-level sibling of the backbone.
        self._validate({"model.lm_head.weight": "lm_head.weight"})

    def test_two_keys_may_share_one_value(self):
        # A shared value is a legal declaration, not a collision: the mapping
        # covers every layout the model can build, while only one of two
        # mutually exclusive output heads is ever instantiated. Rejecting it
        # here would make that declaration unwritable.
        self._validate(
            {
                "model.lm_head.weight": "lm_head.weight",
                "model.shared_head.weight": "lm_head.weight",
            }
        )

    def test_placeholder_must_be_a_whole_known_segment(self):
        # Either form would survive rendering into a bogus checkpoint name.
        for mapping in (
            {"model.layers.l$LAYER_ID.weight": "hf.x.w"},
            {"model.layers.$LAYERID.weight": "hf.x.w"},
        ):
            with self.assertRaisesRegex(ValueError, "has key segment"):
                self._validate(mapping)

    def test_unknown_placeholder_in_a_value_raises(self):
        with self.assertRaisesRegex(ValueError, "has value segment"):
            self._validate({"model.a.w": "hf.x.$FOO.w"})

    def test_value_placeholder_must_be_captured_by_the_key(self):
        with self.assertRaisesRegex(ValueError, "not captured by the key"):
            self._validate({"model.layers.0.w": "hf.blocks.$LAYER_ID.w"})


class TestAOAContext(unittest.TestCase):
    def test_context_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            _ctx().checkpoint_name_prefix = "mut"


if __name__ == "__main__":
    unittest.main()
