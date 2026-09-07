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
# Scope: ``Layer.gen_aoa_statements`` / ``Layer.gen_inv_aoa_statements`` -- the
# recursion that walks the live module tree and emits AOA statements. What is
# pinned here is what those two methods decide: which params and buffers take
# part, in what order, how ``ctx`` / the structured-name prefix / an
# ``AOANameScope`` thread down through nesting, when a sublayer's own override
# takes over, and which dtype-cast endpoint order each direction uses. The
# stateless name and dtype helpers they call live in
# ``flex_checkpoint/aoa/generation.py`` and are pinned directly by
# ``test_aoa_generation.py``.

import unittest

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import (
    AOAContext,
    AOANameScope,
)


class _Leaf(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[1])


class _LeafWithBuffer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[1])
        self.register_buffer(
            "running_stat", paddle.zeros([1]), persistable=True
        )
        self.register_buffer("tmp_cache", paddle.zeros([1]), persistable=False)


class _OverridingLeaf(paddle.nn.Layer):
    """Sublayer that fully overrides statement generation (virtual dispatch)."""

    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[1])

    def gen_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        return [f"__custom_fwd__::{structured_name_prefix}"]

    def gen_inv_aoa_statements(
        self, ctx, *, structured_name_prefix="", aoa_name_scope=None
    ):
        return [f"__custom_inv__::{structured_name_prefix}"]


class _NestedModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = paddle.nn.Layer()
        self.model.stem = _Leaf()
        self.model.block = paddle.nn.Layer()
        self.model.block.proj = _Leaf()


class _RecordingMapping(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries = []

    def __getitem__(self, key):
        self.queries.append(key)
        return super().__getitem__(key)


def _context(
    *,
    checkpoint_name_prefix="checkpoint",
    pp_mapping=None,
    name_mapping=None,
    dtype_rules=None,
    model_name_prefix="model",
):
    return AOAContext(
        config=None,
        checkpoint_name_prefix=checkpoint_name_prefix,
        pp_to_single_mapping=pp_mapping or {},
        checkpoint_name_mapping=name_mapping or {},
        dtype_cast_rules=dtype_rules or {},
        model_name_prefix=model_name_prefix,
    )


class TestGenAOAStatements(unittest.TestCase):
    def test_same_name_identity_is_omitted_by_base_layer(self):
        model = _NestedModel()
        ctx = _context(checkpoint_name_prefix="model")

        self.assertEqual(model.gen_aoa_statements(ctx), [])

    def test_nested_recursion_matches_sharded_state_dict_keys(self):
        model = _NestedModel()
        ctx = _context(checkpoint_name_prefix="hf")

        sharded_keys = set(model.sharded_state_dict())
        forward = model.gen_aoa_statements(ctx)
        model_keys = {statement.split(" -> ")[1] for statement in forward}

        self.assertEqual(model_keys, sharded_keys)
        self.assertEqual(
            forward,
            [
                "hf.stem.weight -> model.stem.weight",
                "hf.block.proj.weight -> model.block.proj.weight",
            ],
        )
        self.assertTrue(all(".." not in statement for statement in forward))

    def test_forward_and_inverse_resolve_the_same_name_pair(self):
        model = _NestedModel()
        ctx = _context(checkpoint_name_prefix="archive.root")

        forward = model.gen_aoa_statements(ctx)
        inverse = model.gen_inv_aoa_statements(ctx)

        self.assertEqual(
            inverse,
            [
                "model.stem.weight -> archive.root.stem.weight",
                "model.block.proj.weight -> archive.root.block.proj.weight",
            ],
        )
        self.assertEqual(
            inverse,
            [" -> ".join(reversed(item.split(" -> "))) for item in forward],
        )

    def test_dtype_cast_endpoints_swap_between_directions(self):
        # Both directions look the rule up by the *model*-side name, so one key
        # template drives both and only the endpoint order differs. Getting the
        # order wrong silently casts the wrong way, so pin the literal suffix.
        model = _NestedModel()
        ctx = _context(
            checkpoint_name_prefix="hf",
            dtype_rules={
                "stem.weight": {
                    "checkpoint_dtype": "float32",
                    "model_dtype": "bfloat16",
                }
            },
        )

        self.assertEqual(
            model.gen_aoa_statements(ctx),
            [
                "hf.stem.weight -> model.stem.weight"
                ", src_dtype='float32', dst_dtype='bfloat16'",
                # Unmatched param: the rule must not leak onto it.
                "hf.block.proj.weight -> model.block.proj.weight",
            ],
        )
        self.assertEqual(
            model.gen_inv_aoa_statements(ctx),
            [
                "model.stem.weight -> hf.stem.weight"
                ", src_dtype='bfloat16', dst_dtype='float32'",
                "model.block.proj.weight -> hf.block.proj.weight",
            ],
        )

    def test_dtype_cast_defeats_the_same_name_skip(self):
        # A cast is a real transform, so it must keep an otherwise identity
        # statement alive -- the ``cast`` half of ``should_skip``, reached
        # through the recursion rather than called directly. ``block.proj``
        # carries no rule and is still omitted.
        model = _NestedModel()
        ctx = _context(
            checkpoint_name_prefix="model",
            dtype_rules={
                "stem.weight": {
                    "checkpoint_dtype": "float32",
                    "model_dtype": "bfloat16",
                }
            },
        )

        self.assertEqual(
            model.gen_aoa_statements(ctx),
            [
                "model.stem.weight -> model.stem.weight"
                ", src_dtype='float32', dst_dtype='bfloat16'"
            ],
        )
        self.assertEqual(
            model.gen_inv_aoa_statements(ctx),
            [
                "model.stem.weight -> model.stem.weight"
                ", src_dtype='bfloat16', dst_dtype='float32'"
            ],
        )

    def test_equal_dtype_endpoints_emit_no_cast(self):
        # A rule whose endpoints agree is a no-op, not a transform: the suffix
        # is empty in both directions, so the identity skip applies again.
        model = _NestedModel()
        ctx = _context(
            checkpoint_name_prefix="model",
            dtype_rules={
                "stem.weight": {
                    "checkpoint_dtype": "bfloat16",
                    "model_dtype": "bfloat16",
                }
            },
        )

        self.assertEqual(model.gen_aoa_statements(ctx), [])
        self.assertEqual(model.gen_inv_aoa_statements(ctx), [])

    def test_ambiguous_dtype_cast_rule_raises(self):
        # A literal template and a placeholder template can both match the same
        # name. Picking either silently would make the emitted cast depend on
        # dict order, so the lookup refuses instead.
        model = paddle.nn.Layer()
        model.layers = paddle.nn.LayerList([_Leaf()])
        ctx = _context(
            checkpoint_name_prefix="hf",
            dtype_rules={
                "layers.$LAYER_ID.weight": {
                    "checkpoint_dtype": "float32",
                    "model_dtype": "bfloat16",
                },
                "layers.0.weight": {
                    "checkpoint_dtype": "float32",
                    "model_dtype": "float16",
                },
            },
        )

        for gen in (model.gen_aoa_statements, model.gen_inv_aoa_statements):
            with self.assertRaisesRegex(
                ValueError, "ambiguous dtype cast rule for 'layers.0.weight'"
            ):
                gen(ctx, structured_name_prefix="model.")

    def test_dtype_cast_rule_missing_an_endpoint_raises(self):
        # A half-declared rule cannot be formatted in either direction; failing
        # at lookup names the offending template instead of raising a KeyError
        # deep inside the formatter.
        model = _NestedModel()
        ctx = _context(
            checkpoint_name_prefix="hf",
            dtype_rules={"stem.weight": {"checkpoint_dtype": "float32"}},
        )

        for gen in (model.gen_aoa_statements, model.gen_inv_aoa_statements):
            with self.assertRaisesRegex(
                ValueError,
                r"rule 'stem.weight' \(matched by 'stem.weight'\) is missing "
                r"\['model_dtype'\]",
            ):
                gen(ctx)

    def test_pp_mapping_uses_pre_mapping_structured_name(self):
        model = paddle.nn.Layer()
        model.block = _Leaf()
        pp_mapping = _RecordingMapping(
            {"pipe.block.weight": "model.layers.0.weight"}
        )
        ctx = _context(checkpoint_name_prefix="hf", pp_mapping=pp_mapping)

        statements = model.gen_aoa_statements(
            ctx, structured_name_prefix="pipe."
        )

        self.assertEqual(pp_mapping.queries, ["pipe.block.weight"])
        self.assertEqual(
            statements, ["hf.layers.0.weight -> model.layers.0.weight"]
        )

    def test_persistable_buffer_is_emitted_but_transient_is_skipped(self):
        model = paddle.nn.Layer()
        model.leaf = _LeafWithBuffer()
        ctx = _context(checkpoint_name_prefix="hf")

        forward = model.gen_aoa_statements(ctx, structured_name_prefix="model.")
        targets = [statement.split(" -> ")[1] for statement in forward]

        # Persistable buffer participates like a parameter; non-persistable is
        # absent, mirroring ``sharded_state_dict`` / ``state_dict`` semantics.
        self.assertIn("model.leaf.weight", targets)
        self.assertIn("model.leaf.running_stat", targets)
        self.assertNotIn("model.leaf.tmp_cache", targets)
        self.assertEqual(
            set(forward),
            set(model.gen_aoa_statements(ctx, structured_name_prefix="model.")),
        )

    def test_recursion_dispatches_to_overridden_sublayer(self):
        model = paddle.nn.Layer()
        model.plain = _Leaf()
        model.special = _OverridingLeaf()
        ctx = _context(checkpoint_name_prefix="hf")

        forward = model.gen_aoa_statements(ctx, structured_name_prefix="model.")
        inverse = model.gen_inv_aoa_statements(
            ctx, structured_name_prefix="model."
        )

        # Plain leaf follows the default recursion; the overriding sublayer is
        # dispatched virtually and contributes only its custom sentinel, with
        # the structured-name prefix threaded through unchanged.
        self.assertIn("hf.plain.weight -> model.plain.weight", forward)
        self.assertIn("__custom_fwd__::model.special.", forward)
        self.assertIn("__custom_inv__::model.special.", inverse)
        self.assertTrue(
            all("special.weight" not in statement for statement in forward)
        )

    def test_ctx_maps_thread_through_recursion(self):
        # Every leaf must resolve against the *same* ctx maps; a RecordingMapping
        # confirms the nested structured names are the exact keys queried.
        model = paddle.nn.Layer()
        model.a = _Leaf()
        model.b = paddle.nn.Layer()
        model.b.c = _Leaf()
        pp_mapping = _RecordingMapping(
            {
                "a.weight": "model.a.weight",
                "b.c.weight": "model.b.c.weight",
            }
        )
        ctx = _context(checkpoint_name_prefix="hf", pp_mapping=pp_mapping)

        forward = model.gen_aoa_statements(ctx)

        self.assertEqual(sorted(pp_mapping.queries), ["a.weight", "b.c.weight"])
        self.assertEqual(
            forward,
            [
                "hf.a.weight -> model.a.weight",
                "hf.b.c.weight -> model.b.c.weight",
            ],
        )

    def test_aoa_name_scope_reroots_mtp_subtree_during_recursion(self):
        # An MTP subtree whose logical root differs from its actual module
        # root. The scope re-roots the checkpoint-relative path so the emitted
        # checkpoint name follows the logical (mtp) layout, independent of the
        # live module path used on the model side.
        name_mapping = {
            "weight": "weight",
        }
        model = paddle.nn.Layer()
        model.mtp = paddle.nn.Layer()
        model.mtp.block = _Leaf()
        scope = AOANameScope(
            checkpoint_prefix="mtp.0",
            logical_model_prefix="model.layers.1",
            actual_model_prefix="model.mtp.block",
        )
        ctx = _context(checkpoint_name_prefix="hf", name_mapping=name_mapping)

        forward = model.mtp.block.gen_aoa_statements(
            ctx,
            structured_name_prefix="model.mtp.block.",
            aoa_name_scope=scope,
        )

        self.assertEqual(len(forward), 1)
        source, target = forward[0].split(" -> ")
        # Checkpoint side is re-rooted under the logical mtp path; model side
        # keeps the live module path.
        self.assertTrue(source.startswith("hf.mtp.0"))
        self.assertTrue(target.startswith("model.mtp.block"))

    def test_aoa_name_scope_with_non_identity_mapping_threaded_from_root(self):
        # Stronger scope test: start from a root model, use a non-identity
        # checkpoint_name_mapping (leaf rename), and verify aoa_name_scope is
        # correctly threaded through multi-level recursion.
        name_mapping = {
            "layers.$LAYER_ID.mlp.down_proj.weight": (
                "layers.$LAYER_ID.ffn.w2.weight"
            ),
        }
        model = paddle.nn.Layer()
        model.mtp = paddle.nn.Layer()
        model.mtp.mlp = paddle.nn.Layer()
        model.mtp.mlp.down_proj = _Leaf()
        scope = AOANameScope(
            checkpoint_prefix="mtp.0",
            logical_model_prefix="model.layers.1",
            actual_model_prefix="model.mtp",
        )
        pp_mapping = {
            "model.mtp.mlp.down_proj.weight": (
                "model.mtp.mlp.down_proj.weight"
            ),
        }
        ctx = _context(
            checkpoint_name_prefix="hf",
            pp_mapping=pp_mapping,
            name_mapping=name_mapping,
        )

        # Start from root's child (mtp), threading aoa_name_scope down.
        forward = model.mtp.gen_aoa_statements(
            ctx,
            structured_name_prefix="model.mtp.",
            aoa_name_scope=scope,
        )

        self.assertEqual(len(forward), 1)
        # Should map through layers.1.mlp.down_proj.weight -> layers.1.ffn.w2.weight,
        # then strip layers.1 and re-anchor under mtp.0.
        self.assertEqual(
            forward[0],
            "hf.mtp.0.ffn.w2.weight -> model.mtp.mlp.down_proj.weight",
        )

        # Inverse should produce the reversed pair.
        inverse = model.mtp.gen_inv_aoa_statements(
            ctx,
            structured_name_prefix="model.mtp.",
            aoa_name_scope=scope,
        )
        self.assertEqual(
            inverse[0],
            "model.mtp.mlp.down_proj.weight -> hf.mtp.0.ffn.w2.weight",
        )

    def test_custom_model_name_prefix_resolves_correctly(self):
        # Simulates a multi-tower model (e.g. Qwen3-VL) where the model root
        # is "model.language_model" instead of the default "model".
        model = paddle.nn.Layer()
        model.language_model = paddle.nn.Layer()
        model.language_model.layers = paddle.nn.Layer()
        model.language_model.layers.proj = _Leaf()
        pp_mapping = {
            "model.language_model.layers.proj.weight": (
                "model.language_model.layers.proj.weight"
            ),
        }
        # A leaf mapping that renames "proj.weight" -> "linear.weight"
        name_mapping = {
            "layers.proj.weight": "layers.linear.weight",
        }
        ctx = _context(
            checkpoint_name_prefix="hf",
            pp_mapping=pp_mapping,
            name_mapping=name_mapping,
            model_name_prefix="model.language_model",
        )

        forward = model.gen_aoa_statements(ctx, structured_name_prefix="model.")
        self.assertEqual(len(forward), 1)
        # single_name = "model.language_model.layers.proj.weight"
        # strip "model.language_model" -> "layers.proj.weight"
        # match mapping -> "layers.linear.weight"
        # checkpoint = "hf.layers.linear.weight"
        self.assertEqual(
            forward[0],
            "hf.layers.linear.weight"
            " -> model.language_model.layers.proj.weight",
        )

    def test_custom_model_name_prefix_identity_fallback(self):
        # Without pp_mapping, the identity fallback must accept
        # structured names starting with "model.language_model".
        model = paddle.nn.Layer()
        model.language_model = paddle.nn.Layer()
        model.language_model.fc = _Leaf()
        ctx = _context(
            checkpoint_name_prefix="hf",
            model_name_prefix="model.language_model",
        )

        forward = model.gen_aoa_statements(ctx, structured_name_prefix="model.")
        self.assertEqual(len(forward), 1)
        # Identity: strip "model.language_model" -> "fc.weight"
        # No mapping -> "fc.weight" (identity)
        # checkpoint = "hf.fc.weight"
        self.assertEqual(
            forward[0],
            "hf.fc.weight -> model.language_model.fc.weight",
        )


if __name__ == "__main__":
    unittest.main()
