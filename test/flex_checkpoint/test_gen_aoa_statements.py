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
# part, in what order, how ``ctx`` / the structured-name prefix / the checkpoint
# lookup drop segment thread down through nesting, and when a sublayer's own
# override takes over. The stateless name helpers they call live in
# ``flex_checkpoint/aoa/generation.py`` and are pinned directly by
# ``test_aoa_generation.py``.

import unittest

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext


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
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
    ):
        return [f"__custom_fwd__::{structured_name_prefix}"]

    def gen_inv_aoa_statements(
        self,
        ctx,
        *,
        structured_name_prefix="",
        checkpoint_lookup_drop_segment=None,
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
    model_name_prefix="model",
):
    return AOAContext(
        config=None,
        checkpoint_name_prefix=checkpoint_name_prefix,
        pp_to_single_mapping=pp_mapping or {},
        checkpoint_name_mapping=name_mapping or {},
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

    def test_drop_segment_lets_an_mtp_subtree_reuse_layer_mapping(self):
        # An MTP block holds its transformer layer one module deeper than the
        # checkpoint does. Dropping that segment before the lookup is what lets
        # the rule written for an ordinary layer hit inside the subtree; the
        # model side keeps the live module path.
        name_mapping = {
            "model.layers.$LAYER_ID.mlp.down_proj.weight": (
                "hf.layers.$LAYER_ID.ffn.w2.weight"
            ),
        }
        model = paddle.nn.Layer()
        model.transformer_layer = paddle.nn.Layer()
        model.transformer_layer.mlp = paddle.nn.Layer()
        model.transformer_layer.mlp.down_proj = _Leaf()
        ctx = _context(checkpoint_name_prefix="hf", name_mapping=name_mapping)

        forward = model.gen_aoa_statements(
            ctx,
            structured_name_prefix="model.layers.13.",
            checkpoint_lookup_drop_segment="transformer_layer",
        )
        inverse = model.gen_inv_aoa_statements(
            ctx,
            structured_name_prefix="model.layers.13.",
            checkpoint_lookup_drop_segment="transformer_layer",
        )

        self.assertEqual(
            forward,
            [
                "hf.layers.13.ffn.w2.weight"
                " -> model.layers.13.transformer_layer.mlp.down_proj.weight"
            ],
        )
        self.assertEqual(
            inverse,
            [
                "model.layers.13.transformer_layer.mlp.down_proj.weight"
                " -> hf.layers.13.ffn.w2.weight"
            ],
        )

    def test_drop_segment_is_a_noop_for_children_without_it(self):
        # The subtree owner passes one value down to every child, including the
        # ones that sit directly under the block rather than under the nested
        # transformer layer.
        model = paddle.nn.Layer()
        model.enorm = _Leaf()
        model.transformer_layer = paddle.nn.Layer()
        model.transformer_layer.norm = _Leaf()
        ctx = _context(checkpoint_name_prefix="hf")

        forward = model.gen_aoa_statements(
            ctx,
            structured_name_prefix="model.layers.13.",
            checkpoint_lookup_drop_segment="transformer_layer",
        )

        self.assertEqual(
            forward,
            [
                "hf.layers.13.enorm.weight -> model.layers.13.enorm.weight",
                "hf.layers.13.norm.weight"
                " -> model.layers.13.transformer_layer.norm.weight",
            ],
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
        # A leaf mapping that renames "proj.weight" -> "linear.weight"; both
        # sides are absolute, so the key carries the tower root and the value
        # carries the checkpoint root.
        name_mapping = {
            "model.language_model.layers.proj.weight": (
                "hf.layers.linear.weight"
            ),
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
        # matches the key as written -> "hf.layers.linear.weight", which is
        # already the full checkpoint name.
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
