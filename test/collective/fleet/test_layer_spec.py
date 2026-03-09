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

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
    get_spec_layer,
    import_spec_layer,
)
from paddle.nn import Identity


class SimpleRotaryEmbeddingForTest(paddle.nn.Layer):
    def __init__(
        self,
        head_dim: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_percent = rotary_percent
        self.rotary_interleaved = rotary_interleaved
        self.rotary_base = rotary_base
        self.rope_scaling = rope_scaling


class TestSpecCustomization(unittest.TestCase):
    def setUp(self):
        head_dim = 128
        rotary_base = 10000
        rotary_percent = 1.0
        rope_scaling = False

        rotary_emb_extra_kwargs = {
            "head_dim": head_dim // 2,
            "rotary_base": rotary_base,
            "rope_scaling": rope_scaling,
            "rotary_percent": rotary_percent,
        }
        self.embedding_spec = LayerSpec(
            layer=SimpleRotaryEmbeddingForTest,
            extra_kwargs=rotary_emb_extra_kwargs,
        )

    def tearDown(self):
        pass

    def test_import_layer(self):
        identity_cls = import_spec_layer(
            layer_path=('paddle.nn.layer.common', 'Identity')
        )
        self.assertEqual(identity_cls, Identity)

    def test_build_layer(self):
        head_dim = 128
        rotary_percent = 1.0
        rotary_base = 10000
        rope_scaling = False

        self.rotary_pos_emb = build_spec_layer(
            self.embedding_spec,
            head_dim=head_dim,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
        )
        self.assertIsInstance(self.rotary_pos_emb, SimpleRotaryEmbeddingForTest)

    def test_layer_spec_basic(self):
        """Test basic LayerSpec functionality."""
        # Test LayerSpec with class
        spec = LayerSpec(layer=Identity)
        self.assertEqual(spec.layer, Identity)
        self.assertEqual(spec.extra_kwargs, {})
        self.assertIsNone(spec.sublayers_spec)

        # Test LayerSpec with extra_kwargs
        spec_with_kwargs = LayerSpec(
            layer=Identity, extra_kwargs={"test_param": 123}
        )
        self.assertEqual(spec_with_kwargs.extra_kwargs, {"test_param": 123})

        # Test LayerSpec with layer path tuple
        spec_with_path = LayerSpec(layer=("paddle.nn.layer.common", "Identity"))
        self.assertEqual(
            spec_with_path.layer, ("paddle.nn.layer.common", "Identity")
        )

    def test_build_layer_with_identity_op(self):
        """Test building layers with Identity."""
        # Build Identity
        identity = build_spec_layer(LayerSpec(layer=Identity))
        self.assertIsInstance(identity, Identity)

        # Test forward pass
        test_input = paddle.randn([2, 3])
        result = identity(test_input)
        self.assertEqual(result.shape, test_input.shape)

    def test_get_spec_layer_with_type(self):
        """Test get_spec_layer with a type directly."""
        # Test with a type directly
        result = get_spec_layer(Identity)
        self.assertEqual(result, Identity)

    def test_get_spec_layer_with_layer_spec(self):
        """Test get_spec_layer with LayerSpec containing a type."""
        # Test with LayerSpec containing a type
        spec = LayerSpec(layer=Identity)
        result = get_spec_layer(spec)
        self.assertEqual(result, Identity)

    def test_get_spec_layer_with_layer_path(self):
        """Test get_spec_layer with LayerSpec containing a layer path tuple."""
        # Test with LayerSpec containing a layer path
        spec = LayerSpec(layer=("paddle.nn.layer.common", "Identity"))
        result = get_spec_layer(spec)
        self.assertEqual(result, Identity)

    def test_get_spec_layer_with_function(self):
        """Test get_spec_layer with a function."""
        # Test with a function (using a lambda for testing)
        test_func = lambda x: x
        result = get_spec_layer(test_func)
        self.assertEqual(result, test_func)

    def test_get_spec_layer_with_extra_kwargs(self):
        """Test get_spec_layer with extra kwargs (should be ignored)."""
        # extra_kwargs should be ignored by get_spec_layer
        spec = LayerSpec(layer=Identity, extra_kwargs={"test_param": 123})
        result = get_spec_layer(spec, additional_arg=456)
        self.assertEqual(result, Identity)

    def test_layer_spec_repr_with_tuple(self):
        """Test LayerSpec __repr__ with tuple layer."""
        # Test with a tuple of layer path
        spec = LayerSpec(layer=("paddle.nn.layer.common", "Identity"))
        repr_str = repr(spec)
        self.assertIn("paddle.nn.layer.common", repr_str)
        self.assertIn("Identity", repr_str)

    def test_import_spec_layer_with_import_error(self):
        """Test import_spec_layer with invalid import path."""
        # Test with invalid module path that raises ImportError
        result = import_spec_layer(("nonexistent.module.path", "SomeClass"))
        self.assertIsNone(result)

    def test_build_spec_layer_with_function(self):
        """Test build_spec_layer with a function directly."""
        # Test with a function directly
        test_func = lambda x: x * 2
        result = build_spec_layer(test_func)
        self.assertEqual(result, test_func)

    def test_build_spec_layer_with_layer_spec_function(self):
        """Test build_spec_layer with LayerSpec containing a function."""
        # Test with LayerSpec containing a function
        test_func = lambda x: x + 1
        spec = LayerSpec(layer=test_func, extra_kwargs={})
        result = build_spec_layer(spec)
        self.assertEqual(result, test_func)

    def test_build_spec_layer_with_type(self):
        """Test build_spec_layer with a type directly."""
        # Test with a type directly - should instantiate the class
        result = build_spec_layer(Identity)
        self.assertIsInstance(result, Identity)

    def test_build_spec_layer_with_layer_spec_type(self):
        """Test build_spec_layer with LayerSpec containing a type."""
        # Test with LayerSpec containing a type (class)
        spec = LayerSpec(layer=Identity)
        result = build_spec_layer(spec)
        self.assertIsInstance(result, Identity)

    def test_build_spec_layer_with_imported_function(self):
        """Test build_spec_layer with layer path that imports a function."""
        # Test with LayerSpec containing a layer path that resolves to a function
        # Using a simple function from paddle.nn.functional
        spec = LayerSpec(layer=("paddle.nn.functional", "relu"))
        result = build_spec_layer(spec)
        # The imported layer should be a function
        self.assertTrue(callable(result))

    def test_build_spec_layer_exception_handling(self):
        """Test build_spec_layer with invalid kwargs that cause exception."""
        # Test that exception is raised with improved error message
        # Use paddle.nn.Linear which requires in_features and out_features
        # Passing empty extra_kwargs will cause TypeError at instantiation
        spec = LayerSpec(layer=paddle.nn.Linear, extra_kwargs={})
        with self.assertRaises(Exception) as context:
            build_spec_layer(spec)
        # Verify the error message includes the layer name
        self.assertIn("Linear", str(context.exception))


if __name__ == "__main__":
    unittest.main()
