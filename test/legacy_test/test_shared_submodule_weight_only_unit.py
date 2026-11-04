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
import warnings
from unittest.mock import MagicMock, patch

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel.parallel_layers.pp_layers import (
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.nn import Layer

hidden_size = 8


class SimpleTransformerLayer(Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    @property
    def transformer_layer_weights(self):
        return self.named_parameters()

    def forward(self, x):
        return self.linear2(self.linear1(x))


class MTPStyleLayer(Layer):
    def __init__(self):
        super().__init__()
        self.transformer_layer = SimpleTransformerLayer()
        self.proj = nn.Linear(hidden_size, hidden_size)

    @property
    def transformer_layer_weights(self):
        return self.transformer_layer.named_parameters()

    def forward(self, x):
        return self.proj(self.transformer_layer(x))


class TestAliasSharedLayerEdgeCases(unittest.TestCase):
    """Edge cases for _alias_shared_layer not covered by integration test."""

    def test_shape_mismatch_asserts(self):
        """Dest has params with different shapes than src -> assertion error."""
        src_layer = SimpleTransformerLayer()
        dest_layer = MTPStyleLayer()
        # Replace transformer_layer with different-shaped linears
        dest_layer.transformer_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        with self.assertRaises(AssertionError):
            PipelineLayer._alias_shared_layer(None, dest_layer, src_layer)

    def test_missing_param_asserts(self):
        """Src missing params that dest has -> assertion error."""
        src_layer = nn.Linear(hidden_size, hidden_size)
        dest_layer = MTPStyleLayer()
        with self.assertRaises(AssertionError):
            PipelineLayer._alias_shared_layer(None, dest_layer, src_layer)

    def test_setattr_fallback_path(self):
        """When param not in _parameters dict, uses setattr."""
        src_layer = SimpleTransformerLayer()
        dest_layer = MTPStyleLayer()
        # Pop param from _parameters to force setattr path
        inner = dest_layer.transformer_layer
        first_name = next(iter(inner.linear1._parameters.keys()))
        p = inner.linear1._parameters.pop(first_name)
        setattr(inner.linear1, first_name, p)

        PipelineLayer._alias_shared_layer(None, dest_layer, src_layer)
        src_params = dict(src_layer.named_parameters())
        for name, param in dest_layer.transformer_layer.named_parameters():
            self.assertIs(param, src_params[name])


class TestSynchronizeSharedWeightsEdgeCases(unittest.TestCase):
    """Branches in _synchronize_shared_weights not hit by integration test."""

    @patch('paddle.distributed.broadcast')
    def test_non_tensor_is_firstly_shared_false(self, mock_broadcast):
        """Non-Tensor obj: global_rank != min -> is_firstly_shared=False on all params."""
        layer = SimpleTransformerLayer()
        mock_group = MagicMock()
        shared_comm = {
            'k': {
                'layer': layer,
                'weight_attr': ['transformer_layer_weights'],
                'ranks': [0, 1],
                'group': mock_group,
            }
        }
        pipe = MagicMock()
        pipe.shared_comm = shared_comm
        pipe.global_rank = 1
        PipelineLayer._synchronize_shared_weights(pipe)
        for _, param in layer.named_parameters():
            self.assertFalse(param.is_firstly_shared)

    @patch('paddle.distributed.broadcast')
    def test_tensor_is_firstly_shared_false(self, mock_broadcast):
        """Tensor obj: global_rank != min -> is_firstly_shared=False."""
        layer = nn.Linear(hidden_size, hidden_size)
        mock_group = MagicMock()
        shared_comm = {
            'k': {
                'layer': layer,
                'weight_attr': ['weight'],
                'ranks': [0, 1],
                'group': mock_group,
            }
        }
        pipe = MagicMock()
        pipe.shared_comm = shared_comm
        pipe.global_rank = 1
        PipelineLayer._synchronize_shared_weights(pipe)
        self.assertFalse(layer.weight.is_firstly_shared)


class TestAllreduceSharedWeightGradientsEdgeCases(unittest.TestCase):
    """Branches in allreduce_shared_weight_gradients not hit by integration test."""

    @patch('paddle.distributed.all_reduce')
    def test_non_tensor_with_main_grad(self, mock_all_reduce):
        """Non-Tensor path with main_grad set."""
        layer = SimpleTransformerLayer()
        for _, param in layer.named_parameters():
            param.main_grad = paddle.ones(param.shape, dtype='float32')
        mock_group = MagicMock()
        shared_comm = {
            'k': {
                'layer': layer,
                'weight_attr': ['transformer_layer_weights'],
                'ranks': [0, 1],
                'group': mock_group,
            }
        }
        pipe = MagicMock()
        pipe.shared_comm = shared_comm
        with patch('paddle.framework.in_dynamic_mode', return_value=True):
            PipelineLayer.allreduce_shared_weight_gradients(pipe)
        num_params = len(list(layer.named_parameters()))
        self.assertEqual(mock_all_reduce.call_count, num_params)

    @patch('paddle.distributed.all_reduce')
    def test_non_tensor_with_none_main_grad_warns(self, mock_all_reduce):
        """Non-Tensor path with main_grad=None triggers warning."""
        layer = SimpleTransformerLayer()
        for _, param in layer.named_parameters():
            param.main_grad = None
        mock_group = MagicMock()
        shared_comm = {
            'k': {
                'layer': layer,
                'weight_attr': ['transformer_layer_weights'],
                'ranks': [0, 1],
                'group': mock_group,
            }
        }
        pipe = MagicMock()
        pipe.shared_comm = shared_comm
        with (
            patch('paddle.framework.in_dynamic_mode', return_value=True),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            PipelineLayer.allreduce_shared_weight_gradients(pipe)
            self.assertTrue(len(w) > 0)

    @patch('paddle.distributed.all_reduce')
    def test_non_tensor_with_none_grad_warns(self, mock_all_reduce):
        """Non-Tensor path without main_grad and grad=None triggers warning."""
        layer = SimpleTransformerLayer()
        mock_group = MagicMock()
        shared_comm = {
            'k': {
                'layer': layer,
                'weight_attr': ['transformer_layer_weights'],
                'ranks': [0, 1],
                'group': mock_group,
            }
        }
        pipe = MagicMock()
        pipe.shared_comm = shared_comm
        with (
            patch('paddle.framework.in_dynamic_mode', return_value=True),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            PipelineLayer.allreduce_shared_weight_gradients(pipe)
            self.assertTrue(len(w) > 0)

    @patch('paddle.distributed.all_reduce')
    def test_non_tensor_with_grad(self, mock_all_reduce):
        """Non-Tensor path with grad set (no main_grad)."""
        layer = SimpleTransformerLayer()
        for _, param in layer.named_parameters():
            param.grad = paddle.ones_like(param)
        mock_group = MagicMock()
        shared_comm = {
            'k': {
                'layer': layer,
                'weight_attr': ['transformer_layer_weights'],
                'ranks': [0, 1],
                'group': mock_group,
            }
        }
        pipe = MagicMock()
        pipe.shared_comm = shared_comm
        with patch('paddle.framework.in_dynamic_mode', return_value=True):
            PipelineLayer.allreduce_shared_weight_gradients(pipe)
        num_params = len(list(layer.named_parameters()))
        self.assertEqual(mock_all_reduce.call_count, num_params)


class TestSharedLayerDescNewParam(unittest.TestCase):
    """Test SharedLayerDesc.shared_submodule_weight_only attribute."""

    def test_default_false(self):
        desc = SharedLayerDesc('key', SimpleTransformerLayer)
        self.assertFalse(desc.shared_submodule_weight_only)

    def test_explicit_true(self):
        desc = SharedLayerDesc(
            'key',
            SimpleTransformerLayer,
            shared_submodule_weight_only=True,
            shared_weight_attr='transformer_layer_weights',
        )
        self.assertTrue(desc.shared_submodule_weight_only)


class TestBuildLayerTraditionalNonTensorPath(unittest.TestCase):
    """Cover traditional SharedLayerDesc path with non-Tensor weight_attr."""

    def test_is_firstly_shared_non_tensor_traditional(self):
        """Simulate _build_layer_impl traditional path: non-Tensor obj marks all params."""
        layer = SimpleTransformerLayer()
        # This is the exact logic in _build_layer_impl lines 1097-1106
        for weight_attr in ['transformer_layer_weights']:
            obj = getattr(layer, weight_attr)
            if isinstance(obj, paddle.Tensor):
                obj.is_firstly_shared = True
            else:
                for _, param in obj:
                    param.is_firstly_shared = True
        for _, param in layer.named_parameters():
            self.assertTrue(param.is_firstly_shared)


class TestAllreduceNonDynamicMode(unittest.TestCase):
    """Cover allreduce_shared_weight_gradients non-dynamic-mode (trace_op) branch."""

    def test_non_tensor_trace_op_path(
        self,
    ):
        """Non-dynamic mode uses trace_op for allreduce."""
        layer = SimpleTransformerLayer()
        for _, param in layer.named_parameters():
            param.grad = paddle.ones_like(param)
        mock_group = MagicMock()
        mock_group.id = 0
        shared_comm = {
            'k': {
                'layer': layer,
                'weight_attr': ['transformer_layer_weights'],
                'ranks': [0, 1],
                'group': mock_group,
            }
        }
        pipe = MagicMock()
        pipe.shared_comm = shared_comm
        mock_tracer = MagicMock()
        with (
            patch('paddle.framework.in_dynamic_mode', return_value=False),
            patch(
                'paddle.framework._dygraph_tracer',
                return_value=mock_tracer,
            ),
        ):
            PipelineLayer.allreduce_shared_weight_gradients(pipe)
        num_params = len(list(layer.named_parameters()))
        self.assertEqual(mock_tracer.trace_op.call_count, num_params)


class TestConstructSharedCommNWay(unittest.TestCase):
    """Unit test for _construct_shared_comm N-way grouping (mock new_group/topo).

    Covers >=3 sharing stages and heterogeneous attrs, which the pp_degree=2
    integration tests cannot reach.
    """

    def _run(self, descs, stage_of, global_rank=0, pp=4):
        from types import SimpleNamespace

        class _Topo:
            # No _parent_hcg attr -> _construct_shared_comm uses self._topo.
            def get_dim(self, name):
                return pp

            def get_comm_list(self, name):
                # One pipe comm list; maps stage idx -> global rank (identity).
                return [{0: 0, 1: 1, 2: 2, 3: 3}]

        pipe = SimpleNamespace(
            _topo=_Topo(),
            _layers_desc=descs,
            get_stage_from_index=lambda idx: stage_of[idx],
            global_rank=global_rank,
            shared_layers={"shared": nn.Linear(hidden_size, hidden_size)},
        )
        pp_cfg = SimpleNamespace(
            sync_moment=False, sync_param=False, shared_nccl_config=None
        )
        fake_fleet = MagicMock()
        fake_fleet._user_defined_strategy.hybrid_configs = {
            "pp_configs": pp_cfg
        }
        with (
            patch(
                'paddle.distributed.new_group',
                side_effect=lambda ranks, **kw: tuple(ranks),
            ),
            patch(
                'paddle.distributed.fleet.base.topology.message2nccl_config',
                return_value=None,
            ),
            patch('paddle.distributed.fleet.fleet', fake_fleet),
        ):
            comm = PipelineLayer._construct_shared_comm(pipe)
        # ranks-set -> sorted attrs
        return {
            tuple(v["ranks"]): sorted(v["weight_attr"]) for v in comm.values()
        }

    def test_three_stage_heterogeneous_attrs(self):
        """w/w2 on stages {0,1,2}, b only on {0,1}: one N-way group per stage set,
        and attrs on the same stage set are aggregated into a single group."""
        descs = [
            SharedLayerDesc(
                "shared", nn.Linear, shared_weight_attr=["w", "w2", "b"]
            ),
            SharedLayerDesc(
                "shared", nn.Linear, shared_weight_attr=["w", "w2", "b"]
            ),
            SharedLayerDesc(
                "shared", nn.Linear, shared_weight_attr=["w", "w2"]
            ),
        ]
        got = self._run(descs, {0: 0, 1: 1, 2: 2})
        # Exactly two groups: the 3-way set and the 2-way set.
        self.assertEqual(len(got), 2)
        # 3-way group covers all stages and aggregates w, w2 (NOT one group per attr).
        self.assertEqual(got[(0, 1, 2)], ["w", "w2"])
        # b only lives on stages {0,1} -> its own N-way(=2) group.
        self.assertEqual(got[(0, 1)], ["b"])

    def test_two_stage_multi_attr_single_group(self):
        """N=2 with k attrs builds ONE group (same communicator count as the
        original pairwise code), not k groups."""
        descs = [
            SharedLayerDesc("shared", nn.Linear, shared_weight_attr=["w", "b"]),
            SharedLayerDesc("shared", nn.Linear, shared_weight_attr=["w", "b"]),
        ]
        got = self._run(descs, {0: 0, 1: 1})
        self.assertEqual(len(got), 1)
        self.assertEqual(got[(0, 1)], ["b", "w"])


if __name__ == '__main__':
    unittest.main()
