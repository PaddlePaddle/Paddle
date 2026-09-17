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

from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.nn import Layer

hidden_size = 16


class LocalLayer(Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)


class TransformerLayer(Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    @property
    def transformer_layer_weights(self):
        return self.named_parameters()

    def forward(self, x):
        return self.norm(self.linear2(self.linear1(x)))


class MTPLayer(Layer):
    def __init__(self):
        super().__init__()
        self.transformer_layer = TransformerLayer()
        self.proj = nn.Linear(hidden_size, hidden_size)

    @property
    def transformer_layer_weights(self):
        return self.transformer_layer.named_parameters()

    def forward(self, x):
        return self.proj(self.transformer_layer(x))


class SharedSubmodulePipe(PipelineLayer):
    def __init__(self, **kwargs):
        layers = [
            LayerDesc(LocalLayer),
            LayerDesc(LocalLayer),
            SharedLayerDesc(
                'shared_transformer',
                TransformerLayer,
                shared_weight_attr='transformer_layer_weights',
                shared_submodule_weight_only=True,
            ),
            SharedLayerDesc(
                'shared_transformer',
                MTPLayer,
                shared_weight_attr='transformer_layer_weights',
                shared_submodule_weight_only=True,
            ),
        ]
        super().__init__(layers=layers, seg_method='layer:LocalLayer', **kwargs)


class VPPSharedSubmodulePipe(PipelineLayer):
    """Shared submodule layers spread across two virtual chunks.

    The second shared layer is deliberately placed inside (rather than at the
    beginning of) the second chunk.  Before the VPP registration fix this made
    the layer appear both under the chunk and as a direct child of the root
    PipelineLayer.
    """

    def __init__(self, **kwargs):
        layers = [
            SharedLayerDesc(
                'shared_transformer',
                TransformerLayer,
                shared_weight_attr='transformer_layer_weights',
                shared_submodule_weight_only=True,
            ),
            LayerDesc(LocalLayer),
            LayerDesc(LocalLayer),
            LayerDesc(nn.Identity),
            SharedLayerDesc(
                'shared_transformer',
                MTPLayer,
                shared_weight_attr='transformer_layer_weights',
                shared_submodule_weight_only=True,
            ),
            LayerDesc(LocalLayer),
            LayerDesc(LocalLayer),
            LayerDesc(nn.Identity),
        ]
        super().__init__(
            layers=layers,
            seg_method='layer:LocalLayer',
            num_virtual_pipeline_stages=2,
            **kwargs,
        )


class TestSharedSubmoduleWeightOnly(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            'dp_degree': 1,
            'mp_degree': 1,
            'pp_degree': 2,
        }
        strategy.pipeline_configs = {
            'accumulate_steps': 1,
            'micro_batch_size': 1,
        }
        strategy.hybrid_configs['pp_configs'].clear_every_step_cache = True

        fleet.init(is_collective=True, strategy=strategy)

    def test_shared_submodule_weight_only(self):
        hcg = fleet.get_hybrid_communicate_group()
        model = SharedSubmodulePipe(topology=hcg.topology())

        if hcg.get_stage_id() != 1:
            return

        transformer_layer = None
        mtp_layer = None
        for layer in model.run_function:
            if isinstance(layer, TransformerLayer):
                transformer_layer = layer
            elif isinstance(layer, MTPLayer):
                mtp_layer = layer

        source_params = dict(transformer_layer.named_parameters())
        for name, param in mtp_layer.transformer_layer.named_parameters():
            self.assertIs(
                param,
                source_params[name],
                f'{name} should be aliased to the source transformer layer.',
            )

    def test_shared_submodule_weight_only_vpp_mapping(self):
        hcg = fleet.get_hybrid_communicate_group()
        model = VPPSharedSubmodulePipe(topology=hcg.topology())

        chunks = model.get_model_chunks()
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 2)

        # Both shared instances are on stage 0.  Stage 1 has no matching
        # SharedLayerDesc and therefore has nothing to validate here.
        if hcg.get_stage_id() != 0:
            return

        source_layer = next(
            layer for layer in chunks[0] if isinstance(layer, TransformerLayer)
        )
        dest_layer = next(
            layer for layer in chunks[1] if isinstance(layer, MTPLayer)
        )

        source_params = dict(source_layer.named_parameters())
        for name, param in dest_layer.transformer_layer.named_parameters():
            self.assertIs(
                param,
                source_params[name],
                f'{name} should be aliased to the source transformer layer.',
            )

        # The VPP instance must be registered by PipelineLayerChunk.append,
        # not as an additional direct child of the root PipelineLayer.
        self.assertTrue(
            any(layer is dest_layer for _, layer in chunks[1].named_children())
        )
        self.assertFalse(
            any(layer is dest_layer for _, layer in model.named_children())
        )


if __name__ == '__main__':
    unittest.main()
