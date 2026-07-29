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

from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.nn import Layer

hidden_size = 16

# Sentinel used to emulate the pre-existing MoE expert color's 'group'
# (the moe_grad_group). The real MoE infrastructure attaches this before
# _construct_shared_comm runs; the shared-weight-sync logic must NOT overwrite it.
MOE_PRESET_GROUP = "moe_grad_group_sentinel"


class LocalLayer(Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)


class MoEMixedLayer(Layer):
    """Shared layer mixing a dense param and an emulated MoE expert param.

    The expert params are pre-marked with is_moe_param=True and a pre-existing
    color dict, mimicking what the real MoE setup does before
    _construct_shared_comm assigns the shared-weight-sync color. This lets the
    integration test exercise both the dense and the MoE branch of the color
    assignment without standing up a full MoE expert-parallel group.
    """

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.expert = nn.Linear(hidden_size, hidden_size)
        for p in self.expert.parameters():
            p.is_moe_param = True
            p.color = {
                "color": "moe_expert_original",
                "group": MOE_PRESET_GROUP,
            }

    @property
    def shared_weights(self):
        return self.named_parameters()

    def forward(self, x):
        return self.dense(x) + self.expert(x)


class SharedMoEPipe(PipelineLayer):
    def __init__(self, **kwargs):
        # With pp_degree=2 and uniform segmentation of 4 layers, stage 0 owns
        # index [0, 1] and stage 1 owns index [2, 3]. The two same-named shared
        # layers (index 0 and 3) therefore land on DIFFERENT stages / ranks,
        # which is the scenario this test targets.
        layers = [
            SharedLayerDesc(
                'shared_moe',
                MoEMixedLayer,
                shared_weight_attr='shared_weights',
            ),
            LayerDesc(LocalLayer),
            LayerDesc(LocalLayer),
            SharedLayerDesc(
                'shared_moe',
                MoEMixedLayer,
                shared_weight_attr='shared_weights',
            ),
        ]
        super().__init__(layers=layers, seg_method='uniform', **kwargs)


class MoENoColorLayer(Layer):
    """Shared layer whose MoE expert param is marked is_moe_param=True but has
    NO pre-existing color dict.

    This emulates the real failure scenario the reviewer flagged: nothing in the
    production code path guarantees an is_moe_param param carries a color before
    _construct_shared_comm runs. The shared-weight-sync logic must raise a clear
    configuration error in this case rather than crash with an opaque
    KeyError/AttributeError.
    """

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.expert = nn.Linear(hidden_size, hidden_size)
        for p in self.expert.parameters():
            p.is_moe_param = True
            # Intentionally do NOT assign p.color here.

    @property
    def shared_weights(self):
        return self.named_parameters()

    def forward(self, x):
        return self.dense(x) + self.expert(x)


class SharedMoENoColorPipe(PipelineLayer):
    def __init__(self, **kwargs):
        layers = [
            SharedLayerDesc(
                'shared_moe_no_color',
                MoENoColorLayer,
                shared_weight_attr='shared_weights',
            ),
            LayerDesc(LocalLayer),
            LayerDesc(LocalLayer),
            SharedLayerDesc(
                'shared_moe_no_color',
                MoENoColorLayer,
                shared_weight_attr='shared_weights',
            ),
        ]
        super().__init__(layers=layers, seg_method='uniform', **kwargs)


class TestSharedSubmoduleWeightOnlyDiffRank(unittest.TestCase):
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
        # sync_param must be on, otherwise _construct_shared_comm skips the
        # color-assignment block entirely.
        strategy.hybrid_configs['pp_configs'].sync_param = True
        strategy.hybrid_configs['pp_configs'].clear_every_step_cache = True

        fleet.init(is_collective=True, strategy=strategy)

    def test_shared_weight_color_across_ranks(self):
        hcg = fleet.get_hybrid_communicate_group()
        model = SharedMoEPipe(topology=hcg.topology())

        # Each stage that contains the shared layer builds its own local copy,
        # so both ranks own 'shared_moe'.
        self.assertIn('shared_moe', model.shared_layers)
        # The two shared layers sit on different stages, so a cross-stage comm
        # group must have been constructed on this rank.
        self.assertGreater(len(model.shared_comm), 0)

        shared_layer = model.shared_layers['shared_moe']
        sharding_group = hcg.get_sharding_parallel_group()

        dense_seen = False
        moe_seen = False
        for name, param in shared_layer.named_parameters():
            self.assertTrue(
                hasattr(param, 'color'),
                f'{name} should have a color assigned for shared sync.',
            )
            color = param.color
            self.assertEqual(color['shared_weight_name'], 'shared_weights')
            self.assertIn('broadcast_group', color)

            if getattr(param, 'is_moe_param', False):
                moe_seen = True
                # MoE branch: color renamed to the moe-experts sync color, but
                # the pre-existing group (moe_grad_group) MUST be preserved.
                self.assertIn('share_moe_experts', color['color'])
                self.assertEqual(color['group'], MOE_PRESET_GROUP)
            else:
                dense_seen = True
                # Dense branch: a fresh color built on the sharding group.
                self.assertIn('share_dense', color['color'])
                self.assertIs(color['group'], sharding_group)

        self.assertTrue(dense_seen, 'dense branch was not exercised')
        self.assertTrue(moe_seen, 'MoE branch was not exercised')

    def test_moe_param_without_color_raises(self):
        # Regression for the real missing-color scenario: an is_moe_param param
        # with no pre-assigned color must trigger a clear ValueError during
        # shared-comm construction, not an opaque KeyError/AttributeError.
        hcg = fleet.get_hybrid_communicate_group()
        with self.assertRaises(ValueError) as ctx:
            SharedMoENoColorPipe(topology=hcg.topology())
        self.assertIn('moe_grad_group', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
