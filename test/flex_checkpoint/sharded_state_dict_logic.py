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

import os

from paddle import nn
from paddle.distributed import ShardedWeight, fleet
from paddle.distributed.fleet.layers.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
)


class SimpleMLPForSharding(nn.Layer):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class TestParallelLayersLogic:
    def __init__(self):
        self.test_type = os.getenv("test_type")
        self.layer_type = os.getenv("layer_type")
        self.tp_degree = int(os.getenv("tp"))
        self.dp_degree = int(os.getenv("dp"))
        self.world_size = int(os.getenv("world_size"))
        self.has_bias = os.getenv("has_bias", "True").lower() == "true"

        self.hidden_size = 32
        self.vocab_size = 1024

    def run_test(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": self.dp_degree,
            "mp_degree": self.tp_degree,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        if self.test_type == "layer":
            self.run_layer_test()
        elif self.test_type == "optimizer":
            self.run_optimizer_test()
        else:
            raise ValueError(f"Unknown test_type: {self.test_type}")

    def run_layer_test(self):
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        layer = self._get_layer()
        sharded_dict = layer.sharded_state_dict()
        self._verify_parallel_layer(
            sharded_dict, tp_group.rank, tp_group.nranks
        )

    def _get_layer(self):
        if self.layer_type == "ColumnParallelLinear":
            return ColumnParallelLinear(
                self.hidden_size, self.hidden_size * 2, has_bias=self.has_bias
            )
        elif self.layer_type == "RowParallelLinear":
            return RowParallelLinear(
                self.hidden_size * 2, self.hidden_size, has_bias=self.has_bias
            )
        elif self.layer_type == "VocabParallelEmbedding":
            return VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        elif self.layer_type == "ColumnSequenceParallelLinear":
            return ColumnSequenceParallelLinear(
                self.hidden_size,
                self.hidden_size * 2,
                has_bias=self.has_bias,
                gather_output=False,
            )
        elif self.layer_type == "RowSequenceParallelLinear":
            return RowSequenceParallelLinear(
                self.hidden_size * 2,
                self.hidden_size,
                has_bias=self.has_bias,
                input_is_parallel=True,
            )
        raise ValueError(f"Unknown layer_type: {self.layer_type}")

    def _verify_parallel_layer(self, sharded_dict, tp_rank, tp_world_size):
        if self.has_bias:
            assert 'bias' in sharded_dict
            bias_shard = sharded_dict['bias']
            assert isinstance(bias_shard, ShardedWeight)
        else:
            assert 'bias' not in sharded_dict

        assert 'weight' in sharded_dict
        weight_shard = sharded_dict['weight']
        assert isinstance(weight_shard, ShardedWeight)

        if self.layer_type == "ColumnParallelLinear":
            in_f, out_f = self.hidden_size, self.hidden_size * 2
            assert weight_shard.global_shape == (in_f, out_f)
            assert weight_shard.local_shape == (in_f, out_f // tp_world_size)
            assert weight_shard.global_offset == (
                0,
                tp_rank * (out_f // tp_world_size),
            )
            if self.has_bias:
                assert bias_shard.global_shape == (out_f,)
                assert bias_shard.local_shape == (out_f // tp_world_size,)
                assert bias_shard.global_offset == (
                    tp_rank * (out_f // tp_world_size),
                )

        elif self.layer_type == "RowParallelLinear":
            in_f, out_f = self.hidden_size * 2, self.hidden_size
            # Weight is sharded on axis 1
            assert weight_shard.global_shape == (in_f, out_f)
            assert weight_shard.local_shape == (in_f // tp_world_size, out_f)
            assert weight_shard.global_offset == (
                tp_rank * (in_f // tp_world_size),
                0,
            )

            if self.has_bias:
                # Bias is replicated, not sharded
                assert bias_shard.global_shape == [out_f]
                assert bias_shard.local_shape == bias_shard.global_shape
                assert bias_shard.global_offset == (0,)

        elif self.layer_type == "VocabParallelEmbedding":
            assert weight_shard.global_shape == (
                self.vocab_size,
                self.hidden_size,
            )
            assert weight_shard.local_shape == (
                self.vocab_size // tp_world_size,
                self.hidden_size,
            )
            assert weight_shard.global_offset == (
                tp_rank * (self.vocab_size // tp_world_size),
                0,
            )

        elif self.layer_type == "ColumnSequenceParallelLinear":
            in_f, out_f = self.hidden_size, self.hidden_size * 2
            assert weight_shard.global_shape == (in_f, out_f)
            assert weight_shard.local_shape == (in_f, out_f // tp_world_size)
            assert weight_shard.global_offset == (
                0,
                tp_rank * (out_f // tp_world_size),
            )
            if self.has_bias:
                assert bias_shard.global_shape == (out_f,)
                assert bias_shard.local_shape == (out_f // tp_world_size,)
                assert bias_shard.global_offset == (
                    tp_rank * (out_f // tp_world_size),
                )

        elif self.layer_type == "RowSequenceParallelLinear":
            in_f, out_f = self.hidden_size * 2, self.hidden_size
            assert weight_shard.global_shape == (in_f, out_f)
            assert weight_shard.local_shape == (in_f // tp_world_size, out_f)
            assert weight_shard.global_offset == (
                tp_rank * (in_f // tp_world_size),
                0,
            )
            if self.has_bias:
                assert bias_shard.global_shape == [out_f]
                assert bias_shard.local_shape == bias_shard.global_shape
                assert bias_shard.global_offset == (0,)

    def run_optimizer_test(self):
        # TODO(@zty-king): Add test for DygraphShardingOptimizerV2 and DygraphShardingOptimizer
        pass


if __name__ == '__main__':
    TestParallelLayersLogic().run_test()
