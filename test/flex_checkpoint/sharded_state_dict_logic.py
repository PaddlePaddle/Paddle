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

import math
import os

import paddle
from paddle import nn
from paddle.distributed import ShardedWeight, fleet
from paddle.distributed.fleet.layers.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
)


class SimpleMLP(
    nn.Layer
):  # embedding_weight_size=24*100=2400,it can't be divided by 256,which is using to check the padding logic
    def __init__(self, hidden_size=100, has_bias=False):
        super().__init__()
        self.embedding = VocabParallelEmbedding(24, hidden_size)
        self.linear1 = ColumnParallelLinear(
            hidden_size, hidden_size, gather_output=False, has_bias=has_bias
        )
        self.linear2 = RowParallelLinear(
            hidden_size, hidden_size, input_is_parallel=True, has_bias=has_bias
        )
        self.llm_head = self.embedding  # test the shared weight

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = paddle.matmul(x, self.llm_head.weight, transpose_y=True)
        return x


class TestParallelLayersLogic:
    def __init__(self):
        self.optimizer_var_suffix = [".moment1_0", ".moment2_0", ".w_0"]
        self.test_type = os.getenv("test_type")
        self.layer_type = os.getenv("layer_type")
        self.tp_degree = int(os.getenv("tp", "1"))
        self.dp_degree = int(os.getenv("dp", "1"))
        self.sharding_degree = int(os.getenv("sharding_degree", "1"))
        self.world_size = int(os.getenv("world_size"))
        self.has_bias = os.getenv("has_bias", "True").lower() == "true"
        self.master_weight = (
            os.getenv("master_weight", "False").lower() == "true"
        )
        self.batch_size = 2
        self.hidden_size = 32
        self.vocab_size = 24
        self.seq_len = 2
        self.hcg = None

    def run_test(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": self.dp_degree,
            "mp_degree": self.tp_degree,
            "sharding_degree": self.sharding_degree,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.hcg = fleet.get_hybrid_communicate_group()
        if self.test_type == "layer":
            self.run_layer_test()
        elif self.test_type == "optimizer":
            self.run_optimizer_test()
        else:
            raise ValueError(f"Unknown test_type: {self.test_type}")

    def run_layer_test(self):
        tp_group = self.hcg.get_model_parallel_group()
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
        model = SimpleMLP(has_bias=self.has_bias)
        model = paddle.amp.decorate(
            models=model, optimizers=None, level="O2", dtype="float16"
        )
        if self.master_weight:  # test the master_weight
            opt = paddle.optimizer.AdamW(
                learning_rate=0.01,
                parameters=model.parameters(),
                multi_precision=True,
            )
        else:
            opt = paddle.optimizer.AdamW(
                learning_rate=0.01,
                parameters=model.parameters(),
                multi_precision=False,
            )
        if self.layer_type == "AdamW":
            model = fleet.distributed_model(model)
            model.train()
            x = paddle.randint(
                low=0,
                high=self.vocab_size,
                shape=[self.batch_size, self.seq_len, self.hidden_size],
                dtype='int64',
            )
            y = model(x).mean()
            y.backward()
            opt.step()
            opt.clear_grad()

            model_sharded_state_dict = model.sharded_state_dict()
            opt_sharded_state_dict = opt.sharded_state_dict(
                model_sharded_state_dict
            )
            for key, value in model_sharded_state_dict.items():
                for state_name in self.optimizer_var_suffix:
                    opt__var_name = key + state_name
                    if opt__var_name in opt_sharded_state_dict:
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].local_shape
                        ) == tuple(value.local_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_shape
                        ) == tuple(value.global_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_offset
                        ) == tuple(value.global_offset)
        elif self.layer_type == "DygraphShardingOptimizer":
            opt = DygraphShardingOptimizer(opt, self.hcg)
            model.train()
            x = paddle.randint(
                low=0,
                high=self.vocab_size,
                shape=[self.batch_size, self.seq_len, self.hidden_size],
                dtype='int64',
            )
            rank = paddle.distributed.get_rank()
            sharidng_x = (
                x[0 : self.batch_size // 2]
                if rank == 0
                else x[self.batch_size // 2 :]
            )
            y = model(sharidng_x).mean()
            y.backward()
            opt.step()
            opt.clear_grad()

            model_sharded_state_dict = model.sharded_state_dict()
            opt_sharded_state_dict = opt.sharded_state_dict(
                model_sharded_state_dict
            )

            for key, value in model_sharded_state_dict.items():
                for state_name in self.optimizer_var_suffix:
                    opt__var_name = key + state_name
                    if opt__var_name in opt_sharded_state_dict:
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].local_shape
                        ) == tuple(value.local_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_shape
                        ) == tuple(value.global_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_offset
                        ) == tuple(value.global_offset)
        elif self.layer_type == "DygraphShardingOptimizerV2":
            opt = DygraphShardingOptimizerV2(opt, self.hcg)
            model.train()
            x = paddle.randint(
                low=0,
                high=self.vocab_size,
                shape=[self.batch_size, self.seq_len, self.hidden_size],
                dtype='int64',
            )
            rank = paddle.distributed.get_rank()
            sharidng_x = (
                x[0 : self.batch_size // 2]
                if rank == 0
                else x[self.batch_size // 2 :]
            )
            y = model(sharidng_x).mean()
            y.backward()
            opt.step()
            opt.clear_grad()

            model_sharded_state_dict = model.sharded_state_dict()
            opt_sharded_state_dict = opt.sharded_state_dict(
                model_sharded_state_dict
            )
            for key, value in model_sharded_state_dict.items():
                for state_name in self.optimizer_var_suffix:
                    opt__var_name = key + state_name
                    if opt__var_name in opt_sharded_state_dict:
                        if opt_sharded_state_dict[
                            opt__var_name
                        ].flattened_range.stop - opt_sharded_state_dict[
                            opt__var_name
                        ].flattened_range.start != math.prod(
                            value.local_shape
                        ):  # check the optimizer_var which isFragment
                            opt_var_globle_flattened_range = []
                            paddle.distributed.all_gather_object(
                                opt_var_globle_flattened_range,
                                opt_sharded_state_dict[
                                    opt__var_name
                                ].flattened_range,
                            )

                            first_fragment = opt_var_globle_flattened_range[0]
                            second_fragment = opt_var_globle_flattened_range[1]
                            assert (
                                first_fragment.stop == second_fragment.start
                            )  # the first_flattened_range_stop == the second_flattened_range_start
                            opt_var_globle_size_flattened = (
                                second_fragment.stop - first_fragment.start
                            )
                            model_var_globle_size_flattened = math.prod(
                                value.local_shape
                            )
                            assert (
                                opt_var_globle_size_flattened
                                == model_var_globle_size_flattened
                            )

                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].local_shape
                        ) == tuple(value.local_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_shape
                        ) == tuple(value.global_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_offset
                        ) == tuple(value.global_offset)

        elif self.layer_type == "GroupShardedOptimizerStage2":
            opt = GroupShardedOptimizerStage2(
                opt._parameter_list, opt, self.hcg.get_sharding_parallel_group()
            )

            model.train()
            x = paddle.randint(
                low=0,
                high=self.vocab_size,
                shape=[self.batch_size, self.seq_len, self.hidden_size],
                dtype='int64',
            )
            rank = paddle.distributed.get_rank()
            sharidng_x = (
                x[0 : self.batch_size // 2]
                if rank == 0
                else x[self.batch_size // 2 :]
            )
            y = model(sharidng_x).mean()
            y.backward()
            opt.step()
            opt.clear_grad()

            model_sharded_state_dict = model.sharded_state_dict()
            opt_sharded_state_dict = opt.sharded_state_dict(
                model_sharded_state_dict
            )

            for key, value in model_sharded_state_dict.items():
                for state_name in self.optimizer_var_suffix:
                    opt__var_name = key + state_name
                    if opt__var_name in opt_sharded_state_dict:
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].local_shape
                        ) == tuple(value.local_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_shape
                        ) == tuple(value.global_shape)
                        assert tuple(
                            opt_sharded_state_dict[opt__var_name].global_offset
                        ) == tuple(value.global_offset)
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")


if __name__ == '__main__':
    TestParallelLayersLogic().run_test()
