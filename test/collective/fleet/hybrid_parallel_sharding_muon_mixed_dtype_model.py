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
# Real single-machine 2-card Muon test for per-dtype 2D owner partitioning in
# MuonShardingOptimizer._partition_2d_parameters.
#
# A FusedCommBuffer never mixes dtypes, because AssignGroupBySize keys its
# groups on dtype. The partitioner therefore bin-packs each dtype separately:
# packing the whole color group at once sizes active_ranks from the group total,
# which scatters a dtype that holds only a small share of the parameters over
# far more ranks than its own volume needs, leaving one tiny buffer per rank.
#
# The model below puts bf16 and fp32 2D params in the same (default) color
# group and picks a bucket size that makes the two strategies disagree, then
# asserts:
#   1. every rank derives the same owner mapping (if they disagreed, the reduce
#      dst and the broadcast src would not match and the job would hang),
#   2. bf16 spreads over both ranks while fp32 concentrates on rank 0,
#   3. the fp32 params end up in a single FusedCommBuffer,
#   4. after real optimizer steps all ranks hold bit-identical parameters.
#
# Assertions 1-3 are guarded by an explicit regime check, so the case cannot
# silently degenerate into the trivial "everything on rank 0" partition that a
# small model with the default 256 MB bucket would produce.

import os
import random
import unittest
from collections import defaultdict

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.optimizer.muon import MuonParamInfo, _default_should_use_muon

os.environ["MUON_DEBUG"] = "0"

SHARDING_DEGREE = 2

# comm_buffer_size_MB is an int32 in distributed_strategy.proto, so 1 MB is the
# smallest usable bucket. The shapes are chosen relative to it, using the same
# `numel * 4` volume estimate the partitioner applies to every dtype:
#   bf16: 4 x [512, 512] -> 4 x 1.0  MB = 4.0 MB -> min(int(4.0/1)+1, 2) = 2 ranks
#   fp32: 2 x [256, 256] -> 2 x 0.25 MB = 0.5 MB -> min(int(0.5/1)+1, 2) = 1 rank
# Packing all six together instead -- what the code did before per-dtype
# partitioning -- gives active_ranks = 2 for the whole group, and the greedy
# fill then lands one fp32 param on each rank, i.e. two one-param fp32 buffers.
COMM_BUFFER_SIZE_MB = 1
BF16_DIM = 512
N_BF16_LAYERS = 4
FP32_DIM = 256
N_FP32_LAYERS = 2

BATCH_SIZE = 16  # divisible by SHARDING_DEGREE
STEPS = 3

EXPECTED_BF16_OWNERS = {0, 1}
EXPECTED_FP32_OWNERS = {0}


def _linear(dim, np_weight):
    return paddle.nn.Linear(
        dim,
        dim,
        bias_attr=False,
        weight_attr=paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Assign(np_weight)
        ),
    )


class MixedDtypeNet(paddle.nn.Layer):
    """Two branches whose 2D weights deliberately end up in different dtypes.

    ``bf16_layers`` is cast by ``amp.decorate(level='O2')``; ``fp32_layers`` is
    passed through ``excluded_layers`` and keeps float32 weights. Both live in
    the same (default) color group, which is the configuration that made the old
    whole-group bin-packing scatter the minority dtype.

    Each branch casts its own input explicitly rather than relying on
    ``auto_cast``, so the parameter dtypes the partitioner sees are unambiguous.
    """

    def __init__(self, bf16_weights, fp32_weights):
        super().__init__()
        self.bf16_layers = paddle.nn.LayerList(
            [_linear(BF16_DIM, w) for w in bf16_weights]
        )
        self.fp32_layers = paddle.nn.LayerList(
            [_linear(FP32_DIM, w) for w in fp32_weights]
        )

    def forward(self, x):
        h = x.astype("bfloat16")
        for fc in self.bf16_layers:
            h = fc(h)
        g = x[:, :FP32_DIM]
        for fc in self.fp32_layers:
            g = fc(g)
        return h.astype("float32").mean() + g.mean()


def _init_weights():
    """Initial weights, identical on every rank."""
    return (
        [
            np.random.random_sample((BF16_DIM, BF16_DIM)).astype("float32")
            for _ in range(N_BF16_LAYERS)
        ],
        [
            np.random.random_sample((FP32_DIM, FP32_DIM)).astype("float32")
            for _ in range(N_FP32_LAYERS)
        ],
    )


def _build_model(weights):
    model = MixedDtypeNet(*weights)
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    paddle.amp.decorate(
        models=model,
        level="O2",
        dtype="bfloat16",
        excluded_layers=[model._layers.fp32_layers],
    )
    return model


def _build_optimizer(model):
    muon_param_info_map = {}
    for name, param in model.named_parameters():
        muon_param_info_map[param.name] = MuonParamInfo(
            use_muon=_default_should_use_muon(name, param.shape, []),
            split_concat_func=None,
        )
    return paddle.optimizer.Muon(
        parameters=model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        muon_param_info_map=muon_param_info_map,
        ns_steps=5,
        ns_coeff_type="simple",
        multi_precision=True,
    )


def _short_dtype(dtype):
    return str(dtype).split(".")[-1]


class TestMuonMixedDtypePartition(unittest.TestCase):
    def setUp(self):
        random.seed(2024)
        np.random.seed(2024)
        paddle.seed(2024)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "sharding_degree": SHARDING_DEGREE,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        strategy.use_muon_sharding = True
        sharding_configs = strategy.hybrid_configs["sharding_configs"]
        sharding_configs.accumulate_steps = 1
        sharding_configs.comm_buffer_size_MB = COMM_BUFFER_SIZE_MB

        fleet.init(is_collective=True, strategy=strategy)

        # Identical full dataset on every rank; each rank slices its own shard.
        self.data = [
            np.random.random_sample((BATCH_SIZE, BF16_DIM)).astype("float32")
            for _ in range(STEPS)
        ]

    @staticmethod
    def _owner_map(inner_opt):
        """{color -> {param name -> owner rank}}, as every rank computed it."""
        return {
            str(color): dict(name2rank)
            for color, name2rank in inner_opt._param2rank_2d_by_color.items()
        }

    @staticmethod
    def _dtype_map(inner_opt):
        return {
            p.name: _short_dtype(p.dtype)
            for params in inner_opt._params_2d_by_color.values()
            for p in params
        }

    def _owners_by_dtype(self, owner_map, dtype_map):
        owners = defaultdict(set)
        for name2rank in owner_map.values():
            for name, rank in name2rank.items():
                owners[dtype_map[name]].add(rank)
        return owners

    def _train(self, model, optimizer):
        hcg = fleet.get_hybrid_communicate_group()
        sharding_rank = hcg.get_sharding_parallel_rank()
        local_bs = BATCH_SIZE // SHARDING_DEGREE
        for idx in range(STEPS):
            start = sharding_rank * local_bs
            batch = paddle.to_tensor(self.data[idx][start : start + local_bs])
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

    def test_mixed_dtype_partition(self):
        model = _build_model(_init_weights())
        optimizer = _build_optimizer(model)
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        inner_opt = optimizer._inner_opt
        self.assertIsInstance(inner_opt, MuonShardingOptimizer)

        owner_map = self._owner_map(inner_opt)
        dtype_map = self._dtype_map(inner_opt)
        owners = self._owners_by_dtype(owner_map, dtype_map)

        # Regime guard: without this the assertions below could hold vacuously,
        # which is what happens with the default 256 MB bucket and a small model
        # (active_ranks collapses to 1 and every param lands on rank 0).
        self.assertEqual(
            set(owners),
            {"bfloat16", "float32"},
            msg=(
                "the two dtypes were not both present among 2D params; "
                f"got {dict(owners)} -- amp.decorate/excluded_layers did not "
                "produce the intended mixed-dtype color group"
            ),
        )

        # bf16 volume needs both ranks, fp32 volume needs only one. Under the
        # previous whole-group packing fp32 would be spread over both ranks.
        self.assertEqual(
            owners["bfloat16"],
            EXPECTED_BF16_OWNERS,
            msg=(
                "bf16 params should span both ranks, so the partition is not "
                f"in the degenerate single-rank regime; got {owners['bfloat16']}"
            ),
        )
        self.assertEqual(
            owners["float32"],
            EXPECTED_FP32_OWNERS,
            msg=(
                "fp32 params should concentrate on a single rank sized by their "
                f"own volume; got {owners['float32']}"
            ),
        )

        # Every rank builds this mapping independently. If they disagreed, the
        # dst of the gradient reduce and the src of the parameter broadcast
        # would not line up and the job would hang instead of failing.
        gathered_maps = []
        paddle.distributed.all_gather_object(gathered_maps, owner_map)
        for rank, other in enumerate(gathered_maps):
            self.assertEqual(
                other,
                gathered_maps[0],
                msg=(
                    f"rank {rank} derived a different 2D owner mapping than "
                    "rank 0; ranks must agree or collectives will mismatch"
                ),
            )

        # The point of per-dtype packing: the minority dtype fuses into one
        # buffer instead of one small buffer per rank it was scattered over.
        buffers_per_dtype = defaultdict(int)
        for buf in inner_opt.comm_buffer_2d:
            buffers_per_dtype[_short_dtype(buf._params[0].dtype)] += 1
        self.assertEqual(
            buffers_per_dtype["float32"],
            1,
            msg=(
                "expected the fp32 2D params to fuse into a single comm "
                f"buffer; got {dict(buffers_per_dtype)}"
            ),
        )

        self._train(model, optimizer)

        # 2D params are updated by their owner and broadcast back, so after the
        # steps above every rank must hold bit-identical values.
        for name, param in model.named_parameters():
            gathered = []
            paddle.distributed.all_gather(gathered, param.cast("float32"))
            reference = gathered[0].numpy()
            for rank, other in enumerate(gathered):
                np.testing.assert_array_equal(
                    other.numpy(),
                    reference,
                    err_msg=(
                        f"param {name!r} differs between rank {rank} and rank "
                        "0 after the optimizer steps; owner update and "
                        "broadcast are inconsistent"
                    ),
                )

        if paddle.distributed.get_rank() == 0:
            print(
                "[PASS] per-dtype 2D partition: "
                f"bf16 owners={sorted(owners['bfloat16'])}, "
                f"fp32 owners={sorted(owners['float32'])}, "
                f"buffers per dtype={dict(buffers_per_dtype)}"
            )


if __name__ == "__main__":
    unittest.main()
