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
# The model below puts a low-precision dtype and fp32 2D params in the same
# (default) color group and picks a bucket size that makes the two strategies
# disagree, then asserts:
#   1. every rank derives the same owner mapping (if they disagreed, the reduce
#      dst and the broadcast src would not match and the job would hang),
#   2. the low-precision params spread over both ranks while fp32 concentrates
#      on rank 0,
#   3. the fp32 params end up in a single FusedCommBuffer,
#   4. after real optimizer steps all ranks hold bit-identical parameters.
#
# Assertions 1-3 are guarded by an explicit regime check, so the case cannot
# silently degenerate into the trivial "everything on rank 0" partition that a
# small model with the default 256 MB bucket would produce.
#
# The low-precision dtype is whichever of bf16/fp16 the device actually has.
# ``amp.decorate`` returns without casting anything when it does not (auto_cast
# gates bf16 on Compute Capability >= 8 and fp16 on >= 7) and says nothing about
# it, so the candidates are applied and the parameter dtype read back rather
# than predicted. This matters: CI runs these distributed cases on V100
# (CC 7.0), where bf16 is unavailable and fp16 is.

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
# `numel * 4` volume estimate the partitioner applies to every dtype, so they
# hold whether the low-precision branch ends up bf16 or fp16:
#   low : 4 x [512, 512] -> 4 x 1.0  MB = 4.0 MB -> min(int(4.0/1)+1, 2) = 2 ranks
#   fp32: 2 x [256, 256] -> 2 x 0.25 MB = 0.5 MB -> min(int(0.5/1)+1, 2) = 1 rank
# Packing all six together instead -- what the code did before per-dtype
# partitioning -- gives active_ranks = 2 for the whole group, and the greedy
# fill then lands one fp32 param on each rank, i.e. two one-param fp32 buffers.
COMM_BUFFER_SIZE_MB = 1
LOW_DIM = 512
N_LOW_LAYERS = 4
FP32_DIM = 256
N_FP32_LAYERS = 2

BATCH_SIZE = 16  # divisible by SHARDING_DEGREE
STEPS = 3

# Most capable first; the first one the device accepts is used.
LOW_DTYPE_CANDIDATES = ("bfloat16", "float16")

EXPECTED_LOW_OWNERS = {0, 1}
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

    ``low_layers`` is cast by ``amp.decorate(level='O2')``; ``fp32_layers`` is
    passed through ``excluded_layers`` and keeps float32 weights. Both live in
    the same (default) color group, which is the configuration that made the old
    whole-group bin-packing scatter the minority dtype.

    Each branch casts its own input to whatever dtype its weights actually have,
    rather than relying on ``auto_cast``, so the parameter dtypes the
    partitioner sees are unambiguous and the forward works on either dtype.
    """

    def __init__(self, low_weights, fp32_weights):
        super().__init__()
        self.low_layers = paddle.nn.LayerList(
            [_linear(LOW_DIM, w) for w in low_weights]
        )
        self.fp32_layers = paddle.nn.LayerList(
            [_linear(FP32_DIM, w) for w in fp32_weights]
        )

    def forward(self, x):
        h = x.astype(self.low_layers[0].weight.dtype)
        for fc in self.low_layers:
            h = fc(h)
        g = x[:, :FP32_DIM]
        for fc in self.fp32_layers:
            g = fc(g)
        return h.astype("float32").mean() + g.mean()


def _init_weights():
    """Initial weights, identical on every rank.

    Scaled by 1/sqrt(dim) so each layer roughly preserves the activation scale.
    Unscaled U[0, 1) weights grow the activations by ~dim/2 per layer, which
    reaches inf by the third of four 512-wide layers in float16 -- and the
    resulting nan parameters would compare equal on every rank, so the
    cross-rank check at the end would pass without meaning anything.
    """
    return (
        [
            (np.random.randn(LOW_DIM, LOW_DIM) / np.sqrt(LOW_DIM)).astype(
                "float32"
            )
            for _ in range(N_LOW_LAYERS)
        ],
        [
            (np.random.randn(FP32_DIM, FP32_DIM) / np.sqrt(FP32_DIM)).astype(
                "float32"
            )
            for _ in range(N_FP32_LAYERS)
        ],
    )


def _short_dtype(dtype):
    return str(dtype).split(".")[-1]


def _build_model(weights):
    """Build the net and cast its low-precision branch.

    Returns the model and the dtype that was actually applied, which is read
    back off a parameter because ``amp.decorate`` is a silent no-op on a device
    that cannot do the requested dtype.
    """
    model = MixedDtypeNet(*weights)
    model = mix_precision_utils.MixPrecisionLayer(
        model, dtype=LOW_DTYPE_CANDIDATES[0]
    )
    probe = model._layers.low_layers[0].weight
    for dtype in LOW_DTYPE_CANDIDATES:
        paddle.amp.decorate(
            models=model,
            level="O2",
            dtype=dtype,
            excluded_layers=[model._layers.fp32_layers],
        )
        if _short_dtype(probe.dtype) == dtype:
            return model, dtype
    return model, _short_dtype(probe.dtype)


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
            np.random.random_sample((BATCH_SIZE, LOW_DIM)).astype("float32")
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
            # The forward must stay in range, especially in float16: an
            # overflowed loss produces nan parameters, and nan compares equal to
            # nan on every rank, which would make the cross-rank check below
            # pass without testing anything.
            self.assertTrue(
                np.isfinite(float(loss)),
                msg=f"step {idx} loss is {float(loss)}, not finite",
            )
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

    def test_mixed_dtype_partition(self):
        model, low_dtype = _build_model(_init_weights())
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
        # (active_ranks collapses to 1 and every param lands on rank 0), or when
        # amp.decorate silently declines to cast on an unsupported device.
        self.assertEqual(
            set(owners),
            {low_dtype, "float32"},
            msg=(
                "the two dtypes were not both present among 2D params; "
                f"got {dict(owners)} with low_dtype={low_dtype!r} -- neither "
                f"{' nor '.join(LOW_DTYPE_CANDIDATES)} could be applied to the "
                "low-precision branch on this device"
            ),
        )

        # The low-precision volume needs both ranks, the fp32 volume needs only
        # one. Under the previous whole-group packing fp32 spanned both ranks.
        self.assertEqual(
            owners[low_dtype],
            EXPECTED_LOW_OWNERS,
            msg=(
                f"{low_dtype} params should span both ranks, so the partition "
                "is not in the degenerate single-rank regime; got "
                f"{owners[low_dtype]}"
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
            self.assertTrue(
                np.isfinite(reference).all(),
                msg=(
                    f"param {name!r} is not finite after the optimizer steps; "
                    "nan/inf compares equal across ranks and would make the "
                    "comparison below vacuous"
                ),
            )
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
                f"{low_dtype} owners={sorted(owners[low_dtype])}, "
                f"fp32 owners={sorted(owners['float32'])}, "
                f"buffers per dtype={dict(buffers_per_dtype)}"
            )


if __name__ == "__main__":
    unittest.main()
