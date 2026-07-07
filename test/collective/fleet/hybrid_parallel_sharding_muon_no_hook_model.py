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
# Real single-machine 2-card Muon test for the sharding-stage1 "no-hook shared
# color" mechanism in MuonShardingOptimizer.
#
# A 3-layer fully connected network is trained under pure sharding (degree=2).
# The middle layer's params are colored ``dense_weight_no_hook`` and its weight
# is deliberately reused in a second (detached) autograd sub-graph, emulating
# the MTP weight-sharing situation: a per-param comm-overlap backward hook would
# fire once per sub-graph and corrupt FusedCommBuffer.add_grad bookkeeping, so
# these params must be reduced *synchronously* (sync-only), never via the hook.
#
# The test asserts that turning comm_overlap ON (where non-shared params reduce
# via backward hooks and the no-hook layer takes the sync-only path) produces
# bit-for-bit identical parameters to the comm_overlap OFF baseline (where every
# param reduces synchronously). Overlap only changes *when* the NCCL reduce is
# launched, not the math, so the two runs must agree to the last bit.

import os
import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.optimizer.muon import (
    MuonParamInfo,
    _default_should_use_muon,
)

# Multi-precision (fp32 master weights) is required by clear_param_storage and
# matches the production EchoMTP setup.
os.environ["MUON_DEBUG"] = "0"

SHARDING_DEGREE = 2
HIDDEN = 64
BATCH_SIZE = 16  # divisible by SHARDING_DEGREE
STEPS = 4
NO_HOOK_COLOR = "dense_weight_no_hook"


class ThreeLayerFC(paddle.nn.Layer):
    """A 3-layer fully connected network.

    fc1 / fc3 are ordinary layers (comm-overlap hooks apply to them).
    fc2 is the "shared" layer: its params are colored ``dense_weight_no_hook``
    and its weight is additionally reused in a detached auxiliary graph, so it
    receives gradients from two sub-graphs — the exact reason it must be
    reduced synchronously rather than through a per-param overlap hook.
    """

    def __init__(self, np_w1, np_w2, np_w3):
        super().__init__()
        self.fc1 = paddle.nn.Linear(
            HIDDEN,
            HIDDEN,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_w1)
            ),
        )
        self.fc2 = paddle.nn.Linear(
            HIDDEN,
            HIDDEN,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_w2)
            ),
        )
        self.fc3 = paddle.nn.Linear(
            HIDDEN,
            HIDDEN,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_w3)
            ),
        )

    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        out = self.fc3(h2)
        # Detached second use of fc2's weight: fc2.weight now backprops from two
        # sub-graphs (main path + this aux path), emulating MTP weight sharing.
        aux = paddle.matmul(h1.detach(), self.fc2.weight)
        return out.mean() + aux.mean()


def _init_weights():
    """Create identical initial weights shared by both runs."""
    return (
        np.random.random_sample((HIDDEN, HIDDEN)).astype("float32"),
        np.random.random_sample((HIDDEN, HIDDEN)).astype("float32"),
        np.random.random_sample((HIDDEN, HIDDEN)).astype("float32"),
    )


def _build_model(weights):
    model = ThreeLayerFC(*weights)
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")
    return model


def _color_no_hook_layer(model):
    """Color fc2's params ``dense_weight_no_hook`` on the sharding group."""
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    for name, p in model.named_parameters():
        if "fc2" in name:
            p.color = {"color": NO_HOOK_COLOR, "group": sharding_group}


def _build_optimizer(model):
    muon_param_info_map = {}
    for name, param in model.named_parameters():
        use_muon = _default_should_use_muon(name, param.shape, [])
        muon_param_info_map[param.name] = MuonParamInfo(
            use_muon=use_muon, split_concat_func=None
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


class TestMuonNoHookOverlapVsBaseline(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": SHARDING_DEGREE,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.use_muon_sharding = True
        sharding_configs = self.strategy.hybrid_configs["sharding_configs"]
        sharding_configs.accumulate_steps = 1
        # Positive buffer size => fused 2D comm buffers exist, so the no-hook 2D
        # params form a real sync-only FusedCommBuffer under comm_overlap.
        sharding_configs.comm_buffer_size_MB = 256

        fleet.init(is_collective=True, strategy=self.strategy)

        # Identical full dataset on every rank; each rank slices its shard.
        self.data = [
            np.random.random_sample((BATCH_SIZE, HIDDEN)).astype("float32")
            for _ in range(STEPS)
        ]

    def _set_comm_overlap(self, enable):
        strategy = fleet.fleet._user_defined_strategy
        strategy.hybrid_configs["sharding_configs"].comm_overlap = enable

    def _train_batch(self, batch, model, optimizer):
        with paddle.amp.auto_cast(dtype="bfloat16"):
            loss = model(batch)
        loss.backward()
        inner_opt = getattr(optimizer, "_inner_opt", optimizer)
        if isinstance(inner_opt, MuonShardingOptimizer):
            # Clear the fused bf16 storage of the no-hook color between steps,
            # mirroring the production SonicMoE callback.
            optimizer.clear_param_storage(NO_HOOK_COLOR)
        optimizer.step()
        optimizer.clear_grad()
        return loss

    def _run(self, weights, comm_overlap):
        """Train the 2-card model for STEPS and return final params as numpy."""
        self._set_comm_overlap(comm_overlap)

        model = _build_model(weights)
        _color_no_hook_layer(model)
        optimizer = _build_optimizer(model)
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        # Sanity-check that the run really exercises the intended paths, so the
        # bit-for-bit comparison cannot pass trivially (e.g. if the overlap flag
        # silently failed to apply).
        inner_opt = optimizer._inner_opt
        assert isinstance(inner_opt, MuonShardingOptimizer)
        assert len(inner_opt._sync_only_buffers) > 0, (
            "fc2 (dense_weight_no_hook) was not routed to a sync-only buffer"
        )
        if comm_overlap:
            assert inner_opt.comm_overlap is True
            assert len(inner_opt._overlap_comm_buffers) > 0, (
                "no overlap buffers registered for the non-shared layers"
            )
        else:
            assert inner_opt.comm_overlap is False

        hcg = fleet.get_hybrid_communicate_group()
        sharding_rank = hcg.get_sharding_parallel_rank()
        local_bs = BATCH_SIZE // SHARDING_DEGREE

        for idx in range(STEPS):
            start = sharding_rank * local_bs
            batch = paddle.to_tensor(self.data[idx][start : start + local_bs])
            self._train_batch(batch, model, optimizer)

        return {
            name: p.cast("float32").numpy()
            for name, p in model.named_parameters()
        }

    def test_no_hook_overlap_matches_baseline_bitwise(self):
        weights = _init_weights()

        # Baseline: comm_overlap OFF (every param reduced synchronously).
        baseline = self._run(weights, comm_overlap=False)
        # Overlap: comm_overlap ON (non-shared params via hooks, no-hook layer
        # via the sync-only path).
        overlap = self._run(weights, comm_overlap=True)

        assert set(baseline) == set(overlap)
        for name in baseline:
            np.testing.assert_array_equal(
                overlap[name],
                baseline[name],
                err_msg=(
                    f"Param {name!r} differs between comm_overlap and baseline; "
                    f"the no-hook sync-only path is not bit-for-bit identical."
                ),
            )
        if paddle.distributed.get_rank() == 0:
            print(
                "[PASS] comm_overlap (no-hook sync-only) == baseline "
                f"bit-for-bit across {len(baseline)} params"
            )


if __name__ == "__main__":
    unittest.main()
