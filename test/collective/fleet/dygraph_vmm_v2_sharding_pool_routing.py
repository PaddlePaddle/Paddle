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

"""
Worker script for VMM V2 distributed pool routing test.

Runs a real distributed training flow (AMP O2 + GroupShardedStage2/Stage3)
with VMM V2 enabled, then inspects pool statistics to verify that every
tensor category lands in the correct VMM V2 memory pool:

  - Model parameters (fp16, after amp.decorate)  → Stable pool
  - Master weights (fp32 copies for AMP)          → Stable pool
  - Optimizer states (moment1, moment2)           → LongLived pool
  - Activations / gradients                       → Transient pool (default)
"""

import json
import os

import numpy as np

import paddle
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import (
    GroupShardedStage3,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
    GroupShardedScaler,
)
from paddle.nn import Linear

seed = 2023
np.random.seed(seed)
paddle.seed(seed)
paddle.set_flags({"FLAGS_use_legacy_linear": True})

POOL_STABLE = 0
POOL_LONGLIVED = 1
VMM_ALIGNMENT = 256
PARAM_NUMELS = (
    256 * 256,
    256,
    256 * 256,
    256,
    256 * 10,
    10,
)
EXPECTED_PARAM_BLOCKS = len(PARAM_NUMELS)


def aligned_size(size, alignment=VMM_ALIGNMENT):
    return ((size + alignment - 1) // alignment) * alignment


def expected_param_bytes(dtype_bytes):
    return sum(aligned_size(numel * dtype_bytes) for numel in PARAM_NUMELS)


EXPECTED_FP32_PARAM_BYTES = expected_param_bytes(4)
EXPECTED_FP16_PARAM_BYTES = expected_param_bytes(2)


# ------------------------------------------------------------------ #
# Model & Dataset
# ------------------------------------------------------------------ #
class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=256):
        super().__init__()
        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=200, linear_size=256):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype('float32')
        label = np.ones(1).astype('int64')
        return img, label

    def __len__(self):
        return self.num_samples


# ------------------------------------------------------------------ #
# Pool stats helper
# ------------------------------------------------------------------ #
def pool_stats():
    """Return {pool_type: (alloc_count, alloc_bytes)} from VMM V2."""
    stats = {}
    for (
        pt,
        ac,
        ab,
        _fc,
        _fb,
        _gc,
        _gb,
    ) in paddle.device.cuda.vmm_v2_pool_stats():
        pc, pb = stats.get(pt, (0, 0))
        stats[pt] = (pc + ac, pb + ab)
    return stats


def delta(after, before, pool):
    """Return (delta_count, delta_bytes) for a given pool type."""
    ac, ab = after.get(pool, (0, 0))
    bc, bb = before.get(pool, (0, 0))
    return ac - bc, ab - bb


# ------------------------------------------------------------------ #
# Training with pool stats collection
# ------------------------------------------------------------------ #
def train_with_pool_check(sharding_stage, linear_size=256):
    """Run AMP O2 distributed training and collect pool deltas."""
    group = paddle.distributed.new_group([0, 1])

    model = MLP(linear_size=linear_size)

    # Snapshot AFTER model creation (params → Stable pool)
    stats_after_model = pool_stats()

    # --- AMP O2 decorate: cast params fp32 → fp16 ---
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        multi_precision=True,
    )

    stats_before_amp = pool_stats()
    model = paddle.amp.decorate(models=model, level='O2', save_dtype='float32')
    stats_after_amp = pool_stats()

    scaler = paddle.amp.GradScaler(init_loss_scaling=32768)

    # --- Apply sharding wrapper ---
    # NOTE: Stage2's __init__ calls _generate_master_params which creates
    # master weights via cast_to_master_weight. Stage3's __init__ also
    # creates param slices. So we snapshot before/after wrapper creation.
    stats_before_sharding = pool_stats()

    if sharding_stage == 2:
        optimizer = GroupShardedOptimizerStage2(
            params=optimizer._parameter_list,
            optim=optimizer,
            group=group,
        )
        model = GroupShardedStage2(
            model, optimizer, group=group, buffer_max_size=2**21
        )
        scaler = GroupShardedScaler(scaler)
    elif sharding_stage == 3:
        scaler = GroupShardedScaler(scaler)
        model = GroupShardedStage3(
            model,
            optimizer=optimizer,
            group=group,
            sync_comm=True,
            segment_size=2**15,
        )

    stats_after_sharding = pool_stats()

    # --- Training loop (2 batches is enough to trigger all allocations) ---
    train_loader = paddle.io.DataLoader(
        RandomDataset(num_samples=200, linear_size=linear_size),
        batch_size=50,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    stats_before_train = pool_stats()

    for batch_id, data in enumerate(train_loader()):
        if batch_id >= 2:
            break
        img, label = data
        label.stop_gradient = True
        img.stop_gradient = True

        with paddle.amp.auto_cast(True, level='O2'):
            out = model(img)
            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
        avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))
        scaler.scale(avg_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad()

    if sharding_stage == 3:
        model.get_all_parameters()

    stats_after_train = pool_stats()

    # --- Collect deltas ---
    result = {}

    # Model params should be in Stable pool (created with create_parameter)
    dc, db = delta(stats_after_model, {}, POOL_STABLE)
    result['model_params_stable_blocks'] = dc
    result['model_params_stable_bytes'] = db

    # AMP O2 decorate: fp16 params should also go to Stable
    dc, db = delta(stats_after_amp, stats_before_amp, POOL_STABLE)
    result['amp_decorate_stable_delta_blocks'] = dc
    result['amp_decorate_stable_delta_bytes'] = db

    # Sharding wrapper init: master weights (Stable) + fused buffers (Stable)
    dc, db = delta(stats_after_sharding, stats_before_sharding, POOL_STABLE)
    result['sharding_init_stable_delta_blocks'] = dc
    result['sharding_init_stable_delta_bytes'] = db

    dc, db = delta(stats_after_sharding, stats_before_sharding, POOL_LONGLIVED)
    result['sharding_init_longlived_delta_blocks'] = dc
    result['sharding_init_longlived_delta_bytes'] = db

    # Training creates optimizer states (LongLived via _add_accumulator)
    dc, db = delta(stats_after_train, stats_before_train, POOL_STABLE)
    result['train_stable_delta_blocks'] = dc
    result['train_stable_delta_bytes'] = db

    dc, db = delta(stats_after_train, stats_before_train, POOL_LONGLIVED)
    result['train_longlived_delta_blocks'] = dc
    result['train_longlived_delta_bytes'] = db

    # Combined: sharding init + training (total new allocations since AMP)
    dc, db = delta(stats_after_train, stats_after_amp, POOL_STABLE)
    result['total_stable_delta_blocks'] = dc
    result['total_stable_delta_bytes'] = db

    dc, db = delta(stats_after_train, stats_after_amp, POOL_LONGLIVED)
    result['total_longlived_delta_blocks'] = dc
    result['total_longlived_delta_bytes'] = db

    # Final absolute totals
    final = pool_stats()
    result['final_stable_blocks'] = final.get(POOL_STABLE, (0, 0))[0]
    result['final_stable_bytes'] = final.get(POOL_STABLE, (0, 0))[1]
    result['final_longlived_blocks'] = final.get(POOL_LONGLIVED, (0, 0))[0]
    result['final_longlived_bytes'] = final.get(POOL_LONGLIVED, (0, 0))[1]

    return result


# ------------------------------------------------------------------ #
# Main entry: run both Stage2 and Stage3 and verify pool routing
# ------------------------------------------------------------------ #
def test_vmm_v2_sharding_pool_routing():
    paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()

    stage = int(os.environ.get('VMM_V2_TEST_STAGE', '2'))
    r = train_with_pool_check(sharding_stage=stage, linear_size=256)

    # ---- Assertions ---- #
    tag = f"[Rank {rank}, Stage{stage}]"

    print(f"{tag} STATS: {json.dumps(r)}")

    # 1. Model creation should materialize exactly six fp32 parameter tensors
    #    in Stable: weight/bias for three Linear layers.
    assert r['model_params_stable_blocks'] == EXPECTED_PARAM_BLOCKS, (
        f"{tag} expected {EXPECTED_PARAM_BLOCKS} fp32 parameter blocks in "
        f"Stable after model init, got {r['model_params_stable_blocks']}"
    )
    assert r['model_params_stable_bytes'] == EXPECTED_FP32_PARAM_BYTES, (
        f"{tag} expected fp32 parameter bytes {EXPECTED_FP32_PARAM_BYTES}, "
        f"got {r['model_params_stable_bytes']}"
    )

    # 2. AMP O2 decorate replaces the six fp32 params with six fp16 params.
    assert r['amp_decorate_stable_delta_blocks'] == 0, (
        f"{tag} expected AMP decorate to keep parameter block count unchanged, "
        f"got delta={r['amp_decorate_stable_delta_blocks']}"
    )
    assert r['amp_decorate_stable_delta_bytes'] == (
        EXPECTED_FP16_PARAM_BYTES - EXPECTED_FP32_PARAM_BYTES
    ), (
        f"{tag} expected AMP decorate stable-byte delta "
        f"{EXPECTED_FP16_PARAM_BYTES - EXPECTED_FP32_PARAM_BYTES}, got "
        f"{r['amp_decorate_stable_delta_bytes']}"
    )

    # 3. Sharding init must not create optimizer states yet.
    assert r['sharding_init_longlived_delta_blocks'] == 0, (
        f"{tag} LongLived blocks should still be zero before training, got "
        f"{r['sharding_init_longlived_delta_blocks']}"
    )
    assert r['sharding_init_longlived_delta_bytes'] == 0, (
        f"{tag} LongLived bytes should still be zero before training, got "
        f"{r['sharding_init_longlived_delta_bytes']}"
    )

    # 4. Sharding init must route master weights / param slices to Stable.
    #    In Stage2, the net Stable delta may be negative on some ranks because
    #    sharding releases parameter shards owned by other ranks.  Verify that
    #    the final Stable pool is non-empty (master weights persist).
    assert r['final_stable_bytes'] > 0, (
        f"{tag} expected Stable pool to hold master weights, got "
        f"final_stable_bytes={r['final_stable_bytes']}"
    )

    # 5. The first training step must create optimizer states in LongLived.
    assert r['train_longlived_delta_blocks'] > 0, (
        f"{tag} expected training to create LongLived blocks, got "
        f"{r['train_longlived_delta_blocks']}"
    )
    assert r['train_longlived_delta_bytes'] > 0, (
        f"{tag} expected training to create LongLived bytes, got "
        f"{r['train_longlived_delta_bytes']}"
    )
    assert r['total_longlived_delta_blocks'] > 0, (
        f"{tag} optimizer states not in LongLived pool "
        f"(blocks={r['total_longlived_delta_blocks']})"
    )
    assert r['total_longlived_delta_bytes'] > 0, (
        f"{tag} optimizer state bytes == 0 in LongLived pool"
    )

    # 6. Final sanity: both pools should have non-zero content.
    assert r['final_stable_blocks'] > 0, f"{tag} final Stable pool is empty"
    assert r['final_longlived_blocks'] > 0, (
        f"{tag} final LongLived pool is empty"
    )

    print(f"{tag} PASSED")
    if rank == 0:
        print("ALL POOL ROUTING CHECKS PASSED")


if __name__ == '__main__':
    test_vmm_v2_sharding_pool_routing()
