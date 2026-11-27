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

"""Sharding stage 1 on two GPUs of one node (sharding_degree=2): training
runs of a small MLP with a sharding option on, compared with the same run
with the option off. Both runs start from the same weights and see the same
data; every run takes STEPS optimizer steps of ACCUMULATE_STEPS micro-batches.

The case is chosen by CASE_ENV (test_sharding_gin_overlap.py):

  PART_A_CASE (no gin_overlap needed)
    - DygraphShardingOptimizerV2 + AdamW, param_allgather_fuse_mb > 0 vs 0:
      the fused parameter all_gather must give bitwise equal parameters
      after every step (and runs several fused batches, one of them with
      more than one bucket).
    - MuonShardingOptimizer, muon_owner_partition_mb > 0 vs 0 (the 2D
      parameters move from two owners to one) and param_allgather_fuse_mb
      > 0 vs 0 (one fused batch: the biases and the head are the only
      reduce_scatter bucket): bitwise equal parameters after every step.
  GIN_CASES (Paddle built with WITH_GIN_OVERLAP)
    - gin_overlap=True with the default configuration (auto-sized recv
      pools, MB options 0) vs gin_overlap off, for AdamW and Muon, and AdamW
      with a dense recv pool at its lower bound, so that its slots are reused
      within a step and a bucket must wait for the credit of the slot's
      previous holder. After reduce_gradients the reduced gradients of this
      rank (its shard of each 1D bucket, the whole 2D bucket on its owner)
      must be bitwise equal to a reference computed here in the summation
      order of gin_overlap, and so must those of the run without gin; the
      parameters must be bitwise equal after every step.

Every rank asserts on its own, so a failure anywhere fails the launcher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import (
    mix_precision_utils,
    tensor_fusion_helper,
)
from paddle.distributed.fleet.utils.tensor_fusion_helper import HOOK_ACTION
from paddle.optimizer.muon import MuonParamInfo

CASE_ENV = "SHARDING_GIN_OVERLAP_CASE"
PART_A_CASE = "fused_all_gather_and_owner_partition"
GIN_ADAMW_CASE = "gin_adamw"
GIN_ADAMW_RECV_POOL_WRAP_CASE = "gin_adamw_recv_pool_wrap"
GIN_MUON_CASE = "gin_muon"
GIN_CASES = (GIN_ADAMW_CASE, GIN_ADAMW_RECV_POOL_WRAP_CASE, GIN_MUON_CASE)

SHARDING_DEGREE = 2
STEPS = 3
ACCUMULATE_STEPS = 2
ROWS_PER_RANK = 2
# Gradient buckets of at most 1 MB (the option is in whole MB): the fp32
# model below has about 8 MB of parameters, so there are several buckets.
COMM_BUFFER_SIZE_MB = 1
# Fused parameter all_gather of at most 2 MB per batch: batches of more than
# one bucket, and more than one batch per step.
PARAM_ALLGATHER_FUSE_MB = 2.0
# Muon owner partition this large puts every 2D parameter on one owner,
# where comm_buffer_size_MB (1 MB) spreads them over both ranks.
MUON_OWNER_PARTITION_MB = 64.0
# Hidden sizes of the MLP: Linear(512, 768), Linear(768, 512),
# Linear(512, 512), then the head Linear(512, 256) without bias. The 2D
# weights of the hidden layers have different shapes, so Muon
# orthogonalizes each of them on its own whatever the owner partition.
HIDDEN_SIZES = (512, 768, 512, 512)
HEAD_SIZE = 256


@dataclass
class RunConfig:
    """One training run. dense_recv_pool_gb None keeps the configured
    gin_overlap_dense_recv_pool_gb (by default auto-sized)."""

    name: str
    optimizer: str  # "adamw" or "muon"
    gin_overlap: bool = False
    param_allgather_fuse_mb: float = 0.0
    muon_owner_partition_mb: float = 0.0
    dense_recv_pool_gb: float | None = None


@dataclass
class RunRecord:
    """What a run leaves behind, per step: the parameters after the step,
    this rank's reduced gradients ({bucket index: values}), and (gin off
    only) every rank's gradients before the reduce ([bucket][rank])."""

    config: RunConfig
    sharding_optimizer: object = None
    params: list = field(default_factory=list)
    reduced_grads: list = field(default_factory=list)
    local_grads_by_rank: list = field(default_factory=list)
    all_gather_batches: list = field(default_factory=list)
    owners_2d: list = field(default_factory=list)


# Number of buckets of each fused all_gather call in the current step,
# recorded around tensor_fusion_helper._all_gather_chunk.
_current_all_gather_batches = []
_all_gather_chunk = tensor_fusion_helper._all_gather_chunk


def _all_gather_chunk_and_record(chunk, *args, **kwargs):
    _current_all_gather_batches.append(len(chunk))
    return _all_gather_chunk(chunk, *args, **kwargs)


tensor_fusion_helper._all_gather_chunk = _all_gather_chunk_and_record


class SmallMLP(paddle.nn.Layer):
    def __init__(self, initial_weights):
        super().__init__()
        self.hidden_layers = paddle.nn.LayerList()
        for weight, bias in initial_weights["hidden"]:
            self.hidden_layers.append(
                paddle.nn.Linear(
                    weight.shape[0],
                    weight.shape[1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Assign(weight)
                    ),
                    bias_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Assign(bias)
                    ),
                )
            )
        head_weight = initial_weights["head"]
        self.head = paddle.nn.Linear(
            head_weight.shape[0],
            head_weight.shape[1],
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(head_weight)
            ),
            bias_attr=False,
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = paddle.tanh(layer(x))
        return self.head(x)


def _initial_weights():
    random_state = np.random.RandomState(2026)
    hidden = []
    for fan_in, fan_out in zip(HIDDEN_SIZES[:-1], HIDDEN_SIZES[1:]):
        weight = random_state.normal(0, fan_in**-0.5, (fan_in, fan_out))
        bias = random_state.normal(0, 0.1, (fan_out,))
        hidden.append((weight.astype("float32"), bias.astype("float32")))
    head = random_state.normal(
        0, HIDDEN_SIZES[-1] ** -0.5, (HIDDEN_SIZES[-1], HEAD_SIZE)
    )
    return {"hidden": hidden, "head": head.astype("float32")}


def _micro_batches(rank):
    """[step][micro-batch] inputs of this rank: its rows of the global batch."""
    random_state = np.random.RandomState(7)
    global_rows = SHARDING_DEGREE * ROWS_PER_RANK
    rows = slice(rank * ROWS_PER_RANK, (rank + 1) * ROWS_PER_RANK)
    return [
        [
            random_state.normal(0, 1, (global_rows, HIDDEN_SIZES[0])).astype(
                "float32"
            )[rows]
            for _ in range(ACCUMULATE_STEPS)
        ]
        for _ in range(STEPS)
    ]


def _init_fleet():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": SHARDING_DEGREE,
    }
    sharding_configs = strategy.hybrid_configs["sharding_configs"]
    sharding_configs.split_param = True
    sharding_configs.accumulate_steps = ACCUMULATE_STEPS
    sharding_configs.comm_buffer_size_MB = COMM_BUFFER_SIZE_MB
    fleet.init(is_collective=True, strategy=strategy)


def _apply_run_config(config):
    """The sharding optimizers read the options from the fleet strategy when
    they are built, so each run sets them there first."""
    strategy = fleet.fleet._user_defined_strategy
    strategy.use_muon_sharding = config.optimizer == "muon"
    sharding_configs = strategy.hybrid_configs["sharding_configs"]
    sharding_configs.gin_overlap = config.gin_overlap
    sharding_configs.param_allgather_fuse_mb = config.param_allgather_fuse_mb
    sharding_configs.muon_owner_partition_mb = config.muon_owner_partition_mb
    if config.dense_recv_pool_gb is not None:
        sharding_configs.gin_overlap_dense_recv_pool_gb = (
            config.dense_recv_pool_gb
        )


def _build(config, initial_weights):
    _apply_run_config(config)
    model = SmallMLP(initial_weights)
    # fp32 parameters with an fp32 main_grad, which gin_overlap requires;
    # dtype only names the low precision type of the hook.
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    if config.optimizer == "adamw":
        optimizer = mix_precision_utils.MixPrecisionOptimizer(
            paddle.optimizer.AdamW(
                learning_rate=1e-3,
                parameters=model.parameters(),
                weight_decay=0.01,
            )
        )
    else:
        # Muon for the 2D weights of the hidden layers, AdamW for the
        # biases and the head.
        muon_param_info_map = {
            param.name: MuonParamInfo(
                use_muon=len(param.shape) == 2 and "head" not in name
            )
            for name, param in model.named_parameters()
        }
        optimizer = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=model.parameters(),
            weight_decay=0.01,
            muon_param_info_map=muon_param_info_map,
        )
    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)
    return model, optimizer


def _sharding_optimizer(optimizer):
    """DygraphShardingOptimizerV2 or MuonShardingOptimizer under the fleet
    wrapper."""
    return optimizer._inner_opt


def _buckets(sharding_optimizer):
    """Gradient buckets in a fixed order: the Muon 2D buckets (reduced to
    their owner), then the reduce_scatter buckets."""
    return list(getattr(sharding_optimizer, "comm_buffer_2d", [])) + list(
        sharding_optimizer._comm_buffer_list
    )


def _bucket_numel(bucket):
    """Elements of the bucket's gradients; a gin_overlap Muon 2D bucket's
    grad_storage is padded beyond them (with zeros)."""
    owner_bucket = getattr(bucket, "_gin_owner_bucket", None)
    if owner_bucket is not None:
        return owner_bucket[0]
    return bucket.grad_storage._numel()


def _reduced_range(bucket, rank):
    """(begin, end) of the elements of a bucket that are reduced on this
    rank, or None: the rank's shard of a reduce_scatter bucket, the whole
    bucket on the owner of a 2D bucket."""
    numel = _bucket_numel(bucket)
    if bucket._act == HOOK_ACTION.REDUCE_SCATTER:
        shard_numel = numel // SHARDING_DEGREE
        return rank * shard_numel, (rank + 1) * shard_numel
    if bucket._dst == rank:
        return 0, numel
    return None


def _all_ranks_values(values):
    """[rank 0 values, rank 1 values] of a float32 numpy array."""
    gathered = []
    dist.all_gather(gathered, paddle.to_tensor(values))
    return [tensor.numpy() for tensor in gathered]


def _record_reduce(record, sharding_optimizer, rank):
    """Wraps reduce_gradients: records each bucket's gradients of every rank
    before the reduce (gin off only) and this rank's reduced gradients
    after it."""
    reduce_gradients = sharding_optimizer.reduce_gradients

    def reduce_gradients_and_record(*args, **kwargs):
        buckets = _buckets(sharding_optimizer)
        if not record.config.gin_overlap:
            record.local_grads_by_rank.append(
                [
                    _all_ranks_values(
                        bucket.grad_storage._slice(
                            0, _bucket_numel(bucket)
                        ).numpy()
                    )
                    for bucket in buckets
                ]
            )
        result = reduce_gradients(*args, **kwargs)
        reduced = {}
        for index, bucket in enumerate(buckets):
            reduced_range = _reduced_range(bucket, rank)
            if reduced_range is not None:
                reduced[index] = bucket.grad_storage._slice(
                    *reduced_range
                ).numpy()
        record.reduced_grads.append(reduced)
        return result

    sharding_optimizer.reduce_gradients = reduce_gradients_and_record


def _train(config, initial_weights, micro_batches):
    rank = dist.get_rank()
    record = RunRecord(config)
    model, optimizer = _build(config, initial_weights)
    record.sharding_optimizer = _sharding_optimizer(optimizer)
    _record_reduce(record, record.sharding_optimizer, rank)
    if config.optimizer == "muon":
        record.owners_2d = _owners_of_2d_params(
            record.sharding_optimizer, model
        )
    for step_inputs in micro_batches:
        for inputs in step_inputs:
            loss = model(paddle.to_tensor(inputs)).square().mean()
            loss.backward()
        _current_all_gather_batches.clear()
        optimizer.step()
        optimizer.clear_grad()
        record.all_gather_batches.append(list(_current_all_gather_batches))
        record.params.append([param.numpy() for param in model.parameters()])
    return record


def _assert_bitwise_equal(actual, expected, what):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, (
        what,
        actual.dtype,
        expected.dtype,
        actual.shape,
        expected.shape,
    )
    if actual.tobytes() != expected.tobytes():
        differing = np.flatnonzero(
            actual.view(np.uint32) != expected.view(np.uint32)
        )
        first = differing[0]
        raise AssertionError(
            f"rank {dist.get_rank()}: {what}: {differing.size} of "
            f"{actual.size} elements differ, first at {first}: "
            f"{actual.flat[first]!r} vs {expected.flat[first]!r} (max abs "
            f"diff {np.max(np.abs(actual - expected))})"
        )


def _assert_same_params(run, baseline):
    for step, (params, baseline_params) in enumerate(
        zip(run.params, baseline.params)
    ):
        for index, (param, baseline_param) in enumerate(
            zip(params, baseline_params)
        ):
            _assert_bitwise_equal(
                param,
                baseline_param,
                f"{run.config.name} vs {baseline.config.name}: parameter "
                f"{index} after step {step}",
            )


def _gin_order_reference(local_grads_by_rank, rank):
    """The reduced gradients as gin_overlap computes them on `rank` (sum.h):
    the intra-node sum from 0 over the sources in ring order starting at
    rank + 1 (here the peer, then this rank), then the final sum over the
    single node from 0, each term __fadd_rn(sum, __fmul_rn(value, 1/2)).
    numpy float32 additions and multiplications round to nearest like
    __fadd_rn / __fmul_rn. With two sources the order cannot change a bit
    (IEEE addition is commutative), so the same formula holds for the
    shards of a 2D bucket that the other rank reduced before the gather."""
    zero = np.float32(0)
    peer = (rank + 1) % SHARDING_DEGREE
    node_sum = (zero + local_grads_by_rank[peer]) + local_grads_by_rank[rank]
    return zero + node_sum * np.float32(1.0 / SHARDING_DEGREE)


def _assert_reduced_grads_match_reference(run, baseline):
    """The reduced gradients of `run` against the gin-order reference built
    from the gradients of every rank recorded by `baseline` (the run without
    gin_overlap, whose local gradients are the same as long as the
    parameters are)."""
    rank = dist.get_rank()
    buckets = _buckets(baseline.sharding_optimizer)
    for step, (reduced, local_grads) in enumerate(
        zip(run.reduced_grads, baseline.local_grads_by_rank)
    ):
        assert sorted(reduced) == sorted(
            index
            for index, bucket in enumerate(buckets)
            if _reduced_range(bucket, rank) is not None
        ), (run.config.name, step, sorted(reduced))
        for index, values in reduced.items():
            begin, end = _reduced_range(buckets[index], rank)
            reference = _gin_order_reference(
                [grads[begin:end] for grads in local_grads[index]], rank
            )
            _assert_bitwise_equal(
                values,
                reference,
                f"{run.config.name}: reduced gradients of bucket {index} in "
                f"step {step} vs the gin-order reference",
            )


def _assert_fused_all_gather_batches(run, expect_several_batches):
    """param_allgather_fuse_mb > 0: every reduce_scatter bucket goes
    through the fused all_gather in every step, and with
    expect_several_batches in more than one batch, at least one of them with
    several buckets; 0: no fused all_gather."""
    num_buckets = sum(
        bucket._act == HOOK_ACTION.REDUCE_SCATTER
        for bucket in _buckets(run.sharding_optimizer)
    )
    for step, batches in enumerate(run.all_gather_batches):
        if run.config.param_allgather_fuse_mb <= 0:
            assert batches == [], (run.config.name, step, batches)
            continue
        assert sum(batches) == num_buckets, (
            run.config.name,
            step,
            batches,
            num_buckets,
        )
        if expect_several_batches:
            assert len(batches) > 1 and max(batches) > 1, (
                run.config.name,
                step,
                batches,
            )


def _owners_of_2d_params(sharding_optimizer, model):
    """Owner (sharding rank) of each parameter of the model kept whole by
    MuonShardingOptimizer, None for the others, in parameter order."""
    owners = {
        param_name: owner
        for owners_of_color in (
            sharding_optimizer._param2rank_2d_by_color.values()
        )
        for param_name, owner in owners_of_color.items()
    }
    return [owners.get(param.name) for param in model.parameters()]


def run_fused_all_gather_and_owner_partition(initial_weights, micro_batches):
    """Part A: the sharding stage1 options of gin_overlap's PR that also work
    without gin_overlap, each against the same run with the option off."""
    adamw_baseline = _train(
        RunConfig("adamw", "adamw"), initial_weights, micro_batches
    )
    adamw_fused = _train(
        RunConfig(
            "adamw fused all_gather",
            "adamw",
            param_allgather_fuse_mb=PARAM_ALLGATHER_FUSE_MB,
        ),
        initial_weights,
        micro_batches,
    )
    _assert_fused_all_gather_batches(adamw_baseline, False)
    _assert_fused_all_gather_batches(adamw_fused, True)
    _assert_same_params(adamw_fused, adamw_baseline)

    muon_baseline = _train(
        RunConfig("muon", "muon"), initial_weights, micro_batches
    )
    muon_owner_partition = _train(
        RunConfig(
            "muon owner partition",
            "muon",
            muon_owner_partition_mb=MUON_OWNER_PARTITION_MB,
        ),
        initial_weights,
        micro_batches,
    )
    muon_fused = _train(
        RunConfig(
            "muon fused all_gather",
            "muon",
            param_allgather_fuse_mb=PARAM_ALLGATHER_FUSE_MB,
        ),
        initial_weights,
        micro_batches,
    )
    # comm_buffer_size_MB spreads the 2D weights over both ranks, the large
    # owner partition puts them all on one rank.
    baseline_owners = {o for o in muon_baseline.owners_2d if o is not None}
    partition_owners = {
        o for o in muon_owner_partition.owners_2d if o is not None
    }
    assert baseline_owners == {0, 1}, muon_baseline.owners_2d
    assert len(partition_owners) == 1, muon_owner_partition.owners_2d
    # Muon keeps only the biases and the head in reduce_scatter buckets
    # (one bucket), so its fused all_gather has a single batch.
    _assert_fused_all_gather_batches(muon_baseline, False)
    _assert_fused_all_gather_batches(muon_fused, False)
    _assert_same_params(muon_owner_partition, muon_baseline)
    _assert_same_params(muon_fused, muon_baseline)


def _dense_recv_pool_lower_bound(sharding_optimizer):
    """Smallest dense recv pool gin_overlap accepts for these buckets on one
    node: the largest intra-node slot (the peer's shard, 256 B aligned),
    rounded up to 4 KB (grad_comm_scheduler._Layout._recv_pool). The
    inter-node segment is empty with one node."""

    def round_up(nbytes, alignment):
        return (nbytes + alignment - 1) // alignment * alignment

    largest_slot = max(
        round_up(_bucket_numel(bucket) // SHARDING_DEGREE * 4, 256)
        for bucket in _buckets(sharding_optimizer)
    )
    return round_up(largest_slot, 4096)


def _assert_gin_run(run, expect_slot_reuse):
    """The run went through gin_overlap: every bucket is in its scheduler,
    which completed every step; the intra-node recv segment reuses slots
    within a step only when the pool is that small."""
    sharding_optimizer = run.sharding_optimizer
    scheduler = sharding_optimizer._gin_scheduler
    assert scheduler is not None, run.config.name
    assert scheduler.completed_steps() == STEPS, scheduler.completed_steps()
    buckets = _buckets(sharding_optimizer)
    assert {id(b) for b in sharding_optimizer._gin_buckets} == {
        id(b) for b in buckets
    }
    layout = sharding_optimizer._gin_scheduler_state[0]
    num_dense = len(layout.dense_order)
    intra_distance = layout.distance[0]
    assert len(intra_distance) == num_dense > 1, intra_distance
    if expect_slot_reuse:
        assert not layout.auto_sized[0], layout.auto_sized
        assert min(intra_distance.values()) < num_dense, intra_distance
    else:
        assert layout.auto_sized[0], layout.auto_sized
        assert set(intra_distance.values()) == {num_dense}, intra_distance


def run_gin(case, initial_weights, micro_batches):
    """Part B: gin_overlap against the same run without it. One case per
    process: a process builds one GinGradCommScheduler."""
    optimizer = "muon" if case == GIN_MUON_CASE else "adamw"
    baseline = _train(
        RunConfig(optimizer, optimizer), initial_weights, micro_batches
    )
    gin_config = RunConfig(f"{optimizer} gin_overlap", optimizer, True)
    expect_slot_reuse = case == GIN_ADAMW_RECV_POOL_WRAP_CASE
    if expect_slot_reuse:
        gin_config.name += " small recv pool"
        gin_config.dense_recv_pool_gb = (
            _dense_recv_pool_lower_bound(baseline.sharding_optimizer) / 2**30
        )
    gin_run = _train(gin_config, initial_weights, micro_batches)
    _assert_gin_run(gin_run, expect_slot_reuse)
    _assert_reduced_grads_match_reference(baseline, baseline)
    _assert_reduced_grads_match_reference(gin_run, baseline)
    _assert_same_params(gin_run, baseline)


def main():
    case = os.environ[CASE_ENV]
    _init_fleet()
    rank = dist.get_rank()
    initial_weights = _initial_weights()
    micro_batches = _micro_batches(rank)
    if case == PART_A_CASE:
        run_fused_all_gather_and_owner_partition(initial_weights, micro_batches)
    elif case in GIN_CASES:
        run_gin(case, initial_weights, micro_batches)
    else:
        raise ValueError(f"unknown {CASE_ENV}={case}")
    dist.barrier()
    print(f"rank {rank}: {case} passed")


if __name__ == "__main__":
    main()
