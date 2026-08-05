# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import load_state_dict, save_state_dict
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight

# Keep 240 elements while reducing the logical tensor to three dimensions.
GLOBAL_SHAPE = (2, 10, 12)
NUMEL = int(np.prod(GLOBAL_SHAPE))

# The source and destination ranges deliberately cross several flattened
# dimension boundaries.  The final case also isolates the last element.
CASES = (
    ("cross_axis", (0, 25), (25, NUMEL), (0, 121), (121, NUMEL)),
    ("exact_axis_boundary", (0, 120), (120, NUMEL), (0, 60), (60, NUMEL)),
    ("last_element", (0, NUMEL - 1), (NUMEL - 1, NUMEL), (0, 1), (1, NUMEL)),
)


def global_tensor(case_index):
    return paddle.arange(NUMEL, dtype="int64").reshape(GLOBAL_SHAPE) + (
        case_index * NUMEL
    )


def make_weight(key, tensor, flat_range):
    return ShardedWeight(
        key=key,
        local_tensor=tensor,
        local_shape=GLOBAL_SHAPE,
        global_shape=GLOBAL_SHAPE,
        global_offset=(0,) * len(GLOBAL_SHAPE),
        is_flattened=True,
        flattened_range=slice(*flat_range),
    )


def save_case(ckpt_path):
    rank = dist.get_rank()
    state_dict = {}
    for case_index, (name, rank0_start, rank1_start, _, _) in enumerate(CASES):
        start, end = rank0_start if rank == 0 else rank1_start
        flat = paddle.flatten(global_tensor(case_index))
        state_dict[name] = make_weight(name, flat[start:end], (start, end))
    save_state_dict(state_dict, ckpt_path)
    dist.barrier()


def load_case(ckpt_path, comm_method):
    rank = dist.get_rank()
    state_dict = {}
    expected = {}
    for case_index, (name, _, _, rank0_end, rank1_end) in enumerate(CASES):
        start, end = rank0_end if rank == 0 else rank1_end
        local_tensor = paddle.zeros([end - start], dtype="int64")
        state_dict[name] = make_weight(name, local_tensor, (start, end))
        expected[name] = paddle.flatten(global_tensor(case_index))[start:end]

    load_state_dict(state_dict, ckpt_path, comm_method=comm_method)
    dist.barrier()
    for name, weight in state_dict.items():
        np.testing.assert_array_equal(
            weight.local_tensor.numpy(), expected[name].numpy(), err_msg=name
        )


if __name__ == "__main__":
    mode = os.environ["high_dim_mode"]
    ckpt_path = os.environ["high_dim_ckpt_path"]
    if mode == "save":
        save_case(ckpt_path)
    elif mode == "load":
        load_case(ckpt_path, os.environ["high_dim_comm_method"])
    else:
        raise ValueError(f"Unsupported high_dim_mode: {mode}")
