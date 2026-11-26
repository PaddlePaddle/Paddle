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
"""Save and load across ranks with an AoA ``reshape`` (run on 2 cards).

The checkpoint holds a grouped-GEMM expert weight in the 2-D layout
``[E * H, 2 * I]`` sharded by rows (expert parallel); the model holds it in
another layout and sharding, chosen by ``reshape_case``:

* ``ep_to_tp``: ``[E, H, 2, I]`` sharded along ``I`` (tensor parallel), the
  weight a parameter and its ``moment1`` on CUDA pinned memory, as an
  offloaded optimizer keeps it;
* ``chain``: reshape, permute to gate/up major and reshape to ``[E * 2 * H, I]``,
  sharded by rows;
* ``reverse``: the checkpoint in ``[E, H, 2, I]`` sharded along ``I`` read back
  into the 2-D layout by two reshapes run with ``aoa_config_reverse``.

Every loaded element is compared with the numpy reference.
"""

import math
import os
import shutil
import tempfile

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight

E, H, I = 4, 6, 8
FLAT = (E * H, 2 * I)
CANON = (E, H, 2, I)
GATE_UP_MAJOR = (E * 2 * H, I)
MOMENT_OFFSET = 1000.0

CASES = {
    "ep_to_tp": {
        "ckpt": (FLAT, 0),
        "model": (CANON, 3),
        "statements": [f"w -> w, reshape = '{list(CANON)}'"],
        "reference": lambda a: a.reshape(CANON),
        "pinned_moment": True,
    },
    "chain": {
        "ckpt": (FLAT, 0),
        "model": (GATE_UP_MAJOR, 0),
        "statements": [
            f"w -> t, reshape = '{list(CANON)}'",
            "t -> u, permute = '[0, 2, 1, 3]'",
            f"u -> w, reshape = '{list(GATE_UP_MAJOR)}'",
        ],
        "reference": lambda a: (
            a.reshape(CANON).transpose(0, 2, 1, 3).reshape(GATE_UP_MAJOR)
        ),
    },
    "reverse": {
        "ckpt": (CANON, 3),
        "model": (FLAT, 0),
        "statements": [
            f"w -> t, reshape = '{[E, H * 2 * I]}'",
            f"t -> w, reshape = '{list(CANON)}'",
        ],
        "reference": lambda a: a.reshape(FLAT),
        "reverse": True,
    },
}


def _shard(shape, axis, rank, world):
    local = list(shape)
    local[axis] //= world
    offset = [0] * len(shape)
    offset[axis] = rank * local[axis]
    index = tuple(slice(o, o + n) for o, n in zip(offset, local))
    return local, offset, index


def _full(shape, base=0.0):
    return np.arange(math.prod(shape), dtype="float32").reshape(shape) + base


def run_test():
    case = CASES[os.environ["reshape_case"]]
    dist.init_parallel_env()
    rank, world = dist.get_rank(), dist.get_world_size()
    paths = [tempfile.mkdtemp() if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    path = paths[0]

    shape, axis = case["ckpt"]
    local, offset, index = _shard(shape, axis, rank, world)
    saved = {}
    for key, base in (("w", 0.0), ("w.moment1_0", MOMENT_OFFSET)):
        tensor = paddle.to_tensor(_full(shape, base)[index])
        saved[key] = ShardedWeight(key, tensor, local, shape, offset)
    dist.save_state_dict(saved, path)
    dist.barrier()

    shape, axis = case["model"]
    local, offset, index = _shard(shape, axis, rank, world)
    weight = paddle.create_parameter(
        local,
        "float32",
        default_initializer=paddle.nn.initializer.Constant(0.0),
    )
    moment = paddle.zeros(local, dtype="float32")
    if case.get("pinned_moment"):
        moment = moment.pin_memory()
    target = {
        "w": ShardedWeight("w", weight, local, shape, offset),
        "w.moment1_0": ShardedWeight(
            "w.moment1_0", moment, local, shape, offset
        ),
    }
    aoa_config = {"aoa_statements": case["statements"]}
    if case.get("reverse"):
        aoa_config["aoa_config_reverse"] = True
    dist.load_state_dict(target, path, aoa_config=aoa_config)

    ckpt_shape = case["ckpt"][0]
    for key, base in (("w", 0.0), ("w.moment1_0", MOMENT_OFFSET)):
        expected = case["reference"](_full(ckpt_shape, base))[index]
        np.testing.assert_array_equal(
            target[key].local_tensor.numpy(), expected, err_msg=key
        )
    if case.get("pinned_moment"):
        assert target["w.moment1_0"].local_tensor.place.is_cuda_pinned_place()

    dist.barrier()
    if rank == 0:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    run_test()
