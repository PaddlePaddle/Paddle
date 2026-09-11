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

import os
from dataclasses import dataclass, field

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.dcp.metadata import LocalTensorMetadata
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    make_replicated_sharded_weight,
)

_QUANT_KEY = "weight.quant"
_SCALE_KEY = "weight.scale"
_GLOBAL_SHAPE = (4, 8)
_SCALE = 2.0


@dataclass(frozen=True)
class _ReadPlan:
    mode: str
    source_slices: dict = field(default_factory=dict)


class DequantTransform:
    """Assembles one logical fp32 weight from a uint8 payload and a scale."""

    def logical_metadata(self):
        return {
            "weight": LocalTensorMetadata(
                global_offset=(0, 0),
                local_shape=_GLOBAL_SHAPE,
                dtype="float32",
                global_shape=_GLOBAL_SHAPE,
            )
        }

    def source_keys(self, logical_key):
        return [_QUANT_KEY, _SCALE_KEY]

    def apply(self, logical_key, source_tensors, output_dtype):
        payload = source_tensors[_QUANT_KEY].astype(output_dtype)
        scale = source_tensors[_SCALE_KEY].astype(output_dtype)
        return payload * scale


class ShardedDequantTransform(DequantTransform):
    """Reads only the rows the local target shard needs.

    ``read_plan()`` is told which shard of the logical tensor this rank holds,
    so the assertions below fail unless that shard is derived from the target's
    placements rather than from its (global) ``shape``.
    """

    def read_plan(self, logical_key, target_metadata, force_global=False):
        rank = dist.get_rank()
        rows = _GLOBAL_SHAPE[0] // dist.get_world_size()
        start = rank * rows
        assert not force_global, "a single Shard(0) target needs no global read"
        assert tuple(target_metadata.global_shape) == _GLOBAL_SHAPE
        assert tuple(target_metadata.local_shape) == (
            rows,
            _GLOBAL_SHAPE[1],
        ), f"rank {rank} was told local shape {target_metadata.local_shape}"
        assert tuple(target_metadata.global_offset) == (
            start,
            0,
        ), f"rank {rank} was told global offset {target_metadata.global_offset}"
        return _ReadPlan(
            mode="local",
            source_slices={
                _QUANT_KEY: LocalTensorMetadata(
                    global_offset=(start, 0),
                    local_shape=(rows, _GLOBAL_SHAPE[1]),
                    dtype="uint8",
                    global_shape=_GLOBAL_SHAPE,
                ),
                _SCALE_KEY: LocalTensorMetadata(
                    global_offset=(0,),
                    local_shape=(1,),
                    dtype="float32",
                    global_shape=(1,),
                ),
            },
        )


class TestLoadTransformShardedTarget:
    """A Shard(0) target must receive only its own rows.

    ``dist.shard_tensor`` reports the global shape, so the local shape and
    global offset of a transform target have to be derived from its placements
    rather than from ``shape``, and the result has to be written into
    ``_local_value()`` instead of the distributed wrapper.
    """

    def __init__(self):
        self.ckpt_path = os.getenv("ckpt_path")
        self.case = os.getenv("transform_case", "global")
        self.payload = np.arange(
            _GLOBAL_SHAPE[0] * _GLOBAL_SHAPE[1], dtype=np.uint8
        ).reshape(_GLOBAL_SHAPE)

    def run_test(self):
        dist.init_parallel_env()
        self.save_physical_checkpoint()

        transform = (
            ShardedDequantTransform()
            if self.case == "local"
            else DequantTransform()
        )
        mesh = dist.ProcessMesh(list(range(dist.get_world_size())))
        state_dict = {
            "weight": dist.shard_tensor(
                paddle.zeros(list(_GLOBAL_SHAPE), dtype="float32"),
                mesh,
                [dist.Shard(0)],
            )
        }
        dist.load_state_dict(
            state_dict,
            self.ckpt_path,
            load_transform=transform,
        )

        rows = _GLOBAL_SHAPE[0] // dist.get_world_size()
        start = dist.get_rank() * rows
        expected = (
            self.payload.astype(np.float32)[start : start + rows] * _SCALE
        )
        local = state_dict["weight"]._local_value().numpy()
        assert local.shape == expected.shape, (
            f"rank {dist.get_rank()} got local shape {local.shape}, "
            f"expected {expected.shape}"
        )
        np.testing.assert_allclose(local, expected)

    def save_physical_checkpoint(self):
        source_state_dict = {
            _QUANT_KEY: make_replicated_sharded_weight(
                _QUANT_KEY, paddle.to_tensor(self.payload)
            ),
            _SCALE_KEY: make_replicated_sharded_weight(
                _SCALE_KEY,
                paddle.to_tensor(np.array([_SCALE], dtype=np.float32)),
            ),
        }
        dist.save_state_dict(source_state_dict, self.ckpt_path)


if __name__ == '__main__':
    TestLoadTransformShardedTarget().run_test()
