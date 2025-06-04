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

from __future__ import annotations

import logging
from typing import Any, Callable, Union

import paddle
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.api import (
    dtensor_from_local,
)
from paddle.utils import map_structure

logger = logging.getLogger(__name__)


def _detach_and_requires_grad(x):
    o = x.detach()
    o.stop_gradient = False
    return o


def _detach_and_keep_grad(x):
    o = x.detach_()
    o.stop_gradient = x.stop_gradient
    return o


def _zero_initialize_with_meta(meta, mesh):
    assert isinstance(meta, TensorMeta)
    x = paddle.zeros(
        meta._local_shape if meta._local_shape else meta.shape, dtype=meta.dtype
    )
    if meta.placements:
        x = dtensor_from_local(x, mesh, meta.placements)
    return x


def _flatten_args(args):
    """
    Flatten the args into a list form.
    """
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        if isinstance(a, paddle.Tensor):
            flat_args.append(a)
        return a

    paddle.utils.map_structure(
        extract_tensor_args,
        args,
    )

    return flat_args


class PipeliningShapeError(RuntimeError):
    """Shape mismatch between configured and runtime values."""


def _validate_tensor_metadata(desc, expected, given):
    if not expected.shape == given.shape:
        raise PipeliningShapeError(
            f"{desc} has a shape mismatch: expected {expected.shape} actual {given.shape}"
        )
    if not expected.dtype == given.dtype:
        raise PipeliningShapeError(
            f"{desc} has a dtype mismatch: expected {expected.dtype} actual {given.dtype}"
        )


def _validate_tensors_metadata(
    desc,
    expected_tensors: list[paddle.Tensor] | tuple[paddle.Tensor, ...],
    actual_tensors: list[paddle.Tensor] | tuple[paddle.Tensor, ...],
):
    if len(expected_tensors) != len(actual_tensors):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(actual_tensors)}) does not match expected number ({len(expected_tensors)})"
        )
    for i in range(len(expected_tensors)):
        _validate_tensor_metadata(
            f"{desc}: value {i}", expected_tensors[i], actual_tensors[i]
        )


NestedStruct = Union[list[Any], tuple[Any, ...], dict[Any, Any]]


def _map_structure_only(
    type_: Any, fn: Callable[[Any], Any], structure: NestedStruct
) -> NestedStruct:
    """
    Apply `fn` to each entry which matches `type_` in `structure` and return a new structure with the same shape.
    """
    return map_structure(
        lambda x: fn(x) if isinstance(x, type_) else x, structure
    )


class TensorMeta:
    def __init__(self, tensor: paddle.Tensor):
        if tensor.is_dist():
            self.shape = tensor.shape
            self._local_shape = tensor._local_shape
        else:
            self.shape = tensor.shape
            self._local_shape = None
        self.dtype = tensor.dtype
        self.placements = None if not tensor.is_dist() else tensor.placements

    def __repr__(self):
        return f"TensorMeta(global_shape={self.shape},local_shape={self._local_shape}, dtype={self.dtype}, placements={self.placements})"


def _get_pp_mesh(pp_idx=0, pp_dim_names="pp"):
    """
    Get the mesh of the {pp_idx}th PipelineStage.
    """
    mesh = fleet.auto.get_mesh()
    assert (
        mesh is not None
    ), "the mesh is None, please call fleet.auto.set_mesh first."
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    else:
        logger.warning(
            f"The dim name of pp {pp_dim_names} not exist in global mesh {mesh}"
        )
    return mesh


def _get_stage_mesh(stage_index, pp_group_size, style=None):
    if style == "v":
        raise NotImplementedError
    if style is not None:
        raise ValueError(f"Unknown style: {style}, style can be None, v.")
    else:

        pp_idx = stage_index % pp_group_size
        return _get_pp_mesh(pp_idx)


def _friendly_debug_info(v):
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, paddle.Tensor):
        return f"Tensor({v.shape}, stop_gradient={v.stop_gradient}, dtype={v.dtype})"
    else:
        return str(v)


def _map_debug_info(a):
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return map_structure(_friendly_debug_info, a)
