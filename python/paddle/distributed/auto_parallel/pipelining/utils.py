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
from paddle.distributed import ProcessMesh, fleet
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


class GET_PP_INFO_OF_LAYER:
    """Class for getting Pipeline Parallel information of model layers

    Args:
        hidden_layer_num (int): Number of model layers
        mesh (ProcessMesh): Device mesh information
        pp_schedule (str, optional): Pipeline parallel scheduling strategy. Defaults to "1F1B", representing the common model allocation strategy.
        vpp_degree (int | None, optional): VPP parallel degree. Defaults to None, indicating VPP strategy is not used.
    """

    def __init__(
        self,
        hidden_layer_num: int,
        mesh: ProcessMesh,
        pp_schedule: str = "1F1B",
        vpp_degree: int | None = None,
    ):
        self.mesh = mesh
        if "pp" in self.mesh.dim_names:
            self.pp_degree = mesh.get_dim_size("pp")
        else:
            raise ValueError("mesh must have 'pp' dimension")

        self.pp_schedule = pp_schedule.upper()
        # Check if schedule is supported
        if self.pp_schedule not in ["VPP", "1F1B", "GPIPE"]:
            raise ValueError(
                f"The pipeline schedule {self.pp_schedule} is not supported currently"
            )
        self.hidden_layer_num = hidden_layer_num
        self.vpp_degree = vpp_degree

        # Initialize VPP mode parameters
        if self.pp_schedule == "VPP":
            if self.vpp_degree is None:
                raise ValueError("VPP mode requires vpp_degree to be specified")
            self.real_pp_degree = self.vpp_degree * self.pp_degree
            if self.hidden_layer_num < self.real_pp_degree:
                raise ValueError(
                    f"In VPP mode, number of layers must be >= vpp_degree * pp_degree, "
                    f"but got {hidden_layer_num} layers with {self.real_pp_degree} stages "
                    f"(vpp_degree={vpp_degree}, pp_degree={self.pp_degree})"
                )

        # Calculate base layers per stage and remaining layers
        if pp_schedule == "VPP":
            self.base_layers = hidden_layer_num // self.real_pp_degree
            self.remaining_layers = hidden_layer_num % self.real_pp_degree
        else:
            self.base_layers = hidden_layer_num // self.pp_degree
            self.remaining_layers = hidden_layer_num % self.pp_degree

    def __getitem__(self, layer_idx: int) -> ProcessMesh:
        """Get device mesh information for specified layer through index access

        Args:
            layer_idx (int): Layer index

        Returns:
            ProcessMesh: Corresponding device mesh information
        """
        return self.get_info_by_index(layer_idx)

    def get_info_by_index(self, layer_idx: int) -> ProcessMesh:
        """Get device mesh information for specified layer

        Args:
            layer_idx (int): Layer index

        Returns:
            ProcessMesh: Corresponding device mesh information
        """
        if layer_idx >= self.hidden_layer_num:
            raise ValueError(
                f"layer_idx {layer_idx} exceeds total number of layers {self.hidden_layer_num}"
            )

        if self.pp_schedule == "VPP":
            # Calculate logical stage index (0 to real_pp_degree-1)
            if layer_idx < self.remaining_layers * (self.base_layers + 1):
                logical_stage_idx = layer_idx // (self.base_layers + 1)
            else:
                layers_not_remaining = layer_idx - self.remaining_layers * (
                    self.base_layers + 1
                )
                logical_stage_idx = (
                    self.remaining_layers
                    + layers_not_remaining // self.base_layers
                )

            # Map logical stage to physical device (0 to pp_degree-1)
            physical_device_idx = logical_stage_idx % self.pp_degree
            return self.mesh.get_mesh_with_dim("pp", physical_device_idx)
        elif (
            self.pp_schedule == "1F1B" or self.pp_schedule == "GPIPE"
        ):  # "1F1B" or "Gpipe"
            if layer_idx < self.remaining_layers * (self.base_layers + 1):
                stage_idx = layer_idx // (self.base_layers + 1)
            else:
                layers_not_remaining = layer_idx - self.remaining_layers * (
                    self.base_layers + 1
                )
                stage_idx = (
                    self.remaining_layers
                    + layers_not_remaining // self.base_layers
                )

            return self.mesh.get_mesh_with_dim("pp", stage_idx)
        else:
            raise ValueError(
                f"The pipeline schedule {self.pp_schedule} is not supported currently"
            )

    def get_info_mapping(self) -> dict[int, ProcessMesh]:
        """Get device mesh information mapping for all layers

        Returns:
            dict[int, ProcessMesh]: key is layer_index (starting from 0), value is the corresponding device group mesh
        """
        return {
            layer_idx: self.get_info_by_index(layer_idx)
            for layer_idx in range(self.hidden_layer_num)
        }
