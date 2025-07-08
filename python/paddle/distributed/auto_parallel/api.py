# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import logging
import os
import warnings
from collections import OrderedDict
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import _C_ops, nn, pir
from paddle.amp.grad_scaler import OptimizerState
from paddle.autograd import PyLayer
from paddle.base import unique_name
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import (
    EagerParamBase,
    Variable,
    default_main_program,
    in_dygraph_mode,
    in_pir_mode,
    use_pir_api,
)
from paddle.distributed import fleet
from paddle.distributed.auto_parallel import Engine, strategy as auto_strategy
from paddle.distributed.auto_parallel.interface import (
    shard_tensor as shard_tensor_static,
)
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.static.completion import (
    mark_as_sharding_propagation_skip_op,
)
from paddle.distributed.auto_parallel.static.dist_context import (
    get_default_distributed_context,
)
from paddle.distributed.auto_parallel.static.dist_op import DistributedOperator
from paddle.distributed.auto_parallel.static.utils import (
    convert_to_dims_mapping,
    fuse_param_func,
    get_dist_attr,
    split_mesh,
    split_param_func,
    to_list,
)
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    align,
    alignment,
    get_current_device_type,
)
from paddle.framework import core
from paddle.io.dataloader.batch_sampler import (
    DistributedBatchSampler,
    _InfiniteIterableSampler,
)
from paddle.optimizer import Optimizer

from .auto_dp_utils import (
    _enable_auto_dp,
    _fake_replicate_grad_to_partial,
    in_auto_dp_mode,
)
from .moe_utils import (
    _cal_local_shape,
    _dist_reshape,
    _dtensor_from_local,
    _NdMeshAlltoAll,
    _reshard_mesh_shape,
    _specific_alltoall_dim,
)
from .placement_type import (
    check_placements_equal,
    get_shard_spec,
    placemetns_to_dist_status,
    to_dim_map,
    to_placements,
)
from .random import determinate_rng, rng_state
from .sharding import (
    ShardingOptimizerStage1,
    get_mesh_comm_list,
    get_placement_with_sharding,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import (
        DTypeLike,
        NestedNumericSequence,
        PlaceLike,
        TensorLike,
    )
    from paddle.amp import GradScaler
    from paddle.base.framework import Program
    from paddle.distributed import Placement
    from paddle.distributed.auto_parallel.static.dist_input_spec import (
        DistributedInputSpec,
    )
    from paddle.io import DataLoader
    from paddle.metric import Metric
    from paddle.nn import Layer

    from .constants import (
        _AMPConfig,
        _DPOptimizationConfig,
        _FusedPassesConfig,
        _GradientMergeConfig,
        _MPOptimizationConfig,
        _PipelineConfig,
        _RecomputeConfig,
        _ShardingConfig,
        _SPOptimizationConfig,
    )

    _Mode: TypeAlias = Literal['train', 'eval', 'predict']

    class _Config(TypedDict, total=False):
        sharding: _ShardingConfig
        fused_passes: _FusedPassesConfig
        gradient_merge: _GradientMergeConfig
        pipeline: _PipelineConfig
        amp: _AMPConfig
        recompute: _RecomputeConfig
        mp_optimization: _MPOptimizationConfig
        dp_optimization: _DPOptimizationConfig
        sp_optimization: _SPOptimizationConfig


# There are the auto parallel API of the unified version of dynamic and static mode.
# Some APIs have the same name with the previous APIs implementation, which are
# a temporary state, and the APIs here will eventually be used.

# Part1: Shard attributes related APIs


def _to_lodtensor(tensor: paddle.Tensor):
    lodtensor = core.DenseTensor()
    if tensor.is_dist():
        if tensor._is_initialized():
            lodtensor._share_data_with(tensor._local_value().get_tensor())
        else:
            lodtensor = None
    else:
        lodtensor._share_data_with(tensor.get_tensor())

    return lodtensor


def _get_suffix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return None


class DistAttr(core.TensorDistAttr):
    """
    DistAttr specifies how tensors are distributed or sliced on ProcessMesh.

    Args:
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        sharding_specs(list[str|None]): The specification describing how to shard the Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
            >>> dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])

            >>> print(dist_attr)

    """

    def __init__(self, mesh, sharding_specs):
        # 1. inputs checking
        if not isinstance(mesh, core.ProcessMesh):
            raise ValueError(
                "The mesh must be an instance of paddle.distributed.ProcessMesh."
            )
        if not isinstance(sharding_specs, list):
            raise ValueError("The sharding_specs must be an instance of list.")
        assert all(
            isinstance(dim_name, str) or dim_name is None
            for dim_name in sharding_specs
        ), 'The dimension name in sharding_specs must be an instance of str.'

        self._sharding_specs = sharding_specs
        dims_mapping = [
            mesh.dim_names.index(dim_name) if dim_name is not None else -1
            for dim_name in sharding_specs
        ]

        # 2. init core.TensorDistAttr
        core.TensorDistAttr.__init__(self)

        self.process_mesh = mesh
        self.dims_mapping = dims_mapping
        self.mark_annotated("process_mesh")
        self.mark_annotated("dims_mapping")

    @property
    def sharding_specs(self):
        """
        Get sharding_specs of the dist_attr
        Returns:
            list[str]: sharding_specs
        """
        return self._sharding_specs


# Part2: DistTensor construction related APIs


def shard_tensor(
    data: Tensor | TensorLike | NestedNumericSequence,
    mesh: ProcessMesh,
    placements: list[Placement],
    dtype: DTypeLike | None = None,
    place: PlaceLike | None = None,
    stop_gradient: bool | None = None,
) -> Tensor:
    """
    Creates a distributed Tensor (i.e., Tensor with distributed attributes or DistTensor for short)
    from the input data, which can be a scalar, tuple, list, numpy.ndarray, or paddle.Tensor.

    If the ``data`` is already a Tensor, it will be transformed into a distributed Tensor.

    Args:
        data(scalar|tuple|list|ndarray|Tensor): Initial data for the tensor.
            Can be a scalar, list, tuple, numpy.ndarray, paddle.Tensor.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        placements(list[paddle.distributed.Placement]): the placements describe how to place the tensor on ProcessMesh, it can
            be Shard, Replicate and Partial.
        dtype(str|np.dtype, optional): The desired data type of returned tensor.
            It Can be 'bool' , 'float16' , 'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None. If None, the the dtype is inferred from ``data``
            except for python float number, in which case the dtype is inferred from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional): The place to allocate Tensor. Can be
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs.
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. If
            ``stop_gradient`` is None, set the returned Tensor's ``stop_gradient`` identical as the
            ``data.stop_gradient`` when ``data`` has ``stop_gradient`` attribute and True otherwise.
            Default: None.

    Returns:
        Tensor: A Tensor constructed from ``data`` with distributed attributes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])

            >>> # dense tensor
            >>> a = paddle.to_tensor([[1,2,3],
            ...                       [5,6,7]])

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # distributed tensor
            >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Shard(0), dist.Shard(1)])

            >>> print(d_tensor)

    """
    if place is None:
        place = paddle.framework._current_expected_place()
    place = paddle.framework._get_paddle_place(place)

    # 1. create dense tensor
    if stop_gradient is None:
        stop_gradient = getattr(data, "stop_gradient", True)

    if paddle.framework.in_pir_mode():
        assert isinstance(
            data, (type(None), pir.Value)
        ), "input tensor is not pir value."
        assert (
            data.is_dense_tensor_type()
        ), "shard_tensor() input data only supported dense tensor type right."
        tensor = data
    else:
        if isinstance(data, EagerParamBase) and not data._is_initialized():
            assert (
                data._init_func is not None
            ), "Get an uninitialized param with an unregistered init_func."
            tensor = data
        elif isinstance(data, paddle.Tensor) and dtype is None:
            # if place is not equal, it is handled in paddle.Tensor()
            tensor = data
        else:
            # `paddle.to_tensor` supports both dynamic and static mode
            tensor = paddle.to_tensor(
                data, dtype=dtype, place=place, stop_gradient=stop_gradient
            )

    if paddle.in_dynamic_mode():
        # here the dist tensor is deep copy constructed
        if isinstance(data, EagerParamBase):

            def lazy_init_hook(param, origin_hook):
                for placement in param.placements:
                    assert not placement.is_partial(), (
                        "Lazy init not support partial reshard. Notice that: shard a param to partial "
                        "won't save any memory, but will increase the communication cost!"
                    )

                # lazy init hook with randomness controlling
                def _init_func(var, block):
                    if dist.get_rank() not in param.process_mesh.process_ids:
                        # None calc rank, just return no init.
                        return
                    # get the unique rng name
                    rng_name = determinate_rng(
                        dist.get_rank(),
                        process_mesh=param.process_mesh,
                        placements=param.placements,
                    )
                    # real call the init function
                    with rng_state(rng_name):
                        origin_hook(var, block)

                return _init_func

            dist_param = EagerParamBase.from_tensor(
                tensor,
                process_mesh=mesh,
                placements=placements,
                **tensor.__dict__,
            )
            dist_param.stop_gradient = tensor.stop_gradient
            if tensor._init_func is not None:
                origin_init_func = tensor._init_func
                dist_param.set_init_func(
                    lazy_init_hook(dist_param, origin_init_func)
                )

            return dist_param
        else:
            dist_tensor = paddle.Tensor(
                tensor, process_mesh=mesh, placements=placements, place=place
            )
            # InitDistTensorWithTensor won't pass the stop gradient attribute,
            # have to pass it manually.
            dist_tensor.stop_gradient = tensor.stop_gradient
            return dist_tensor
    elif paddle.framework.in_pir_mode():
        dist_tensor = paddle._C_ops.shard_tensor(tensor, mesh, placements)
        dist_tensor.stop_gradient = tensor.stop_gradient
        dist_tensor.persistable = tensor.persistable
        return dist_tensor
    else:
        # TODO(zhiqiu): we need to refine the static shard_tensor
        sharding_specs = get_shard_spec(mesh, placements, tensor.ndim)
        return shard_tensor_static(tensor, mesh, sharding_specs)


class _moe_global_mesh_tensor(PyLayer):
    @staticmethod
    def forward(
        ctx,
        local_tensor_list,
        local_mesh_list,
        local_placements,
        mesh,
        placements,
        global_dims,
        idx=None,
    ):
        # NOTE: _local_value/Paddle.Tensor is only supported in dynamic mode
        if paddle.in_dynamic_mode():
            local_tensor = local_tensor_list[idx]
            if local_tensor.is_dist():
                local_mesh = local_tensor.process_mesh
                local_val = local_tensor._local_value()
            else:
                local_val = local_tensor
                local_mesh = None

            ctx.save_for_backward(
                copy.deepcopy(mesh),  #  global_mesh
                local_tensor.shape,  #  local_dims
                copy.deepcopy(local_mesh_list),  #  local_mesh_list
                copy.deepcopy(local_placements),  #  local_placements
            )

            place = paddle.framework._current_expected_place()
            place = paddle.framework._get_paddle_place(place)

            global_tensor = paddle.Tensor(
                local_val,
                dims=global_dims,
                process_mesh=mesh,
                placements=placements,
                place=place,
            )
            global_tensor.stop_gradient = local_tensor.stop_gradient
            return global_tensor
        else:
            ctx.save_for_backward(
                copy.deepcopy(mesh),  #  global_mesh
                copy.deepcopy(placements),  #  global_placements
                copy.deepcopy(local_mesh_list),  #  local_mesh_list
                copy.deepcopy(local_placements),  #  local_placements
            )
            dist_tensor = paddle._C_ops.moe_global_mesh_tensor(
                local_tensor_list,
                local_mesh_list,
                local_placements,
                mesh,
                placements,
                global_dims,
            )
            dist_tensor.stop_gradient = local_tensor_list[0].stop_gradient
            dist_tensor.persistable = local_tensor_list[0].persistable
            return dist_tensor

    @staticmethod
    def backward(ctx, grad_tensor):
        if paddle.in_dynamic_mode():
            global_mesh, local_dims, local_mesh_list, local_placements = (
                ctx.saved_tensor()
            )
            if local_mesh_list is None:
                return grad_tensor._local_value()
            else:
                place = paddle.framework._current_expected_place()
                place = paddle.framework._get_paddle_place(place)
                out = []
                for i, local_mesh in enumerate(local_mesh_list):
                    out.append(
                        paddle.Tensor(
                            grad_tensor._local_value(),
                            dims=local_dims,
                            process_mesh=local_mesh,
                            placements=local_placements,
                            place=place,
                        )
                    )
                    out[-1].get_tensor()._unsafe_set_skip_check_mesh(True)
                return out
        else:
            (
                global_mesh,
                global_placements,
                local_mesh_list,
                local_placements,
            ) = ctx.saved_tensor()
            return paddle._C_ops.moe_sub_mesh_tensors(
                grad_tensor,
                local_mesh_list,
                local_placements,
                global_mesh,
                global_placements,
            )


def _get_sub_meshes_and_local_placements(
    global_mesh, global_placements, sub_mesh_dim
):
    if global_mesh is None or sub_mesh_dim is None or global_placements is None:
        raise ValueError(
            "the args global_mesh, global_placements and local_mesh_dim should all be set."
        )

    sub_mesh_list = split_mesh(global_mesh, sub_mesh_dim)

    local_placements = list(global_placements)
    if sub_mesh_dim < len(local_placements):
        local_placements[sub_mesh_dim] = dist.Replicate()

    return sub_mesh_list, local_placements


def _cal_global_shape(local_shape, mesh, placements):
    # assume the each rank has the same tensor shape for now,
    # just use the local shape to calculate the global shape
    global_shape = list(local_shape)
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = placement.get_dim()
            if global_shape[shard_dim] == -1:
                continue
            local_dim_size = global_shape[shard_dim]
            global_shape[shard_dim] = local_dim_size * mesh.shape[idx]
    return global_shape


def moe_global_mesh_tensor(
    local_tensor_list, mesh, placements, local_mesh_dim=-1
):
    placements = copy.deepcopy(placements)
    local_mesh_list, local_placements = _get_sub_meshes_and_local_placements(
        mesh, placements, local_mesh_dim
    )
    process_ids = np.array(mesh.process_ids).reshape(mesh.shape)
    local_coord = np.where(process_ids == dist.get_rank())
    # when rank is not in current mesh, local_coord is empty, so we should calculate the
    # local tensor's shape.
    if local_coord[0].size == 0:
        local_tensor_idx = 0
    else:
        local_tensor_idx = local_coord[local_mesh_dim][0]
    local_tensor = local_tensor_list[local_tensor_idx]

    if paddle.in_dynamic_mode():
        # NOTE: _local_value and Paddle.Tensor() is only supported in dynamic mode
        if local_coord[0].size == 0:
            local_tensor_shape = _cal_local_shape(
                local_tensor_list[0].shape, local_mesh_list[0], local_placements
            )
        else:
            local_tensor_shape = (
                local_tensor_list[local_tensor_idx]._local_value().shape
            )
        global_dims = _cal_global_shape(local_tensor_shape, mesh, placements)
        resharded_local_tensor_list = []
        for i, tensor in enumerate(local_tensor_list):
            tensor.get_tensor()._unsafe_set_skip_check_mesh(True)
            if (
                not check_placements_equal(tensor.placements, local_placements)
                or tensor.process_mesh != local_mesh_list[i]
            ):
                resharded_local_tensor_list.append(
                    reshard(tensor, local_mesh_list[i], local_placements)
                )
                resharded_local_tensor_list[
                    -1
                ].get_tensor()._unsafe_set_skip_check_mesh(True)
            else:
                resharded_local_tensor_list.append(tensor)

        return _moe_global_mesh_tensor.apply(
            resharded_local_tensor_list,
            local_mesh_list,
            local_placements,
            mesh,
            placements,
            global_dims,
            local_tensor_idx,
        )
    elif paddle.framework.in_pir_mode():
        global_dims = _cal_global_shape(
            local_tensor._local_shape, mesh, placements
        )
        dist_tensor = paddle._C_ops.moe_global_mesh_tensor(
            local_tensor_list,
            local_mesh_list,
            local_placements,
            mesh,
            placements,
            global_dims,
        )
        dist_tensor.stop_gradient = local_tensor_list[0].stop_gradient
        dist_tensor.persistable = local_tensor_list[0].persistable
        return dist_tensor
    else:
        raise NotImplementedError(
            "dtensor_from_local_list() are only supported in dynamic and pir mode."
        )


class _moe_sub_mesh_tensors(PyLayer):
    @staticmethod
    def forward(
        ctx,
        dist_tensor,
        local_mesh_list=None,
        local_placements=None,
        local_mesh_dim=None,
        global_mesh=None,
        global_placements=None,
    ):
        ctx.save_for_backward(
            copy.deepcopy(local_mesh_list),  # local_mesh_list,
            local_placements,  # local_placements,
            local_mesh_dim,  # local_mesh_dim,
            copy.deepcopy(global_mesh),  # global_mesh,
            global_placements,  # global_placements,
            dist_tensor.shape,  # global_shape,
        )
        if paddle.in_dynamic_mode():
            if global_mesh is None and global_placements is None:
                return dist_tensor._local_value()
            else:
                if global_mesh is None or global_placements is None:
                    raise ValueError(
                        "the args global_mesh and global_placements should be set together"
                    )
                ori_mesh = dist_tensor.process_mesh
                if global_mesh != dist_tensor.process_mesh:
                    raise ValueError(
                        "the global_mesh should be the same as dist_tensor's process_mesh."
                    )
                assert check_placements_equal(
                    global_placements, dist_tensor.placements
                ), f"the global_placements ({global_placements}) is not equal to dist_tensor's placements ({dist_tensor.placements})."
                local_shape = _cal_local_shape(
                    dist_tensor.shape, global_mesh, global_placements
                )
                for idx, placement in enumerate(local_placements):
                    if placement.is_shard():
                        shard_dim = placement.get_dim()
                        local_dim_size = local_shape[shard_dim]
                        local_shape[shard_dim] = (
                            local_dim_size * local_mesh_list[0].shape[idx]
                        )

                place = paddle.framework._current_expected_place()
                place = paddle.framework._get_paddle_place(place)
                local_tensor_list = []
                for i, local_mesh in enumerate(local_mesh_list):
                    local_tensor = paddle.Tensor(
                        dist_tensor._local_value(),
                        dims=local_shape,
                        process_mesh=local_mesh,
                        placements=local_placements,
                        place=place,
                    )
                    local_tensor.get_tensor()._unsafe_set_skip_check_mesh(True)
                    local_tensor.stop_gradient = dist_tensor.stop_gradient
                    local_tensor_list.append(local_tensor)
                return local_tensor_list
        elif paddle.framework.in_pir_mode():
            local_tensors = paddle._C_ops.moe_sub_mesh_tensors(
                dist_tensor,
                local_mesh_list,
                local_placements,
                global_mesh,
                global_placements,
            )
            for local_tensor in local_tensors:
                local_tensor.stop_gradient = dist_tensor.stop_gradient
                local_tensor.persistable = dist_tensor.persistable
            return local_tensors

    @staticmethod
    def backward(ctx, *grad_tensor):
        (
            local_mesh_list,
            local_placements,
            local_mesh_dim,
            global_mesh,
            global_placements,
            global_shape,
        ) = ctx.saved_tensor()
        place = paddle.framework._current_expected_place()
        place = paddle.framework._get_paddle_place(place)
        mesh = global_mesh
        process_ids = np.array(mesh.process_ids).reshape(mesh.shape)
        local_coord = np.where(process_ids == dist.get_rank())
        if local_coord[0].size == 0:
            local_tensor_idx = 0
        else:
            local_tensor_idx = local_coord[local_mesh_dim][0]
        local_grad = grad_tensor[local_tensor_idx]

        if paddle.in_dynamic_mode():
            place = paddle.framework._current_expected_place()
            place = paddle.framework._get_paddle_place(place)
            global_tensor = paddle.Tensor(
                local_grad._local_value(),
                dims=global_shape,
                process_mesh=mesh,
                placements=global_placements,
                place=place,
            )
            return global_tensor
        elif paddle.framework.in_pir_mode():
            global_dims = _cal_global_shape(
                local_grad._local_shape, mesh, global_placements
            )

            return paddle._C_ops.moe_global_mesh_tensor(
                grad_tensor,
                local_mesh_list,
                local_placements,
                global_mesh,
                global_placements,
                global_dims,
            )


def moe_sub_mesh_tensors(
    dist_tensor, global_mesh=None, local_mesh_dim=None, global_placements=None
):
    """
    Get the local part of the ``dist_tensor`` on the specific ``local_mesh_dim``.
    """
    global_placements = copy.deepcopy(global_placements)
    local_mesh_list, local_placements = _get_sub_meshes_and_local_placements(
        global_mesh, global_placements, local_mesh_dim
    )

    if paddle.framework.in_dynamic_mode():
        return _moe_sub_mesh_tensors.apply(
            dist_tensor,
            local_mesh_list,
            local_placements,
            local_mesh_dim,
            global_mesh,
            global_placements,
        )
    elif paddle.framework.in_pir_mode():
        local_tensors = paddle._C_ops.moe_sub_mesh_tensors(
            dist_tensor,
            local_mesh_list,
            local_placements,
            global_mesh,
            global_placements,
        )
        for local_tensor in local_tensors:
            local_tensor.stop_gradient = dist_tensor.stop_gradient
            local_tensor.persistable = dist_tensor.persistable
        return local_tensors
    else:
        raise NotImplementedError(
            "moe_sub_mesh_tensors is only supported in dynamic mode."
        )


def dtensor_from_local(local_tensor, mesh, placements):
    if paddle.in_dynamic_mode():
        if local_tensor.is_dist() is True and local_tensor._is_initialized():
            raise ValueError("The input should be a local tensor.")

        return paddle.base.core.dtensor_from_local(
            local_tensor, mesh, placements
        )

    # TODO Adopt Mix2Dist Pass to allow the program could be executed actually.
    elif paddle.framework.in_pir_mode():
        return paddle._C_ops.dtensor_from_local(local_tensor, mesh, placements)
    else:
        raise RuntimeError(
            "dtensor_from_local() are only supported in dynamic or pir mode."
        )


def dtensor_to_local(dist_tensor, mesh, placements):
    if paddle.in_dynamic_mode():
        if dist_tensor.is_dist() is False:
            raise ValueError("The input should be a distributed tensor.")

        return paddle.base.core.dtensor_to_local(dist_tensor, mesh, placements)
    elif paddle.framework.in_pir_mode():
        return paddle._C_ops.dtensor_to_local(dist_tensor, mesh, placements)
    else:
        raise RuntimeError(
            "dtensor_to_local() are only supported in dynamic or pir mode."
        )


def dtensor_from_fn(
    fn: Callable[..., Tensor],
    mesh: ProcessMesh,
    placements: list[Placement],
    *args: Any,
    **kwargs: Any,
) -> Tensor:
    """
    Construct a Distributed Tensor from a function of arguments.

    Args:
        fn (callable): A callable function that takes arguments of Distributed Tensor and returns tensor.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        placements(list[paddle.distributed.Placement]): the placements describe how to place the tensor on ProcessMesh, it can
            be Shard, Replicate and Partial.
        *args (tuple): A tuple of arguments to be passed to the ``fn`` function.
        **kwargs (dict): A dict of arguments to be passed to the ``fn`` function.

    Returns:
        Tensor: A Tensor constructed from ``fn`` with distributed attributes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist
            >>> # Create a distributed attribute
            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> # Call the function dtensor_from_fn with dist_attr parameter
            >>> d_tensor = dist.dtensor_from_fn(paddle.ones, mesh, [dist.Replicate()], shape=[1])
            >>> print(d_tensor)

    """
    tensor = fn(*args, **kwargs)
    return shard_tensor(tensor, mesh, placements)


# Part3: Data conversion related APIs


def reshard(
    dist_tensor: Tensor, mesh: ProcessMesh, placements: list[Placement]
) -> Tensor:
    """
    Reshard a distributed ``paddle.Tensor`` with given distributed attributes.

    Args:
        dist_tensor(Tensor): the distributed tensor to be resharded.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        placements(list[paddle.distributed.Placement]): the placements describe how to place the tensor on ProcessMesh, it can
            be Shard, Replicate and Partial.

    Returns:
        Tensor: A Distributed Tensor resharded with distributed attributes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> # dense tensor
            >>> a = paddle.ones([10, 20])

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # distributed tensor
            >>> d_tensor = dist.shard_tensor(a, mesh, [dist.Partial()])

            >>> out_d_tensor = dist.reshard(d_tensor, mesh, [dist.Replicate()])

            >>> print(out_d_tensor)

    """

    if paddle.framework.in_dynamic_mode():
        # TODO(LiYuRio): static logic here, reshard should be changed for dygraph logic
        # when reshard has been changed align dygraph logic, delete it.

        dims_mapping, partial_status, split_factor = placemetns_to_dist_status(
            placements, dist_tensor.ndim, return_split_factor=True
        )
        dist_attr = core.TensorDistAttr()
        dist_attr.multi_dims_mapping = dims_mapping
        dist_attr.process_mesh = mesh
        dist_attr.mark_annotated("process_mesh")
        dist_attr.mark_annotated("dims_mapping")
        if len(split_factor) > 0:
            for dim, sf in split_factor.items():
                dist_attr._set_split_factor(dim, sf)
        if len(partial_status) > 0:
            dims = []
            for dim, _ in partial_status.items():
                dims.append(dim)
            dist_attr._set_partial_dims(dims)

        alltoall_dim = _specific_alltoall_dim(dist_tensor, mesh, placements)
        if alltoall_dim is not None:
            return _NdMeshAlltoAll.apply(
                dist_tensor, mesh, placements, alltoall_dim
            )

        if _reshard_mesh_shape(dist_tensor, mesh, placements):
            return _dist_reshape(
                dist_tensor, dist_tensor.shape, mesh, placements
            )
        return paddle.base.core.reshard(dist_tensor, dist_attr)
    elif in_pir_mode():
        return paddle._C_ops.reshard(dist_tensor, mesh, placements)
    else:
        assert isinstance(
            dist_tensor, Variable
        ), f"in dy2static mode, reshard's input should be Variable, but got [{dist_tensor}]"
        sharding_specs = get_shard_spec(mesh, placements, dist_tensor.ndim)
        main_program = default_main_program()
        default_dist_ctx = get_default_distributed_context()

        # output variable
        out_var = main_program.current_block().create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(['reshard_api', 'tmp'])
            ),
            dtype=dist_tensor.dtype,
            shape=dist_tensor.shape,
            type=dist_tensor.type,
            persistable=dist_tensor.persistable,
            stop_gradient=dist_tensor.stop_gradient,
        )

        # transition op
        # optimization in future to remove redundant D2D memory copy
        target_dims_mapping = convert_to_dims_mapping(sharding_specs, mesh)
        trans_op = main_program.current_block().append_op(
            type='assign',
            inputs={'X': [dist_tensor]},
            outputs={'Out': [out_var]},
        )
        dist_op = DistributedOperator(trans_op)
        dist_op.dist_attr.process_mesh = mesh
        dist_op.dist_attr.mark_annotated("process_mesh")
        dist_op.dist_attr.chunk_id = 0

        input_dist_attr = dist_op.dist_attr.get_input_dist_attr(
            dist_tensor.name
        )
        input_dist_attr.dims_mapping = target_dims_mapping
        input_dist_attr.mark_annotated("dims_mapping")
        output_dist_attr = dist_op.dist_attr.get_output_dist_attr(out_var.name)
        output_dist_attr.dims_mapping = target_dims_mapping
        output_dist_attr.mark_annotated("dims_mapping")

        default_dist_ctx.add_dist_op_for_program(dist_op)
        mark_as_sharding_propagation_skip_op(trans_op)
        # trans_op = shard_op_static(paddle.assign, mesh, [sharding_specs])
        # out_var = trans_op(dist_tensor)

        return out_var


def shard_layer(
    layer: Layer,
    process_mesh: ProcessMesh,
    shard_fn: Callable[[str, Layer, ProcessMesh], None] | None = None,
    input_fn: Callable[[Any, ProcessMesh], list[Tensor]] | None = None,
    output_fn: Callable[[Any, ProcessMesh], list[Tensor]] | None = None,
) -> Layer:
    """
    Converts all layer's parameters to DistTensor parameters according to
    the `shard_fn` specified. It could also control the conversion of input
    or output of the layer by specifying the `input_fn` and `output_fn`.
    (i.e. convert the input to `paddle.Tensor` with distributed attributes,
    convert output back to `paddle.Tensor` without distributed attributes.)

    The `shard_fn` should have the following signature:

        def shard_fn(layer_name, layer, process_mesh) -> None

    The `input_fn` should have the following signature:

        def input_fn(inputs, process_mesh) -> list(paddle.Tensor)

    In general, the type of `input_fn` return value is paddle.Tensor with distributed attributes.

    The `output_fn` should have the following signature:

        def output_fn(outputs, process_mesh) -> list(paddle.Tensor)

    In general, the type of `output_fn` return value is paddle.Tensor with distributed attributes.

    Args:
        layer (paddle.nn.Layer): The Layer object to be shard.
        process_mesh (paddle.distributed.ProcessMesh): The `ProcessMesh` information
            to be place the input `layer`.
        shard_fn (Callable): The function to shard layer parameters across
            the `process_mesh`. If not specified, by default we replicate
            all parameters of the layer across the `process_mesh`.
        input_fn (Callable): Specify how the input of the layer is sharded.
            The `input_fn` will be registered for the Layer as a `forward pre-hook`.
            By default we do not shard the input.
        output_fn (Callable): Specify how the output of the layer is sharded or
            convert it back to `paddle.Tensor` without distributed attributes.
            The `output_fn` will be registered for the Layer as `forward post-hook`.
            By default we do not shard or convert the output.
    Returns:
        Layer: A layer that contains parameters/buffers
            that are all `paddle.Tensor` with distributed attributes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> def shard_fn(layer_name, layer, process_mesh):
            ...     if layer_name == 'fc1':
            ...         layer.weight = dist.shard_tensor(layer.weight, process_mesh, [dist.Shard(0)])

            >>> layer = MLP()
            >>> layer = dist.shard_layer(layer, mesh, shard_fn)
            >>> print(layer)

            >>> # This case need to be executed in multi-card environment
            >>> # export CUDA_VISIBLE_DEVICES=0,1
            >>> # python -m paddle.distributed.launch {test_case}.py
    """
    # Ensure that process_mesh is not an empty object
    if process_mesh is None:
        raise ValueError("The argument `process_mesh` cannot be empty.")

    # Check the legality of process_mesh
    if not isinstance(process_mesh, ProcessMesh):
        raise ValueError(
            "The argument `process_mesh` is not `dist.ProcessMesh` type."
        )

    def replicate_layer_params_and_buffers(
        layer: nn.Layer, mesh: ProcessMesh
    ) -> None:
        for key, param in layer._parameters.items():
            if param is not None and not param.is_dist():
                placements = [
                    paddle.distributed.Replicate()
                    for _ in range(len(param.shape))
                ]
                layer.add_parameter(
                    key,
                    shard_tensor(param, mesh, placements),
                )
            else:
                # do nothing, the dist parameters has already been shard by shard_fn
                pass
        for key, buffer in layer._buffers.items():
            if buffer is not None and not buffer.is_dist():
                placements = [
                    paddle.distributed.Replicate()
                    for _ in range(len(buffer.shape))
                ]
                layer.register_buffer(
                    key,
                    shard_tensor(buffer, mesh, placements),
                )
            else:
                # do nothing, the dist buffers has already been shard by shard_fn
                pass

    if paddle.in_dynamic_mode():
        if shard_fn is None:
            # if shard_fn not specified, by default replicate
            # all layer's parameters and buffers
            for name, sublayers in layer.named_sublayers(include_self=True):
                replicate_layer_params_and_buffers(sublayers, process_mesh)
        else:
            # apply shard_fn to sublayers, contains self
            for name, sublayers in layer.named_sublayers(include_self=True):
                shard_fn(name, sublayers, process_mesh)
                # shard_fn may not deal with all parameters and buffers,
                # the parameters and buffers that are not shard by shard_fn
                # still need to be shard to replicated
                replicate_layer_params_and_buffers(sublayers, process_mesh)

        # register input_fn as layer's forward pre hook
        if input_fn is not None:
            layer.register_forward_pre_hook(
                lambda _, inputs: input_fn(inputs, process_mesh)
            )
        # register output_fn as layer's forward post hook
        if output_fn is not None:
            layer.register_forward_post_hook(
                lambda _, inputs, outputs: output_fn(outputs, process_mesh)
            )

        return layer
    else:
        # TODO(chenweihang): Support static mode branch later.
        raise NotImplementedError(
            "`paddle.distributed.shard_layer` only supports dynamic graph mode."
        )


def is_dist_tensor(tensor) -> bool:
    """
    Check if an input is a dist_tensor in both dynamic and static modes.
    Args:
        tensor: The input to check
    Returns:
        bool: True if the input is a dist_tensor, False otherwise
    """
    if paddle.in_dynamic_mode():
        return (
            isinstance(tensor, paddle.Tensor)
            and hasattr(tensor, 'is_dist')
            and tensor.is_dist()
        )
    else:
        return (
            isinstance(tensor, paddle.base.libpaddle.pir.Value)
            and tensor.dist_attr() is not None
        )


class _ShardOptimizer(Optimizer):
    def __init__(self, optimizer, shard_fn=None, gradient_accumulation_steps=1):
        assert (
            optimizer is not None
        ), "The argument `optimizer` cannot be empty."
        assert isinstance(
            optimizer, (paddle.optimizer.AdamW, paddle.optimizer.SGD)
        ), "`paddle.distributed.ShardOptimizer` only supports AdamW and SGD optimizer for now."

        # self.target_block = (
        #     paddle.base.framework.default_main_program().global_block()
        # )
        optimizer.helper = paddle.base.layer_helper.LayerHelper(
            optimizer.__class__.__name__
        )
        self.__dict__["_inner_opt"] = optimizer
        self._shard_clip = False
        if (
            hasattr(optimizer, "_grad_clip")
            and optimizer._grad_clip is not None
            and isinstance(optimizer._grad_clip, paddle.nn.ClipGradByGlobalNorm)
        ):
            self._shard_clip = True

        self._shard_fn = shard_fn
        self._sharding_axis = None
        self._sharding_degree = None
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if self._shard_fn is None:
            self._shard_fn = _ShardingStage0(0)

        assert isinstance(
            self._shard_fn,
            (_ShardingStage0, ShardingStage1, ShardingStage2, ShardingStage3),
        ), "shard_fn must be an instance of one of: _ShardingStage0, ShardingStage1, ShardingStage2, ShardingStage3"

        if isinstance(
            self._shard_fn, (ShardingStage1, ShardingStage2, ShardingStage3)
        ):
            self._set_and_check_sharding_prop_from_param()
            self._shard_fn._set_sharding_axis(self._sharding_axis)

        # Invoke shard_parameter in sharding stage 3 strategy
        if isinstance(self._shard_fn, ShardingStage3):
            for param in self._inner_opt._parameter_list:
                self._shard_fn._shard_parameter(param)

        self.fuse_param_view = []
        self.param_storage = []
        self.grad_storage = []
        self._sharding_group = None
        self._mp_group = None
        self.do_tensor_fusion_once = True
        self._strategy = Strategy()
        self.enable_tensor_fusion = os.getenv("FLAGS_enable_tensor_fusion") in [
            "True",
            "true",
            "1",
        ]
        self.enable_sharding_overlap = os.getenv(
            "FLAGS_enable_sharding_overlap"
        ) in ["True", "true", "1"]

    def _set_and_check_sharding_prop_from_param(self):
        global_mesh = fleet.auto.get_mesh()
        if global_mesh:
            self._sharding_degree = global_mesh.get_dim_size(
                self._shard_fn._sharding_mesh_dim
            )
        elif self._shard_fn._mesh:
            self._sharding_degree = self._shard_fn._mesh.get_dim_size(
                self._shard_fn._sharding_mesh_dim
            )
        else:
            raise ValueError(
                "The global mesh or shard_fn mesh should be set for the sharding strategy."
            )

        # Note(luchang): Now we suggest using 0 axis as sharding axis.
        self._sharding_axis = 0

        # check the placement on sharding axis is Replicate
        param_list = self._inner_opt._parameter_list
        for param in param_list:
            if not param.is_dist():
                continue
            mesh = param.process_mesh
            placements = param.placements

            if not isinstance(placements[self._sharding_axis], dist.Replicate):
                # try to infer the sharding axis
                for dim, placement in enumerate(placements):
                    if isinstance(placement, dist.Replicate):
                        self._sharding_axis = dim

            # check the placement on sharding axis is Replicate
            assert isinstance(
                placements[self._sharding_axis], dist.Replicate
            ), "The placement on sharding_axis should be Replicate"

            # check the sharding degree since it has already been set,
            # skip check when mesh is true subset of global_mesh
            if global_mesh:
                if set(mesh.process_ids) < set(global_mesh.process_ids):
                    continue
            elif self._shard_fn._mesh:
                if set(mesh.process_ids) < set(
                    self._shard_fn._mesh.process_ids
                ):
                    continue
            else:
                assert (
                    mesh.dim_size(self._sharding_axis) == self._sharding_degree
                ), "The sharding degree of all parameters must be equal currently."

    def _shard_accumulator(self, param):
        # Note (luchang): Some models may have parameters whose first dimension is 1,
        # such as modulation parameters in DiT models. These parameters can not be sharded.
        if param.shape[0] == 1:
            return

        target_name = param.name
        if param.name in self._inner_opt._master_weights.keys():
            master_weight = self._inner_opt._master_weights[param.name]
            target_name = master_weight.name
            # shard the master weight
            if isinstance(
                self._shard_fn, (ShardingStage1, ShardingStage2, ShardingStage3)
            ):
                self._inner_opt._master_weights[param.name] = (
                    self._shard_fn.shard_master_weight(param, master_weight)
                )
                self._inner_opt._master_weights[param.name].name = target_name

        # shard the accumulators
        for key in self._inner_opt._accumulators.keys():
            accumulator = self._inner_opt._accumulators[key][target_name]
            if accumulator.is_dist() and not isinstance(accumulator, pir.Value):
                continue

            if paddle.in_dynamic_mode():
                origin_accumulator_name = accumulator.name

            if isinstance(
                self._shard_fn, (ShardingStage1, ShardingStage2, ShardingStage3)
            ):
                self._inner_opt._accumulators[key][target_name] = (
                    self._shard_fn(key, param, accumulator)
                )
            else:
                if param.is_dist():
                    if 'beta' not in key:
                        # If param is a dist tensor should keep the shard info
                        # for accumulators except beta.
                        placements = param.placements
                    else:
                        # The beta should be replicated cross param's mesh
                        placements = [
                            dist.Replicate()
                            for _ in range(len(param.process_mesh.shape))
                        ]
                    self._inner_opt._accumulators[key][target_name] = (
                        shard_tensor(
                            accumulator,
                            mesh=param.process_mesh,
                            placements=placements,
                        )
                    )
            if paddle.in_dynamic_mode():
                self._inner_opt._accumulators[key][
                    target_name
                ].name = origin_accumulator_name

    def _reset_placements(self, param):
        if param.is_dist() and isinstance(
            self._shard_fn, (ShardingStage1, ShardingStage2)
        ):
            # in pir mode, reshard pass will automatically handle inplace case, so no extra work is required here.
            if not isinstance(param, pir.Value):
                new_placement = param.placements
                new_placement[self._sharding_axis] = dist.Replicate()
                out_param = dist.reshard(
                    param, param.process_mesh, new_placement
                )
                param.get_tensor()._share_data_with(out_param.get_tensor())

    def _create_accumulators(self, block, parameters):
        if isinstance(parameters, dict):
            parameters = parameters.get('params')
        # NOTE(zhiqiu): we need to create and shard accumulators for parameters one by one,
        # to avoid OOM caused by replcated accumulators.
        for p in parameters:
            self._inner_opt._create_accumulators(block, [p])
            self._shard_accumulator(p)

    def _finish_update(self, block, parameters_and_grads):
        self._inner_opt._finish_update(block, parameters_and_grads)
        if self.enable_tensor_fusion:
            # zero the grad storage for add_ op in inplace_master_grad
            for grad_storage in self.grad_storage:
                grad_storage.zero_()
                grad_storage.check_in = 0
            if not self.enable_sharding_overlap:
                for i in range(len(self.fuse_param_view)):
                    shard_size = (
                        self.param_storage[i]._numel()
                        // self._sharding_group.nranks
                    )
                    begin = shard_size * max(self._sharding_group.rank, 0)
                    end = begin + shard_size
                    slice_buffer = paddle._C_ops.view_slice(
                        self.param_storage[i], begin, end
                    )
                    self._sharding_group.process_group.all_gather(
                        slice_buffer, self.param_storage[i]
                    ).wait()
        else:
            if isinstance(parameters_and_grads, list):
                for p, _ in parameters_and_grads:
                    self._reset_placements(p)
            else:
                # reset the parameter and grad to right placements
                for p, _ in parameters_and_grads['params']:
                    self._reset_placements(p)

    def apply_gradients(self, params_grads):
        new_params_grads = []

        for param, grad in params_grads:
            new_params_grads.append(
                (param, self._shard_fn("grad", param, grad))
            )
        return Optimizer.apply_gradients(self, new_params_grads)

    def state_dict(self):
        """
        Create and shard the optimizer states e.g., accumulators and master_weights before load_state_dict.
        If training has already started or the optimizer states are already created and sharded, do nothing.
        """
        state_dict = self._inner_opt.state_dict()
        # training has already started.
        param_list = []
        if isinstance(self._inner_opt._parameter_list[0], dict):
            for param_group in self._inner_opt._parameter_list:
                param_list += param_group["params"]
        else:
            param_list = self._inner_opt._parameter_list
        for param in param_list:
            if param.stop_gradient:
                continue
            if hasattr(param, "main_grad"):
                if param.main_grad is not None:
                    return state_dict
            else:
                if param.grad is not None:
                    return state_dict

        # TODO(pangengzheng): deal with master_weights and LR_Scheduler later
        # the optimizer states are already created and sharded
        if any(
            v.is_dist()
            for k, v in state_dict.items()
            if k not in ["master_weights", "LR_Scheduler"]
        ):
            return state_dict

        # create and shard the optimizer states
        # fake the parameter gradient and invoke step to implicitly create the optimizer states.
        if not isinstance(self._inner_opt._parameter_list[0], dict):
            for param in self._inner_opt._parameter_list:
                if param.stop_gradient:
                    continue
                if hasattr(param, "main_grad"):
                    if param.main_grad is not None:
                        raise ValueError(
                            f"gradient should be None, but is {param.main_grad}"
                        )
                    param.main_grad = paddle.zeros_like(
                        param, dtype=paddle.float32
                    )
                else:
                    if param.grad is not None:
                        raise ValueError(
                            f"gradient should be None, but is {param.grad}"
                        )
                    param.grad = paddle.zeros_like(param, dtype=param.dtype)
        else:
            for param_group in self._inner_opt._param_groups:
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if hasattr(param, "main_grad"):
                        if param.main_grad is not None:
                            raise ValueError(
                                f"gradient should be None, but is {param.main_grad}"
                            )
                        param.main_grad = paddle.zeros_like(
                            param, dtype=paddle.float32
                        )
                    else:
                        if param.grad is not None:
                            raise ValueError(
                                f"gradient should be None, but is {param.grad}"
                            )
                        param.grad = paddle.zeros_like(param, dtype=param.dtype)
        self.step()
        # clear the parameter gradient
        self._inner_opt.clear_grad(set_to_zero=False)

        return self._inner_opt.state_dict()

    def _append_optimize_op(self, block, param_and_grad):
        if (
            in_auto_parallel_align_mode()  # In align mode, we use enable_delay_scale_loss by default
            and param_and_grad[1].is_dist()
        ):
            placements = param_and_grad[1].placements
            meshs = param_and_grad[1].process_mesh
            grad = param_and_grad[1]
            grad_mesh = grad.process_mesh

            def get_mesh(pp_idx=0):
                """
                获得pp_idx的mesh
                """
                mesh = fleet.auto.get_mesh()
                if "pp" in mesh.dim_names:
                    mesh = mesh.get_mesh_with_dim("pp", pp_idx)
                return mesh

            ipp = 0
            global_mesh = fleet.auto.get_mesh()
            if "pp" in global_mesh.dim_names:
                pp_degree = global_mesh.get_dim_size("pp")
                for i in range(pp_degree):
                    if meshs.process_ids == get_mesh(i).process_ids:
                        ipp = i
                        break

            change_mesh = False
            if any(
                isinstance(placement, dist.Partial) for placement in placements
            ) and (
                (meshs.process_ids == get_mesh(ipp).process_ids)
                and (meshs.dim_names != get_mesh(ipp).dim_names)
            ):
                change_mesh = True

            if change_mesh:
                grad = dist.auto_parallel.moe_utils._dist_reshape(
                    grad,
                    grad.shape,
                    get_mesh(ipp),
                    [
                        dist.Partial(dist.ReduceType.kRedSum),
                        dist.Partial(dist.ReduceType.kRedSum),
                    ],
                )
                placements = grad.placements

            for i in range(len(placements) - 1, -1, -1):
                if isinstance(placements[i], dist.Partial):
                    placements[i] = dist.Replicate()
                    grad = dist.reshard(grad, grad.process_mesh, placements)
            if self.gradient_accumulation_steps > 1 and in_dygraph_mode():
                grad /= self.gradient_accumulation_steps

            if change_mesh:
                grad = dist.auto_parallel.moe_utils._dist_reshape(
                    grad, grad.shape, grad_mesh, [dist.Replicate()]
                )
            param_and_grad = (param_and_grad[0], grad)
        self._inner_opt._append_optimize_op(block, param_and_grad)
        if self.enable_sharding_overlap:
            # overlap the first param all_gather with optimizer pass
            if hasattr(param_and_grad[0], 'last_idx'):
                idx = param_and_grad[0].last_idx
                if param_and_grad[0].last_idx == 0:
                    shard_size = (
                        self.param_storage[idx]._numel()
                        // self._sharding_group.nranks
                    )
                    begin = shard_size * max(self._sharding_group.rank, 0)
                    end = begin + shard_size
                    slice_buffer = paddle._C_ops.view_slice(
                        self.param_storage[idx], begin, end
                    )
                    task = paddle.distributed.all_gather(
                        self.param_storage[idx],
                        slice_buffer,
                        group=self._sharding_group,
                        sync_op=False,
                    )
                    self.param_storage[idx].is_sync = True
                else:
                    self.param_storage[idx].is_sync = False

    def _enable_tensor_fusion(self):
        # TODO: enable after clear FLAGS_enable_tensor_fusion
        # self.enable_tensor_fusion = True
        pass

    def _enable_sharding_overlap(self, layers):
        if hasattr(layers, 'config') and layers.config.get("to_static", False):
            return
        # self.enable_sharding_overlap = True
        if not isinstance(layers, paddle.nn.Layer):
            raise RuntimeError(
                f"`layers` must be `paddle.nn.Layer` but got {type(layers)}"
            )
        self._layers = layers

    def _reduce_scatter_gradients(self, grad_storage):
        shard_size = grad_storage._numel() // self._sharding_group.nranks
        begin = shard_size * max(self._sharding_group.rank, 0)
        end = begin + shard_size
        reduce_scattered = paddle._C_ops.view_slice(grad_storage, begin, end)
        paddle.distributed.reduce_scatter(
            reduce_scattered,
            grad_storage,
            op=paddle.distributed.ReduceOp.SUM,
            group=self._sharding_group,
            sync_op=False,
        ).wait()

    def _async_sharding_comm(self):
        if not self._layers:
            raise RuntimeError(
                "Sharding overlap requires an initialized model. "
                "Call `_enable_sharding_overlap()` to set model."
            )
        param2layer = {}
        for layer in self._layers.sublayers():
            for p in layer.parameters(include_sublayers=False):
                param2layer[id(p)] = layer

        for i in range(len(self.fuse_param_view)):
            self._reduce_scatter_gradients(self.grad_storage[i])

            def fuse_comm_hook_func(param_group_len, grad_storage, comm_group):
                @paddle.autograd.no_grad()
                def fuse_comm(*_):
                    # Ensures all gards in grad_storage have be checked in
                    grad_storage.check_in += 1
                    if grad_storage.check_in == param_group_len:
                        shard_size = grad_storage._numel() // comm_group.nranks
                        begin = shard_size * max(comm_group.rank, 0)
                        end = begin + shard_size
                        reduce_scattered = paddle._C_ops.view_slice(
                            grad_storage, begin, end
                        )
                        task = paddle.distributed.reduce_scatter(
                            reduce_scattered,
                            grad_storage,
                            op=paddle.distributed.ReduceOp.SUM,
                            group=comm_group,
                            sync_op=False,
                        )
                        grad_storage.comm_task = task

                return fuse_comm

            def fuse_all_gather_hook_func(param_storage, comm_group):
                @paddle.autograd.no_grad()
                def fuse_comm(*_):
                    # Ensures all_gather param just once per nosync param_storage
                    if not param_storage.is_sync:
                        shard_size = param_storage._numel() // comm_group.nranks
                        begin = shard_size * max(comm_group.rank, 0)
                        end = begin + shard_size
                        slice_buffer = paddle._C_ops.view_slice(
                            param_storage, begin, end
                        )
                        task = paddle.distributed.all_gather(
                            param_storage,
                            slice_buffer,
                            group=comm_group,
                            sync_op=False,
                        )
                        param_storage.is_sync = True

                return fuse_comm

            # Register reduce_scatter hooks on all parameters in this group
            param_group_len = (
                len(self.fuse_param_view[i]) * self.gradient_accumulation_steps
            )
            if "pp" in fleet.auto.get_mesh().dim_names:
                param_group_len = (
                    param_group_len * fleet.auto.get_mesh().get_dim_size("pp")
                )
            for name, view in self.fuse_param_view[i].items():
                view['param']._register_backward_hook(
                    fuse_comm_hook_func(
                        param_group_len,
                        self.grad_storage[i],
                        self._sharding_group,
                    )
                )

            # Register all_gather hooks for next chuck's parameters
            # (Uses i+1 because we need to prefetch parameters for next layer)
            if i < len(self.fuse_param_view) - 1:
                first_param = next(iter(self.fuse_param_view[i].values()))[
                    'param'
                ]
                layer = param2layer.get(id(first_param))
                layer.register_forward_pre_hook(
                    fuse_all_gather_hook_func(
                        self.param_storage[i + 1],
                        self._sharding_group,
                    )
                )

    def _build_fuse_param_view(
        self,
        params_and_grads,
        sharding_degree,
    ):
        def get_padded_size(param):
            size = np.prod(param._local_shape)
            align_size = (
                alignment[get_current_device_type()]
                // align[param.dtype]
                * sharding_degree
            )
            return ((size + align_size - 1) // align_size) * align_size

        # Calculate total buffer size needed (with padding)
        total_buffer_size = 0
        param2index = {}
        for param, _ in params_and_grads:
            param2index[param.name] = total_buffer_size
            total_buffer_size += get_padded_size(param)

        # Create fused buffers
        param_buffer = paddle.zeros(
            shape=[total_buffer_size], dtype=params_and_grads[0][0].dtype
        )
        param_buffer.is_sync = False
        grad_dtype = paddle.float32
        grad_buffer = paddle.zeros(shape=[total_buffer_size], dtype=grad_dtype)
        grad_buffer.check_in = 0
        grad_buffer.comm_task = None

        # Create views into the fused buffers
        views = {}
        for param, grad in params_and_grads:
            padded_size = get_padded_size(param)
            views[param.name] = {
                'param': param,
                'index': param2index[param.name],
            }

            index = param2index[param.name]
            param_shape = param.shape
            stop_gradient = param.stop_gradient
            param.stop_gradient = True
            param._local_value().flatten_()
            paddle.assign(
                param._local_value(),
                param_buffer._slice(
                    index,
                    index + param._numel(),
                ),
            )
            param.stop_gradient = stop_gradient
            tmp_param = paddle._C_ops.view_slice(
                param_buffer,
                index,
                index + param._numel(),
            )
            tmp_param.get_tensor()._set_dims(param._local_shape)
            tmp_param = _dtensor_from_local(
                tmp_param,
                param.process_mesh,
                param.placements,
            )
            param.get_tensor()._share_data_with(tmp_param.get_tensor())

            paddle.assign(
                grad._local_value(),
                grad_buffer._slice(
                    index,
                    index + grad._local_value()._numel(),
                ),
            )
            tmp_grad = paddle._C_ops.view_slice(
                grad_buffer,
                index,
                index + grad._local_value()._numel(),
            )
            tmp_grad.get_tensor()._set_dims(grad._local_shape)
            tmp_grad = _dtensor_from_local(
                tmp_grad,
                grad.process_mesh,
                grad.placements,
            )
            param.main_grad = tmp_grad

            # Clean up original gradient storage
            grad.get_tensor()._clear()
            paddle.device.cuda.empty_cache()

        return (views, param_buffer, grad_buffer)

    def _tensor_fusion(self, params_grads):
        """
        1. Tensor Fusion
            - Groups params/grads into contiguous param_storage/grad_storage buffers
            - Supports non-uniform partitioning across GPUs
            - Uses view_slice to access individual params/grads each step
        2. Reduce_scatter Overlap
            - Overlaps grad reduce_scatter with backward
        3. All_gather Overlap
            - Overlaps param all_gather with forward
            - Strategically scatters all_gather during forward
            (Launching all all_gather at once blocks overlap with other sync/comm ops)
        """
        if self.do_tensor_fusion_once:
            # Execute only once during first step
            # Groups params/grads and registers hooks for comm overlap
            mesh = dist.auto_parallel.get_mesh()
            shard_groups = get_mesh_comm_list(mesh, "dp")
            for group in shard_groups:
                comm_group = dist.new_group(sorted(group))
                if dist.get_rank() in group:
                    self._sharding_group = comm_group
            if "mp" in mesh._dim_names:
                mp_mesh_axis = mesh._dim_names.index("mp")
                self._mp_degree = mesh._shape[mp_mesh_axis]
                mp_groups = get_mesh_comm_list(mesh, "mp")
                for group in mp_groups:
                    comm_group = dist.new_group(sorted(group))
                    if dist.get_rank() in group:
                        self._mp_group = comm_group
            self.do_tensor_fusion_once = False
            parameters = [p_g[0] for p_g in params_grads]
            comm_buffer_size_MB = self._strategy.sharding.comm_buffer_size_MB
            if comm_buffer_size_MB < 0:
                comm_buffer_size_MB = 256
            group_size = comm_buffer_size_MB * 1024 * 1024
            is_sparse_gradient = [False] * len(parameters)
            shape_dict = {param.name: param.shape for param in parameters}
            dense_params = [param._local_value() for param in parameters]

            # group params according to comm_buffer_size_MB
            group_indices = core.eager_assign_group_by_size(
                dense_params, is_sparse_gradient, [group_size, group_size]
            )
            var_groups = OrderedDict()
            for group_idx, indices in enumerate(group_indices):
                for i in indices:
                    var_groups.setdefault(group_idx, []).append(params_grads[i])

            # create fuse_param_view, param_storage, grad_storage with groups
            for group_idx, params_and_grads in var_groups.items():
                (
                    fuse_param_view,
                    param_storage,
                    grad_storage,
                ) = self._build_fuse_param_view(
                    params_and_grads,
                    self._sharding_group.nranks,
                )
                self.fuse_param_view.append(fuse_param_view)
                self.param_storage.append(param_storage)
                self.grad_storage.append(grad_storage)

            if self.enable_sharding_overlap:
                # overlap reduce_scatter with backward
                # overlap all_gather with forward
                self._async_sharding_comm()

            # Configure gradient clipping for sharding
            if self._inner_opt._grad_clip is not None:
                self._inner_opt._grad_clip.should_comm_on_shard_dim = True
                self._inner_opt._grad_clip.sharding_group = self._sharding_group
                if "mp" in mesh._dim_names and self._mp_degree > 1:
                    self._inner_opt._grad_clip.mp_group = self._mp_group

        new_params = []
        new_grads = []
        for i in range(len(self.fuse_param_view)):
            if not self.enable_sharding_overlap:
                self._reduce_scatter_gradients(self.grad_storage[i])

            for name, view in self.fuse_param_view[i].items():
                param = view['param']
                index = view['index']
                shard_size = (
                    self.param_storage[i]._numel()
                    // self._sharding_group.nranks
                )
                rank_begin = shard_size * max(self._sharding_group.rank, 0)
                rank_end = rank_begin + shard_size
                param_begin = max(index, rank_begin)
                param_end = min(index + param._numel(), rank_end)
                if param_begin >= param_end:
                    continue
                # get new_param from param_storage
                new_param = paddle._C_ops.view_slice(
                    self.param_storage[i], param_begin, param_end
                )
                new_param = _dtensor_from_local(
                    new_param,
                    param.process_mesh,
                    [dist.Replicate()],
                )
                new_param.name = name
                new_param.stop_gradient = param.stop_gradient
                new_param.need_clip = param.need_clip
                new_param.persistable = True
                new_param.trainable = param.trainable
                new_param.stop_gradient = param.stop_gradient
                new_param.optimize_attr = param.optimize_attr
                new_param.regularizer = param.regularizer
                new_param.do_model_average = param.do_model_average
                new_param.is_distributed = param.is_distributed
                new_params.append(new_param)

                # get new_grad from grad_storage
                new_grad = paddle._C_ops.view_slice(
                    self.grad_storage[i], param_begin, param_end
                )
                new_grad = _dtensor_from_local(
                    new_grad, param.process_mesh, [dist.Replicate()]
                )
                new_grads.append(new_grad)

            if self.enable_sharding_overlap:
                # last_idx marks the last param, start asyn comm
                new_params[-1].last_idx = i
                if self.grad_storage[i].comm_task is not None:
                    self.grad_storage[i].comm_task.wait()

        new_params_grads = list(zip(new_params, new_grads))

        return new_params_grads

    def _fused_comm_before_apply_optimize(self, flags, params_grads):
        '''
        In sharding dynamic mode, optimize grad clip on partial grads causes redundant allreduce.
        Below are 2 methods to modify grad partial state by fusing comms before optimize:
            1) fuse allreduce: Change all partial state in placements to replicate via `allreduce` comms,
                e.g.
                    a) sharding_axis = 0, tensor rank = 2,
                       placements: [partial, shard(0), partial] -> [replicate, shard(0), replicate].

            2) fuse reduce_scatter: Keep shard states in placements unchanged, transform others to `shard(dim)`
               states via `reduce_scatter` comms if possible, or replicate states otherwise. In particular,
               the `placement[sharding_axis]` should be `shard(0)` if possible.
                e.g.
                    a) sharding_axis = 0, tensor rank = 2,
                       placements: [partial, partial, partial] -> [shard(0), shard(1), replicate]
                    b) sharding_axis = 0, tensor rank = 2,
                       placements: [partial, shard(0), partial ] -> [shard(1), shard(0), replicate]
        '''
        new_params_grads = []

        if flags["fuse_allreduce"]:
            for param, grad in params_grads:
                new_grad = grad
                new_placements = copy.deepcopy(grad.placements)
                for mesh_axis, placement in enumerate(grad.placements):
                    # 1. Convert all partial states to allreduce states.
                    if placement.is_partial():
                        new_placements[mesh_axis] = dist.Replicate()

                if grad.placements != new_placements:
                    # 2. Add allreduce comms via reshard API.
                    new_grad = dist.reshard(
                        grad, grad.process_mesh, new_placements
                    )
                new_params_grads.append((param, new_grad))

        elif flags["fuse_reducescatter"]:
            # Get the first non-shard dim of tensor shape in ascending order.
            # `shard_dims_set` records if dim is marked as shard in placement.
            def get_first_can_shard_dim(tensor_shape, shard_dims_set):
                for dim in range(len(tensor_shape)):
                    if dim not in shard_dims_set:
                        return dim
                return -1

            for param, grad in params_grads:
                new_placements = copy.deepcopy(grad.placements)
                new_grad = grad
                tensor_shape = grad._local_shape
                shard_dims_set = set()

                # 1. `shard_dims_set` records dims marked as shard in placement.
                for placement in grad.placements:
                    if placement.is_shard():
                        dim = placement.get_dim()
                        shard_dims_set.add(dim)

                # 2. Prioritize setting placement[sharding_axis] as shard (usually shard(0)), otherwise set as replicate.
                dim = get_first_can_shard_dim(tensor_shape, shard_dims_set)
                if dim != -1:
                    shard_dims_set.add(dim)
                    new_placements[self._sharding_axis] = dist.Shard(dim)
                else:
                    new_placements[self._sharding_axis] = dist.Replicate()

                for mesh_axis, placement in enumerate(grad.placements):
                    if mesh_axis == self._sharding_axis:
                        continue
                    # 3. Keep shard states in placements unchanged.
                    if placement.is_shard():
                        continue
                    dim = get_first_can_shard_dim(tensor_shape, shard_dims_set)
                    if dim != -1:
                        # 4. Turn other placements into shard state if possible.
                        shard_dims_set.add(dim)
                        new_placements[mesh_axis] = dist.Shard(dim)
                    else:
                        # 5. When all tensor dims are in shard state, set remaining placements to replicate.
                        new_placements[mesh_axis] = dist.Replicate()

                if grad.placements != new_placements:
                    # 6. Add reduce_scatter comms via reshard API.
                    new_grad = dist.reshard(
                        grad, grad.process_mesh, new_placements
                    )

                new_params_grads.append((param, new_grad))
        else:
            new_params_grads = params_grads
        return new_params_grads

    def _apply_optimize(
        self, loss, startup_program, params_grads, param_group_idx=0
    ):
        if paddle.in_dynamic_mode() and isinstance(
            self._shard_fn, ShardingStage1
        ):
            if self.enable_tensor_fusion:
                # tensor fusion fuse params/grads into large chunks, no need _fused_comm_before_apply_optimize.
                params_grads = self._tensor_fusion(params_grads)
            else:
                # Fuse the communication of gradients prior to the optimization operation in the dynamic mode.
                # Get fuse optimization flag.
                def get_env(flag_name):
                    if os.getenv(flag_name) in ['True', 'true', '1']:
                        return True
                    return False

                # TODO: This optimization hasn't been verified on a wide range of models. Currently,
                # it's controlled by a flag switch and will be removed later.
                fuse_allreduce_in_opt = get_env("FLAGS_fuse_allreduce_in_opt")
                fuse_reducescatter_in_opt = get_env(
                    "FLAGS_fuse_reducescatter_in_opt"
                )
                assert not (
                    fuse_allreduce_in_opt and fuse_reducescatter_in_opt
                ), "The `FLAGS_fuse_allreduce_in_opt` switch and the `FLAGS_fuse_reducescatter_in_opt` switch cannot be turned on simultaneously."

                if fuse_allreduce_in_opt or fuse_reducescatter_in_opt:
                    flags = {}
                    flags["fuse_allreduce"] = fuse_allreduce_in_opt
                    flags["fuse_reducescatter"] = fuse_reducescatter_in_opt
                    params_grads = self._fused_comm_before_apply_optimize(
                        flags, params_grads
                    )

        return super()._apply_optimize(
            loss, startup_program, params_grads, param_group_idx
        )

    def __getattr__(self, item):
        if "_inner_opt" in self.__dict__:
            if item == "_inner_opt":
                return self.__dict__[item]
            return getattr(self.__dict__["_inner_opt"], item)
        else:
            raise AttributeError

    def __setattr__(self, item, value):
        if item == '_inner_opt':
            msg = f'{type(self).__name__}._inner_opt is READ ONLY'
            raise AttributeError(msg)
        return setattr(self._inner_opt, item, value)


class _ShardingStageBase:
    def __init__(self, mesh, sharding_mesh_dim):
        self._mesh = mesh
        self._sharding_axis = 0
        self._sharding_mesh_dim = sharding_mesh_dim

    def _set_sharding_axis(self, sharding_axis):
        self._sharding_axis = sharding_axis

    def shard_master_weight(
        self, param: Tensor, master_weight: Tensor
    ) -> Tensor:
        if param.is_dist():
            if os.getenv("FLAGS_enable_tensor_fusion") in ["True", "true", "1"]:
                placements = param.placements
            else:
                placements = get_placement_with_sharding(
                    param, self._sharding_axis
                )
            if isinstance(master_weight, pir.Value):
                data_op = master_weight.get_defining_op()
                assert (
                    data_op.name() == "pd_op.data"
                ), "The master weight must be a result of data op."
                dim_map, partial_status = to_dim_map(
                    placements, len(master_weight.shape)
                )
                dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        param.process_mesh, dim_map, partial_status
                    )
                )
                dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    master_weight.type(), dist_attr
                )
                master_weight.set_type(dist_type)
                data_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        param.process_mesh, [], [dist_attr]
                    )
                )

            if paddle.in_dynamic_mode() and master_weight.is_dist():
                master_weight = reshard(
                    master_weight,
                    mesh=param.process_mesh,
                    placements=placements,
                )
        return master_weight

    def _init_dist_attr(self, tensor: Tensor, param: Tensor, placements: list):
        dim_map, partial_status = to_dim_map(placements, len(tensor.shape))
        dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            param.process_mesh, dim_map, partial_status
        )
        dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            tensor.type(), dist_attr
        )
        tensor.set_type(dist_type)
        op_dist_attr = paddle.base.libpaddle.pir.create_op_dist_attribute(
            param.process_mesh, [], [dist_attr]
        )
        tensor.get_defining_op().dist_attr = op_dist_attr

    def _apply_placement(
        self, tensor: Tensor, param: Tensor, placements: list
    ) -> Tensor:
        if tensor.is_dist():
            op = tensor.get_defining_op()
            if op.name() == "pd_op.data":
                self._init_dist_attr(tensor, param, placements)
                return tensor
            return dist.reshard(tensor, param.process_mesh, placements)

        return shard_tensor(
            tensor,
            mesh=param.process_mesh,
            placements=placements,
        )

    def _reshard_fake_replicate_grad_to_partial(self, grad: Tensor) -> Tensor:
        return _fake_replicate_grad_to_partial(grad, self._sharding_axis)


class _ShardingStage0(_ShardingStageBase):
    def __init__(
        self, sharding_mesh_dim: int | str, mesh: ProcessMesh | None = None
    ) -> None:
        super().__init__(mesh, sharding_mesh_dim)
        self.sharding_axis = 0

    def __call__(self, key: str, param: Tensor, tensor: Tensor) -> Tensor:
        if key == "grad" and in_auto_dp_mode():
            return self._reshard_fake_replicate_grad_to_partial(tensor)

        return tensor


class ShardingStage1(_ShardingStageBase):
    """
    A builtin shard_fn for shard_optimizer interface, users can pass it to shard_optimizer to implement sharding optimization with stage 1.

    Args:
        sharding_mesh_dim(int|str): The sharding dimension in the mesh.
        mesh(None|paddle.distributed.ProcessMesh): If mesh is not None, the `ProcessMesh` object describes the Cartesian topology of the used processes for dense type parameters. Note: Currently, only one mesh configuration is supported for all dense parameters. If there is a need for multiple mesh configurations, please configure them yourself in the upper layer networking code.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> opt = dist.shard_optimizer(opt, dist.ShardingStage1("x", mesh))
            >>> for _ in range(5):
            >>>     loss = layer(batch)
            >>>     loss.backward()
            >>>     opt.step()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py
    """

    def __init__(
        self,
        sharding_mesh_dim: int | str,
        mesh: ProcessMesh | None = None,
    ) -> None:
        super().__init__(mesh, sharding_mesh_dim)

    def __call__(self, key: str, param: Tensor, tensor: Tensor) -> Tensor:
        if not param.is_dist():
            return tensor

        # Only deal with momentum in optimizer, beta should be replicated cross param's mesh
        if (
            os.getenv("FLAGS_enable_tensor_fusion") not in ["True", "true", "1"]
            and 'beta' not in key
        ):
            placements = get_placement_with_sharding(param, self._sharding_axis)
        else:
            placements = [
                dist.Replicate() for _ in range(len(param.process_mesh.shape))
            ]

        if key == "grad" and in_auto_dp_mode():
            tensor = self._reshard_fake_replicate_grad_to_partial(tensor)

        return self._apply_placement(tensor, param, placements)


class ShardingStage2(ShardingStage1):
    """
    A builtin shard_fn for shard_optimizer interface, users can pass it to shard_optimizer to implement sharding optimization with stage 2.

    Args:
        sharding_mesh_dim(int|str): The sharding dimension name in the mesh.
        mesh(None|paddle.distributed.ProcessMesh): If mesh is not None, the `ProcessMesh` object describes the Cartesian topology of the used processes for dense type parameters. Note: Currently, only one mesh configuration is supported for all dense parameters. If there is a need for multiple mesh configurations, please configure them yourself in the upper layer networking code.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> opt = dist.shard_optimizer(opt, dist.ShardingStage2("x", mesh))
            >>> for _ in range(5):
            >>>     loss = layer(batch)
            >>>     loss.backward()
            >>>     opt.step()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py
    """

    # Note(luchang): Due to reshard optimizations in Paddle where all-reduce + slicing is fused into reduce_scatter,
    # the current behavior of ShardingStage2 is effectively the same as ShardingStage1.
    pass


class ShardingStage3(_ShardingStageBase):
    """
    A builtin shard_fn for shard_optimizer interface, users can pass it to shard_optimizer to implement sharding optimization with stage 3.

    Args:
        sharding_mesh_dim(int|str): The sharding dimension name in the mesh.
        mesh(None|paddle.distributed.ProcessMesh): If mesh is not None, the `ProcessMesh` object describes the Cartesian topology of the used processes for dense type parameters. Note: Currently, only one mesh configuration is supported for all dense parameters. If there is a need for multiple mesh configurations, please configure them yourself in the upper layer networking code.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> opt = dist.shard_optimizer(opt, dist.ShardingStage3("x", mesh))
            >>> for _ in range(5):
            >>>     loss = layer(batch)
            >>>     loss.backward()
            >>>     opt.step()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py
    """

    def __init__(
        self,
        sharding_mesh_dim: int | str,
        mesh: ProcessMesh | None = None,
    ) -> None:
        super().__init__(mesh, sharding_mesh_dim)

    def _shard_parameter(self, param):
        if param.is_dense() and self._mesh is not None:
            placements = []
            for _ in range(len(self._mesh.shape)):
                placements.append(dist.Replicate())
            param._to_dist_(placements, self._mesh)

        if param.is_dist():
            new_placements = get_placement_with_sharding(
                param, self._sharding_axis
            )
            shard_param = dist.reshard(
                param, param.process_mesh, new_placements
            )
            # change the holder of param to new shard_param
            param.get_tensor()._share_data_with(shard_param.get_tensor())

    def _unshard_parameter(self, param):
        if param.is_dist():
            new_placements = param.placements
            if isinstance(new_placements[self._sharding_axis], dist.Shard):
                new_placements[self._sharding_axis] = dist.Replicate()

            new_param = dist.reshard(param, param.process_mesh, new_placements)
            param.get_tensor()._share_data_with(new_param.get_tensor())

    def __call__(self, key: str, param: Tensor, tensor: Tensor) -> Tensor:
        if not param.is_dist():
            return tensor

        if key == "grad" and in_auto_dp_mode():
            raise RuntimeError(
                "Sharding Stage 3 does not support auto dp mode yet."
            )

        if 'beta' not in key:
            placements = param.placements
            if all(isinstance(p, dist.Replicate) for p in placements):
                placements = get_placement_with_sharding(
                    param, self._sharding_axis
                )
        else:
            placements = [dist.Replicate() for _ in param.process_mesh.shape]
        return self._apply_placement(tensor, param, placements)


def shard_optimizer(
    optimizer: Optimizer,
    shard_fn: Callable[[str, Tensor, Tensor], Tensor] | None = None,
    gradient_accumulation_steps: int = 1,
) -> _ShardOptimizer:
    """

    Warp the global view optimizer to distributed view.

    Note:
        The `shard_fn` should have the following signature:
            def shard_fn(accumulator_name, param, accumulator) -> sharded_accumulator

    Args:
        optimizer (paddle.optimizer.Optimizer): The optimizer to be sharded.
        shard_fn (Callable, optional): The function to shard accumulators. If not specified,
           we simply pass down the dist attr of the params.

    Returns:
        An optimizer with distributed view.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist
            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> opt = dist.shard_optimizer(opt)
            >>> for _ in range(5):
            >>>     loss = layer(batch)
            >>>     loss.backward()
            >>>     opt.step()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py

    """
    return _ShardOptimizer(optimizer, shard_fn, gradient_accumulation_steps)


def shard_scaler(scaler: GradScaler) -> GradScaler:
    """

    Warp the global view grad_scaler to distributed view.

    Args:
        scaler (paddle.amp.GradScaler): The GradScaler to be sharded.

    Returns:
        A GradScaler with distributed view.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist
            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> class MLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc1 = paddle.nn.Linear(8, 8)
            ...         self.fc2 = paddle.nn.Linear(8, 8)
            ...
            ...     def forward(self, input):
            ...         return self.fc2(self.fc1(input))
            >>> layer = MLP()
            >>> batch = paddle.rand(shape=[8, 8])
            >>> opt = paddle.optimizer.AdamW(parameters=layer.parameters())
            >>> layer, opt = paddle.amp.decorate(layer, opt, level='O2')
            >>> scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            >>> scaler = dist.shard_scaler(scaler)
            >>> opt = dist.shard_optimizer(opt)
            >>> for _ in range(5):
            >>>     with paddle.amp.auto_cast(True):
            >>>         loss = layer(batch)
            >>>     scaled = scaler.scale(loss)
            >>>     scaled.backward()
            >>>     scaler.step(opt)
            >>>     scaler.update()
            >>>     opt.clear_grad()
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1 {test_case}.py

    """

    def unscale_method(self, optimizer):
        if not self._enable:
            return

        optimizer_state = self._optimizer_states[id(optimizer)]

        if optimizer_state["state"] is OptimizerState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["state"] is OptimizerState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        src_mesh = None
        current_process_mesh = None

        self._found_inf = paddle.to_tensor(np.array([0]).astype(np.bool_))
        mesh2param_grads = {}
        if getattr(optimizer, '_param_groups', None) and isinstance(
            optimizer._param_groups[0], dict
        ):
            for group in optimizer._param_groups:
                for param in group['params']:
                    tgt_grad = param._grad_ivar()
                    if (
                        tgt_grad is not None
                        and getattr(
                            tgt_grad, '_is_initialized', lambda: False
                        )()
                    ):
                        if src_mesh is None:
                            src_mesh = tgt_grad.process_mesh
                        if (
                            current_process_mesh is None
                            and tgt_grad._is_initialized()
                        ):
                            current_process_mesh = tgt_grad.process_mesh
                        if tgt_grad.process_mesh not in mesh2param_grads:
                            mesh2param_grads[tgt_grad.process_mesh] = [tgt_grad]
                        else:
                            mesh2param_grads[tgt_grad.process_mesh].append(
                                tgt_grad
                            )
        else:
            for param in optimizer._parameter_list:
                tgt_grad = param._grad_ivar()
                if (
                    tgt_grad is not None
                    and getattr(tgt_grad, '_is_initialized', lambda: False)()
                ):
                    if src_mesh is None:
                        src_mesh = tgt_grad.process_mesh
                    if (
                        current_process_mesh is None
                        and tgt_grad._is_initialized()
                    ):
                        current_process_mesh = tgt_grad.process_mesh
                    if tgt_grad.process_mesh not in mesh2param_grads:
                        mesh2param_grads[tgt_grad.process_mesh] = [tgt_grad]
                    else:
                        mesh2param_grads[tgt_grad.process_mesh].append(tgt_grad)

        for _, param_grads in mesh2param_grads.items():
            temp_param_grads_half = []
            temp_param_grads_fp32 = []
            temp_found_inf = paddle.to_tensor(np.array([0]).astype(np.bool_))
            temp_found_inf_half = paddle.to_tensor(
                np.array([0]).astype(np.bool_)
            )
            temp_found_inf_fp32 = paddle.to_tensor(
                np.array([0]).astype(np.bool_)
            )
            if self._scale.is_dist():
                temp_scale = self._scale._local_value()
            else:
                temp_scale = self._scale
            for grad in param_grads:
                if grad.dtype in [
                    core.VarDesc.VarType.FP16,
                    paddle.float16,
                    core.VarDesc.VarType.BF16,
                    paddle.bfloat16,
                ]:
                    temp_param_grads_half.append(grad)
                else:
                    temp_param_grads_fp32.append(grad)
            if len(temp_param_grads_half):
                _, temp_found_inf_half = _C_ops.check_finite_and_unscale_(
                    temp_param_grads_half,
                    temp_scale,
                )

                # AllReduce for "bool" is not supported on XPU
                if "xpu" in paddle.device.get_device():
                    temp_param_grads_half = paddle.cast(
                        temp_param_grads_half, "int32"
                    )
                    temp_param_grads_half = paddle.sum(temp_param_grads_half)
                    temp_param_grads_half = paddle.cast(
                        temp_param_grads_half, "bool"
                    )

                temp_found_inf = _C_ops.bitwise_or(
                    temp_found_inf, temp_found_inf_half
                )
            if len(temp_param_grads_fp32):
                _, temp_found_inf_fp32 = _C_ops.check_finite_and_unscale_(
                    temp_param_grads_fp32,
                    temp_scale,
                )

                # AllReduce for "bool" is not supported on XPU
                if "xpu" in paddle.device.get_device():
                    temp_found_inf_fp32 = paddle.cast(
                        temp_found_inf_fp32, "int32"
                    )
                    temp_found_inf_fp32 = paddle.sum(temp_found_inf_fp32)
                    temp_found_inf_fp32 = paddle.cast(
                        temp_found_inf_fp32, "bool"
                    )

                temp_found_inf = _C_ops.bitwise_or(
                    temp_found_inf, temp_found_inf_fp32
                )
            # All the 'temp_found_inf' will be `resharded` to `src_mesh` to calculate the value of `self._found_inf`.
            temp_found_inf = dist.reshard(
                temp_found_inf, src_mesh, temp_found_inf.placements
            )
            self._found_inf = _C_ops.bitwise_or(self._found_inf, temp_found_inf)

        # The rank of src_mesh, should not overwrite the original variable `self._found_inf`
        if self._found_inf.process_mesh == current_process_mesh:
            for process_mesh in mesh2param_grads.keys():
                _ = dist.reshard(
                    self._found_inf, process_mesh, self._found_inf.placements
                )
        else:
            # The rank of other mesh, should overwrite the original variable `self._found_inf`
            self._found_inf = dist.reshard(
                self._found_inf,
                current_process_mesh,
                self._found_inf.placements,
            )
        optimizer_state["state"] = OptimizerState.UNSCALED

    scaler._unscale = MethodType(unscale_method, scaler)

    return scaler


# Part4: Convert To Static Graph related APIs
class FusePasses:
    """
    A helper class for users to configure the fuse passes.
    """

    enable: bool
    gemm_epilogue: bool
    dropout_add: bool

    def __init__(self, config_dict=None):
        self.enable = False
        self.gemm_epilogue = False
        self.dropout_add = False
        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown fuse pass {key}")


class Strategy(auto_strategy.BaseConfig):
    """
    The `Strategy` object is used to configure the parallelization
    and optimization strategies for static graph. Currently supports
    configuring ``sharding``, ``fused_passes``, ``gradient_merge``
    and ``pipeline``. More strategies will be supported in the future.

    ``sharding`` is used to configure the sharding states of the optimizer,
    for saving the GPU memory.

    ``fused_passes`` is used to configure the fusion of the computation in
    the model.

    ``gradient_merge`` is used to configure the gradient merge strategy in
    training.

    ``pipeline`` is used to configure the pipeline parallelism strategy.

    Args:
        config(dict|None, optional): The user-defined configurations.
            If ``config`` is None, use default configurations. If it is
            a dict, the items inside the dict will be used to set the
            configurations, and the others remain the default values.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> strategy = dist.Strategy()

            >>> strategy.sharding.enable = True
            >>> strategy.sharding.stage = 2
            >>> strategy.sharding.degree = 2

            >>> strategy.gradient_merge.enable = True
            >>> strategy.gradient_merge.k_steps = 2
            >>> strategy.gradient_merge.avg = False

            >>> strategy.pipeline.enable = True
            >>> strategy.pipeline.schedule_mode = "1F1B" # default is "1F1B"
            >>> strategy.pipeline.micro_batch_size = 2
    """

    def __init__(self, config: _Config | None = None) -> None:
        if config is not None:
            if isinstance(config, dict):
                self._config_dict = copy.deepcopy(config)
            else:
                raise ValueError(
                    f"Expected a dictionary. But received: {config}"
                )
        else:
            self._config_dict = {}

        category = auto_strategy.constants.BASE
        super().__init__(category, self._config_dict)

        config_dict = self._config_dict.get(
            auto_strategy.constants.SHARDING, None
        )
        self._sharding = auto_strategy.ShardingConfig(config_dict)

        config_dict = self._config_dict.get(
            auto_strategy.constants.GRADIENT_MERGE, None
        )
        self._gradient_merge = auto_strategy.GradientMergeConfig(config_dict)

        config_dict = self._config_dict.get(
            auto_strategy.constants.PIPELINE, None
        )
        self._pipeline = auto_strategy.PipelineConfig(config_dict)

        config_dict = self._config_dict.get(auto_strategy.constants.AMP, None)
        self._amp = auto_strategy.AMPConfig(config_dict)

        config_dict = self._config_dict.get(
            auto_strategy.constants.FUSED_PASSES, None
        )
        self._fused_passes = FusePasses(config_dict)

        # template interface
        config_dict = self._config_dict.get(
            auto_strategy.constants.RECOMPUTE, None
        )
        self._recompute = auto_strategy.RecomputeConfig(config_dict)

        config_dict = self._config_dict.get(
            auto_strategy.constants.MP_OPTIMIZATION, None
        )
        self._mp_optimization = auto_strategy.MPOptimizationConfig(config_dict)

        config_dict = self._config_dict.get(
            auto_strategy.constants.DP_OPTIMIZATION, None
        )
        self._dp_optimization = auto_strategy.DPOptimizationConfig(config_dict)
        config_dict = self._config_dict.get(
            auto_strategy.constants.SP_OPTIMIZATION, None
        )
        self._sp_optimization = auto_strategy.SPOptimizationConfig(config_dict)

        self._full_graph = self._config_dict.get("full_graph", True)

    def _from_legacy_strategy(self, legacy_strategy):
        """
        NOTE(lizhiyu): This is a template function to get `dist.Strategy` from `fleet.auto.Strategy`.
        """
        import copy

        category = auto_strategy.constants.BASE
        base_config = auto_strategy.constants.get_category_default_config(
            category
        )
        for key in base_config.keys():
            setattr(self, key, getattr(legacy_strategy, key))
        self._fused_passes.enable = legacy_strategy.fused_passes.enable
        if (
            "fused_gemm_epilogue_pass"
            in legacy_strategy.fused_passes.fused_passes_list
        ):
            self._fused_passes.gemm_epilogue = True
        if (
            "fused_dropout_add_pass"
            in legacy_strategy.fused_passes.fused_passes_list
        ):
            self._fused_passes.dropout_add = True

        self._amp = copy.deepcopy(legacy_strategy.amp)
        self._sharding = copy.deepcopy(legacy_strategy.sharding)
        self._gradient_merge = copy.deepcopy(legacy_strategy.gradient_merge)
        self._pipeline = copy.deepcopy(legacy_strategy.pipeline)
        # The below are template interfaces
        self._recompute = copy.deepcopy(legacy_strategy.recompute)
        self._mp_optimization = copy.deepcopy(legacy_strategy.mp_optimization)
        self._dp_optimization = copy.deepcopy(legacy_strategy.dp_optimization)
        self._sp_optimization = copy.deepcopy(legacy_strategy.sp_optimization)

    @property
    def full_graph(self) -> bool:
        """
        Whether to use AST mode.
        """
        return self._full_graph

    @property
    def sharding(self) -> auto_strategy.ShardingConfig:
        """
        ``sharding`` is used to configure the sharding states of the optimizer,
        containing following configs:

            ``enable`` (bool): whether to enable sharding. Default: False.

            ``stage`` (int): can be set to 1, 2 or 3. 1 indicates the optimizer states segmentation,
            2 indicates optimizer states and gradient segmentation, 3 indicates the segmentation
            of optimizer states, gradient and parameters. Default: 1.

            ``degree`` (int): the number of segmentation pieces. Default: 8.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.distributed as dist

                >>> strategy = dist.Strategy()

                >>> strategy.sharding.enable = True
                >>> strategy.sharding.stage = 2
                >>> strategy.sharding.degree = 2
        """
        return self._sharding

    @property
    def gradient_merge(self) -> auto_strategy.GradientMergeConfig:
        """
        ``gradient_merge`` is used to configure the gradient merge strategy in
        training, containing following configs:

            ``enable`` (bool): whether to enable gradient merge. Default: False.

            ``k_steps`` (int): the number of steps for merging gradients. Default: 1.

            ``avg`` (bool): whether to average the gradients of each step. Default: True.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.distributed as dist

                >>> strategy = dist.Strategy()

                >>> strategy.gradient_merge.enable = True
                >>> strategy.gradient_merge.k_steps = 2
                >>> strategy.gradient_merge.avg = True
        """
        return self._gradient_merge

    @property
    def fused_passes(self) -> FusePasses:
        """
        ``fused_passes`` is used to configure the fusion of the computation in
        the model, containing following configs:

            ``enable`` (bool): whether to enable fused passes. Default: False.

            ``gemm_epilogue`` (bool): whether to fuse ``matmul`` and ``add`` computation
            in the ``Linear`` layer. Default: False

            "dropout_add" (bool): whether to fuse ``dropout`` and ``add`` computation. Default: False.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.distributed as dist

                >>> strategy = dist.Strategy()

                >>> strategy.fused_passes.enable = True
                >>> strategy.fused_passes.gemm_spilogue = True
                >>> strategy.fused_passes.dropout_add = True
        """
        return self._fused_passes

    @property
    def pipeline(self) -> auto_strategy.PipelineConfig:
        """
        ``pipeline`` is used to configure the pipeline parallelism,
        containing following configs:

            ``enable`` (bool): whether to enable pipeline parallelism. Default: False.

            ``schedule_mode`` (str): the scheduling mode of pipeline parallelism. Default: "1F1B".

            ``micro_batch_size`` (int): the size of each micro-batch inside a mini-batch. Default: 1.

            ``accumulate_steps`` (int): number of steps for accumulating. Default: 1.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.distributed as dist

                >>> strategy = dist.Strategy()

                >>> strategy.pipeline.enable = True
                >>> strategy.pipeline.micro_batch_size = 2
        """
        return self._pipeline

    @property
    def amp(self) -> auto_strategy.AMPConfig:
        """
        ``amp`` is used to configure the amp,
        containing following configs:

            ``enable`` (bool):  whether to enable AMP. Default: False.
            ``dtype``, (str): the data type of AMP. Default: "float16".
            ``level``, (str): the level of AMP. Default: "O1".
            ``init_loss_scaling``, (float): the initial value of loss scaling. Default: 32768.0
            ``incr_every_n_steps``, (int): the number of steps for increasing loss scaling. Default: 1000
            ``decr_every_n_nan_or_inf``, (int): the number of steps for decreasing loss scaling. Default: 2
            ``incr_ratio``, (float): the ratio for increasing loss scaling. Default: 2.0
            ``decr_ratio``, (float): the ratio for decreasing loss scaling. Default: 2.0
            ``use_dynamic_loss_scaling``, (bool): whether to use dynamic loss scaling. Default: False
            ``custom_white_list``, (list): the list of names for which AMP will be applied. Default: []
            ``custom_black_list``, (list): the list of names for which AMP will not be applied. Default: []
            ``custom_black_varnames``, (list): the list of names for which AMP will not be applied. Default: []
            ``use_fp16_guard``, (bool): whether to use fp16 guard. Default: False
            ``use_bf16_guard``, (bool): whether to use bf16 guard. Default: False
            ``use_master_grad``, (bool): whether to use master grad. Default: False

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.distributed as dist

                >>> strategy = dist.Strategy()

                >>> strategy.amp.enable = True
                >>> strategy.amp.dtype = "float16"
                >>> strategy.amp.level = "O2"
        """
        return self._amp


class DistModel:
    """
    `DistModel` is the model converted from a ``paddle.nn.layer`` with distributed
    tensors as its parameters. It contains the static graph converted from a
    ``paddle.nn.layer`` whose parameters are distributed tensors (constructed
    from ``paddle.distributed.shard_tensor``), and provides the APIs for training,
    evaluation and prediction with the static graph.

    It is suggested to generate DistModel by ``paddle.distributed.to_static``,
    not directly by ``paddle.distributed.DistModel``.

    Please first set the DistModel to "train", "eval" or "predict" mode with
    ``train()/eval()/predict()`` method and then use the ``__call__`` method for
    training, evaluation and prediction respectively.

    For more details of the usage, please refer to the sample code in
    ``paddle.distributed.to_static``.

    Args:
        layer(paddle.nn.Layer): The layer in dygraph mode, whose parameters
            are distributed tensors generated by ``shard_tensor``.
        loader(ShardDataLoader|paddle.io.DataLoader): The data loader used in dygraph mode,
            used to infer inputs_spec and labels_spec.
        loss(Loss|Callable|None, optional): The loss function for training
            or evaluating the model. Can be a `paddle.nn.Layer` instance or
            any callable function. If loss is not None, DistModel will be set
            to "train" (when the optimizer is also not None) or "eval" mode
            (when optimizer is None) in default. If it is None, DistModel will
            be set to "predict" mode in default. Default: None.
        optimizer(paddle.optimizer.Optimizer|None, optional): The optimizer
            for training. If both optimizer and loss are set, DistModel will
            be set to "train" mode in default. Default: None.
        strategy(paddle.distributed.Strategy|None, optional): Configs for
            parallel strategies and optimization settings (e.g. sharding,
            pipeline parallelism). Default: None.
        input_spec(list[list[paddle.distributed.DistributedInputSpec]]|None, optional):
            The custom input specs specify the shape, dtype, and name information
            of model inputs and labels. If it is not None, the input specs and
            label specs will be inferred from the custom input specs. The custom
            input specs should be a list containing two sublists: the first
            sublist represents theinput specs, and the second sublist represents
            the label specs. Default: None.
    """

    def __init__(
        self,
        layer: Layer,
        loader: ShardDataloader | DataLoader,
        loss: Layer | Callable[..., Any] | None = None,
        optimizer: Optimizer | None = None,
        strategy: Strategy | None = None,
        metrics: list[Metric] | None = None,
        input_spec: list[list[DistributedInputSpec]] | None = None,
    ) -> None:
        self._inner_strategy = self.__convert_strategy(strategy)
        self._structured_to_parameter_name = {
            k: v.name for k, v in layer.state_dict().items()
        }
        self._parameter_to_structured_name = {
            v: k for k, v in self._structured_to_parameter_name.items()
        }
        if os.getenv("POD_NAME"):
            dist.utils.log_utils.get_logger(logging.INFO).info(
                "Distribute training by paddle.distributed.launch"
            )
            dist.fleet.init(is_collective=True)

        if (
            strategy
            and strategy.sharding.enable_tensor_fusion
            and isinstance(optimizer, _ShardOptimizer)
            and use_pir_api()
        ):
            assert isinstance(optimizer._shard_fn, ShardingStage1), (
                "The shard_fn should be ShardingStage1 "
                "when stage1 tensor fusion is enabled."
            )
            if isinstance(optimizer._shard_fn, ShardingStage1):
                shard_fn = optimizer._shard_fn
                inner_opt = optimizer._inner_opt
                optimizer = ShardingOptimizerStage1(
                    inner_opt, shard_fn, self._inner_strategy
                )
            else:
                logging.warning(
                    "Sharding tensor fusion only support ShardingStage1 now."
                )

        self._engine = Engine(
            layer, loss, optimizer, metrics, strategy=self._inner_strategy
        )
        self._mode = None
        self._feed_name_list = {}

        # convert dygraph model to static model
        if input_spec is not None:
            self._engine._inputs_spec = input_spec[0]
            self._engine._labels_spec = input_spec[1]
        elif isinstance(loader, ShardDataloader):
            (
                self._engine._inputs_spec,
                self._engine._labels_spec,
            ) = self._engine._prepare_data_spec_from_dataloader(loader)
        else:
            batch_size = loader.batch_sampler.batch_size
            (
                self._engine._inputs_spec,
                self._engine._labels_spec,
            ) = self._engine._prepare_data_spec(
                loader.dataset, None, batch_size
            )

        # paddle.enable_static() will be called implicitly in self._engine.prepare.
        # call paddle.disable_static to keep the outside of DistModel in dynamic graph mode

        # set the default mode
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]
        if (
            not self._in_pir_mode
        ):  # TODO (2024-Q2) remove this when pir mode is fully constructed.
            if optimizer is not None and loss is not None:
                self.train()
            elif loss is not None:
                self.eval()
            else:
                self.predict()

    def train(self) -> None:
        """
        Set the DistModel to "train" mode. In "train" mode,
        executing ``__call__`` method will update the
        parameters of the model and return the loss.
        """
        if not self._engine._has_prepared["train"]:
            self._engine._prepare_program(mode="train", init_parameters=False)

        self._mode = "train"
        self._engine.to_mode("train")
        paddle.disable_static()

    def eval(self) -> None:
        """
        Set the mode of DistModel to "eval". In "eval" mode,
        executing ``__call__`` will return the loss.
        """
        if not self._engine._has_prepared["eval"]:
            self._engine._prepare_program(mode="eval", init_parameters=False)

        self._mode = "eval"
        self._engine.to_mode("eval")
        paddle.disable_static()

    def predict(self) -> None:
        """
        Set the mode of DistModel to "predict". In "predict" mode,
        executing ``__call__`` returns a dict that contains the
        outputs of the model.
        """
        if not self._engine._has_prepared["predict"]:
            self._engine.prepare(
                copy.deepcopy(self._engine._inputs_spec),
                None,
                mode="predict",
                init_parameters=False,
            )

        self._mode = "predict"
        self._engine.to_mode("predict")
        paddle.disable_static()

    def __validate_mode(self, mode):
        if mode is None and self._mode is None:
            raise ValueError(
                "Please set the mode or call train()/eval()/predict() first."
            )
        if mode is None:
            mode = self._mode
        if mode not in ["train", "eval", "predict"]:
            raise ValueError("mode can only be 'train', 'eval' or 'predict'.")
        return mode

    def dist_main_program(self, mode: _Mode | None = None) -> Program:
        """
        Get the distributed main program of the specified ``mode``. Each
        'mode' has its own distributed main program, ``dist_main_program``
        returns the corresponding distributed main program of ``mode``.

        Args:
            mode (str|None, optional): Can be 'train' , 'eval' , 'predict' or None.
                'train' : Return the distributed main program for training.
                'eval' : Return the distributed main program for evaluation.
                'predict' : Return the distributed main program for prediction.
                None : The current mode of the DistModel will be used.
                Default : None.

        Returns:
            The distributed main program of ``mode``.
        """
        mode = self.__validate_mode(mode)
        return self._engine.get_dist_main_program(mode)

    def dist_startup_program(self, mode: _Mode | None = None) -> Program:
        """
        Get the corresponding distributed startup program of ``mode``,
        which is used for initializing the parameters.

        Args:
            mode (str|None, optional): Can be 'train' , 'eval' , 'predict' or None.
                'train' : Return the distributed startup program for training.
                'eval' : Return the distributed startup program for evaluation.
                'predict' : Return the distributed startup program for prediction.
                None: The current mode of the DistModel will be used.
                Default : None.

        Returns:
            The distributed startup program of ``mode``.
        """
        mode = self.__validate_mode(mode)
        return self._engine.get_dist_startup_program(mode)

    def serial_main_program(self, mode: _Mode | None = None) -> Program:
        """
        Get the corresponding serial main program of ``mode``, containing
        the whole variables and operators of the given ``layer``.

        Args:
            mode (str|None, optional): Can be 'train', 'eval', 'predict' or None.
                'train' : Return the main program for training.
                'eval' : Return the main program for evaluation.
                'predict' : Return the main program for prediction.
                None : The current mode of the DistModel will be used.
                Default : None.

        Returns:
            The serial main program of ``mode``.
        """
        mode = self.__validate_mode(mode)
        return self._engine.get_serial_main_program(mode)

    def serial_startup_program(self, mode: _Mode | None = None) -> Program:
        """
        Get the corresponding serial startup program of ``mode``.

        Args:
            mode (str|None, optional): Can be 'train' , 'eval' , 'predict' or None.
                'train' : Return the serial startup program for training.
                'eval' : Return the serial startup program for evaluation.
                'predict' : Return the serial startup program for prediction.
                None : The current mode of the DistModel will be used.
                Default : None.

        Returns:
            The serial startup program of ``mode``.
        """
        mode = self.__validate_mode(mode)
        return self._engine.get_serial_startup_program(mode)

    def _make_feeds(self, data_list):
        if (
            self._mode not in self._feed_name_list
            or self._feed_name_list[self._mode] == []
        ):
            self._feed_name_list[self._mode] = self._engine.get_feed_name_list()

        feed_name_list = self._feed_name_list[self._mode]
        if len(feed_name_list) != len(data_list):
            raise ValueError(
                "The input data and feed_list are not consistent."
                f"The model takes {feed_name_list} as input"
            )

        feed_list = []
        no_data_ids = []
        # If the feed_var is None, its feed_name should be deleted.
        # This scenario is very common if using `PipeLine Parallelism`.
        for idx, data in enumerate(data_list):
            if isinstance(data, paddle.Tensor):
                feed_var = _to_lodtensor(data)
                if feed_var is None:
                    no_data_ids.append(idx)
                else:
                    feed_list.append(feed_var)
            else:
                feed_list.append(data)
        feed_name_list_with_data = []
        for idx, feed_name in enumerate(feed_name_list):
            if idx not in no_data_ids:
                feed_name_list_with_data.append(feed_name)
        return dict(zip(feed_name_list_with_data, feed_list))

    def __convert_strategy(self, strategy):
        import copy

        if strategy is None:
            return None
        inner_strategy = auto_strategy.Strategy()
        category = auto_strategy.constants.BASE
        base_config = auto_strategy.constants.get_category_default_config(
            category
        )
        for key in base_config.keys():
            setattr(inner_strategy, key, getattr(strategy, key))
        inner_strategy.fused_passes.enable = strategy.fused_passes.enable
        if getattr(strategy.fused_passes, "gemm_epilogue", False):
            inner_strategy.fused_passes.fused_passes_list.append(
                "fused_gemm_epilogue_pass"
            )
        if getattr(strategy.fused_passes, "dropout_add", False):
            inner_strategy.fused_passes.fused_passes_list.append(
                "fused_dropout_add_pass"
            )

        inner_strategy.amp = copy.deepcopy(strategy.amp)
        inner_strategy.sharding = copy.deepcopy(strategy.sharding)
        inner_strategy.gradient_merge = copy.deepcopy(strategy.gradient_merge)
        inner_strategy.pipeline = copy.deepcopy(strategy.pipeline)
        # The below are template interfaces
        if hasattr(strategy, "_recompute"):
            inner_strategy.recompute = copy.deepcopy(strategy._recompute)

        if hasattr(strategy, "_mp_optimization"):
            inner_strategy.mp_optimization = copy.deepcopy(
                strategy._mp_optimization
            )
        if hasattr(strategy, "_dp_optimization"):
            inner_strategy.dp_optimization = copy.deepcopy(
                strategy._dp_optimization
            )
        if hasattr(strategy, "_sp_optimization"):
            inner_strategy.sp_optimization = copy.deepcopy(
                strategy._sp_optimization
            )

        return inner_strategy

    @switch_to_static_graph
    def __call__(self, *args: Sequence[Any] | Tensor) -> Any:
        if self._mode is None:
            raise ValueError("Please call train()/eval()/predict() first.")
        if self._mode == "train":
            if self._engine._optimizer is None or self._engine._loss is None:
                raise ValueError(
                    "Please set optimizer and loss function before training."
                )
        if self._mode == "eval":
            if self._engine._loss is None:
                raise ValueError("Please set loss function before evaluation.")

        feed_list = []
        for feed_item in list(args):
            if isinstance(feed_item, (list, tuple)):
                feed_list += list(feed_item)
            elif isinstance(feed_item, (paddle.Tensor, core.DenseTensor)):
                feed_list += [feed_item]
            else:
                raise TypeError(
                    f"The inputs of DistModel should be list or tensor, but got {type(feed_item)}"
                )

        feeds = self._make_feeds(feed_list)
        outs = self._engine.run(feeds)
        self.outs = outs

        if self._mode == "predict":
            if "outputs" in self.outs:
                return self.outs["outputs"]
            else:
                return None
        else:
            if "loss" in self.outs:
                return self.outs["loss"]
            else:
                return None

    def _fetch_value(self, value, name=None):
        """
        Get the value of the variable with the given name.

        Args:
            value (pir.Value): The pir Value to fetch.
            name (str|None, optional): The user-defined name of
                the fetched result. If None, the order of the Value
                in the fetch list will be used. Default: None.
        """
        self._engine._pir_fetch_values.append(value)
        if name is None:
            name = len(self._engine._pir_fetch_values) - 1
        self._engine._pir_user_defined_fetch_names.append(name)

    def state_dict(
        self,
        mode: Literal['opt', 'param', 'all'] = "all",
        split_fusion: bool = True,
    ) -> dict[str, Tensor]:
        """
        Get the state dict of model and optimizer.

        Args:
            mode (str): Can be ['opt', 'param', 'all'],
                'opt' :  The return value only contains the variable in the optimizer.
                'param' : The return value only contains the variable in the network, not the variable in the optimizer.
                'all' : The return value contains the variable in the network and optimizer.
                Default: 'all'
        """
        if use_pir_api():
            scope = paddle.static.global_scope()
            local_state_dict = self.dist_main_program(
                mode=self._engine._mode
            ).state_dict(mode, scope)
        else:
            local_state_dict = self.dist_main_program(
                mode=self._engine._mode
            ).state_dict(mode)

        dist_state_dict = self._build_distributed_state_dict(local_state_dict)

        # The parameters fused in the ffn and qkv fusion pass will be split back into their original, unfused state.
        if self._engine.fused_ffn_qkv is not None and split_fusion:
            with paddle.base.dygraph.guard():
                # Traverse each fusion structure, the key could be ffn or qkv.
                for key, pat_list in self._engine.fused_ffn_qkv.items():
                    # Traverse each fusion pattern dict, such as: fused_p1_p2:[p1, p2].
                    for fusion_map in pat_list:
                        ((fused_param, ori_params_meta),) = fusion_map.items()
                        origin_params = list(dist_state_dict.keys())
                        for param in origin_params:
                            suffix = _get_suffix(param, fused_param)
                            if suffix is not None:
                                value = dist_state_dict[param]
                                assert (
                                    value.is_dist()
                                ), f"key {param} value:{value} is not a dist tensor."
                                mesh = value.process_mesh
                                placements = value.placements
                                if "_pow_acc" in suffix:
                                    out = (value._local_value(),) * len(
                                        ori_params_meta
                                    )
                                else:
                                    if len(ori_params_meta) == 3:
                                        is_qkv = True
                                        num_heads = ori_params_meta[
                                            0
                                        ].local_num_head
                                        num_key_value_heads = ori_params_meta[
                                            1
                                        ].local_num_head
                                    else:
                                        is_qkv = False
                                        num_heads = None
                                        num_key_value_heads = None
                                    out = split_param_func(
                                        value._local_value(),
                                        split_nums=len(ori_params_meta),
                                        is_qkv=is_qkv,
                                        num_heads=num_heads,
                                        num_key_value_heads=num_key_value_heads,
                                    )
                                for i in range(len(ori_params_meta)):
                                    dist_tensor = dtensor_from_local(
                                        out[i], mesh, placements
                                    )
                                    paddle.assign(
                                        out[i], dist_tensor._local_value()
                                    )
                                    dist_state_dict[
                                        ori_params_meta[i].name + suffix
                                    ] = dist_tensor
                                dist_state_dict.pop(param)

        mapping_names = [
            (
                self._parameter_to_structured_name[k]
                if k in self._parameter_to_structured_name
                else k
            )
            for k in dist_state_dict.keys()
        ]
        dist_state_dict = dict(
            zip(mapping_names, list(dist_state_dict.values()))
        )
        return dist_state_dict

    def _build_distributed_state_dict(self, local_state_dict):
        """
        Args:
            local_state_dict(Dict[str, libpaddle.Tensor]): The state dict from program.
        """
        dist_main_program = self.dist_main_program(mode=self._engine._mode)
        if use_pir_api():
            dist_attrs = get_dist_attr(dist_main_program)
        else:
            # Dict[var.name, Dict["process_shape": process_mesh.shape, "process_group": process_mesh.process_ids, "dims_mapping": dims_mapping]]
            dist_attrs = get_dist_attr(
                dist_main_program, self._engine._dist_contexts[self._mode]
            )

        def build_distributed_tensor(local_tensor, dist_attr):
            assert isinstance(
                local_tensor, (paddle.Tensor, np.ndarray, paddle.base.Tensor)
            )
            if not isinstance(local_tensor, paddle.Tensor):
                local_tensor = paddle.Tensor(local_tensor)
            assert isinstance(
                local_tensor, paddle.Tensor
            ), f"local tensor:{local_tensor} type {type(local_tensor)} is not paddle.Tensor."
            assert len(local_tensor.shape) == len(
                dist_attr["dims_mapping"]
            ), f"local tensor shape {local_tensor.shape} not equal to dims_mapping shape {dist_attr['dims_mapping']}."
            global_shape = local_tensor.shape
            mesh = ProcessMesh(
                np.array(dist_attr["process_group"]).reshape(
                    dist_attr["process_shape"]
                ),
                dim_names=dist_attr["dim_names"],
            )
            placements = to_placements(dist_attr["dims_mapping"], mesh)
            dist_tensor = dtensor_from_local(local_tensor, mesh, placements)
            assert (
                dist_tensor._local_value().shape == local_tensor.shape
            ), f"local tensor shape {dist_tensor._local_value().shape} not equal to local_tensor.shape:{local_tensor.shape}"
            paddle.assign(local_tensor, dist_tensor._local_value())
            return dist_tensor

        global_state_dict = {}
        with paddle.base.dygraph.guard():
            for var_name, tensor in local_state_dict.items():
                assert (
                    var_name in dist_attrs
                ), f"var {var_name} not in dist attrs:{dist_attrs}."
                global_state_dict[var_name] = build_distributed_tensor(
                    tensor, dist_attrs[var_name]
                )
        return global_state_dict

    def set_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        local_state_dict = {}
        dist_main_program = self.dist_main_program(mode=self._engine._mode)
        cur_state_dict = self.state_dict(split_fusion=False)
        copy_tensor = False

        # When using the tensor-fusion strategy, model parameters are shared with
        # slice@ parameters. When setting the state_dict, we must copy the tensor
        # instead of changing the handle directly, as this could cause errors in
        # the slice@ parameters and increase memory usage.
        enable_tensor_fusion = (
            self._inner_strategy.sharding.enable_tensor_fusion
            if self._inner_strategy
            else False
        )
        if self._engine._optimizer is not None and enable_tensor_fusion:
            copy_tensor = True

        for k, v in state_dict.items():
            assert v.is_dist(), f"key {k} value:{v} is not a dist tensor."
            if k in cur_state_dict:
                cur_v = cur_state_dict[k]
                assert v.process_mesh == cur_state_dict[
                    k
                ].process_mesh or check_placements_equal(
                    v.placements, cur_v.placements
                ), f"process_mesh:{v.process_mesh} != {cur_v.process_mesh} or placements:{v.placements} != {cur_v.placements} not match"
            param_name = (
                self._structured_to_parameter_name[k]
                if k in self._structured_to_parameter_name
                else k
            )
            local_state_dict[param_name] = _to_lodtensor(v._local_value())

        # The structure of ffn and qkv in the network has been fused, and the unfused parameters in the original state_dict are fused.
        if self._engine.fused_ffn_qkv is not None:
            with paddle.base.dygraph.guard():
                # Traverse each fusion structure, the key could be ffn or qkv.
                for key, pat_list in self._engine.fused_ffn_qkv.items():
                    # Traverse each fusion pattern dict, such as: fused_p1_p2:[p1, p2].
                    for fusion_map in pat_list:
                        ((fused_param, ori_params_meta),) = fusion_map.items()
                        # Obtain all the parameters to be fused, differentiated by suffixes, such as: beta1_pow_acc_0, _fp32_master_0_moment1_0.
                        suffix_names = []
                        for k, v in local_state_dict.items():
                            suffix = _get_suffix(ori_params_meta[0].name, k)
                            if suffix is not None:
                                suffix_names.append(suffix)
                        if len(suffix_names) == 0:
                            continue
                        # Traverse through each parameter for fusion, insert the fused parameters, and delete the pre-fusion parameters.
                        for suffix in suffix_names:
                            concat_tensors = []
                            for ori_p in ori_params_meta:
                                if ori_p.name + suffix not in local_state_dict:
                                    warnings.warn(
                                        f"{ori_p.name + suffix} is not in state_dict."
                                    )
                                    break
                                else:
                                    concat_tensors.append(
                                        local_state_dict[ori_p.name + suffix]
                                    )
                            if len(concat_tensors) == len(ori_params_meta):
                                if "_pow_acc" in suffix:
                                    fused_w = concat_tensors[0]
                                else:
                                    if len(ori_params_meta) == 3:
                                        is_qkv = True
                                        num_heads = ori_params_meta[
                                            0
                                        ].local_num_head
                                        num_key_value_heads = ori_params_meta[
                                            1
                                        ].local_num_head
                                    else:
                                        is_qkv = False
                                        num_heads = None
                                        num_key_value_heads = None
                                    fused_w = fuse_param_func(
                                        concat_tensors,
                                        is_qkv=is_qkv,
                                        num_heads=num_heads,
                                        num_key_value_heads=num_key_value_heads,
                                    )

                                local_state_dict[fused_param + suffix] = (
                                    _to_lodtensor(fused_w)
                                )
                                for ori_p in ori_params_meta:
                                    local_state_dict.pop(ori_p + suffix)

        if use_pir_api():
            dist_main_program.set_state_dict(
                local_state_dict, paddle.static.global_scope(), copy_tensor
            )
        else:
            dist_main_program.set_state_dict(
                local_state_dict, paddle.static.global_scope()
            )

    def _get_shard_stage1_optimizer(self):
        optimizer = self._engine._optimizer
        if optimizer is None:
            return optimizer

        if isinstance(
            optimizer,
            paddle.static.amp.decorator.OptimizerWithMixedPrecision,
        ):
            optimizer = optimizer._optimizer

        assert isinstance(
            optimizer, ShardingOptimizerStage1
        ), "The optimizer should be ShardingOptimizerStage1 when stage1 tensor fusion is enabled."

        return optimizer

    def _convert_state_dict_tensor_fusion(self, state_dict, optimizer_function):
        enable_tensor_fusion = (
            self._inner_strategy.sharding.enable_tensor_fusion
            if self._inner_strategy
            else False
        )

        assert (
            enable_tensor_fusion
        ), "Can only convert state_dict when tensor fusion is enabled."
        optimizer = self._get_shard_stage1_optimizer()
        assert optimizer is not None, "The optimizer should not be None."

        parameter_names = [
            (
                self._structured_to_parameter_name[k]
                if k in self._structured_to_parameter_name
                else k
            )
            for k in state_dict.keys()
        ]
        state_dict = dict(zip(parameter_names, list(state_dict.values())))

        optimizer_function(optimizer, state_dict)

        structured_names = [
            (
                self._parameter_to_structured_name[k]
                if k in self._parameter_to_structured_name
                else k
            )
            for k in state_dict.keys()
        ]
        state_dict = dict(zip(structured_names, list(state_dict.values())))

        return state_dict

    def _convert_state_dict_with_rank_unique_name(self, state_dict):
        def optimizer_function(optimizer, state_dict):
            optimizer.convert_state_dict_with_rank_unique_name(state_dict)

        return self._convert_state_dict_tensor_fusion(
            state_dict, optimizer_function
        )

    def _convert_state_dict_without_tensor_fusion_param(self, state_dict):
        def optimizer_function(optimizer, state_dict):
            optimizer.convert_state_dict_without_tensor_fusion_param(state_dict)

        return self._convert_state_dict_tensor_fusion(
            state_dict, optimizer_function
        )

    def _convert_state_dict_with_tensor_fusion_param(self, state_dict):
        def optimizer_function(optimizer, state_dict):
            optimizer.convert_state_dict_with_tensor_fusion_param(state_dict)

        return self._convert_state_dict_tensor_fusion(
            state_dict, optimizer_function
        )

    def _convert_state_dict_with_origin_name(self, state_dict):
        def optimizer_function(optimizer, state_dict):
            optimizer.convert_state_dict_with_origin_name(state_dict)

        return self._convert_state_dict_tensor_fusion(
            state_dict, optimizer_function
        )


def to_static(
    layer: Layer,
    loader: ShardDataloader | DataLoader | None = None,
    loss: Layer | Callable[..., Any] | None = None,
    optimizer: Optimizer | None = None,
    strategy: Strategy | None = None,
    input_spec: list[list[DistributedInputSpec]] | None = None,
) -> DistModel:
    """
    Converts the ``layer`` with distributed tensor (constructed from
    ``paddle.distributed.shard_tensor``) to a static graph. ``to_static``
    returns a DistModel instance containing the static graph for
    distributed training, evaluation and prediction.

    Args:
        layer(paddle.nn.Layer): The layer in dygraph mode, the parameters
            or its inputs can be distributed tensors.
        loader(ShardDataloader|paddle.io.DataLoader): The data loader used in dygraph mode,
            used to infer inputs_spec and labels_spec.
        loss(Loss|Callable|None, optional): The loss function for training
            or evaluating the model. Can be a `paddle.nn.Layer` instance or
            any callable function. Default: None.
        optimizer(paddle.optimizer.Optimizer|_ShardOptimizer|None, optional):
            The optimizer for training. It can `paddle.optimizer.Optimizer`
            or `_ShardOptimizer` wrapped by `shard_optimizer`. Default: None.
        strategy(paddle.distributed.Strategy|None, optional): Configs for
            parallel strategies and optimization settings (e.g. sharding,
            pipeline parallelism). Default: None.
        input_spec(list[list[paddle.distributed.DistributedInputSpec]]|None, optional):
            The custom input specs specify the shape, dtype, and name information
            of model inputs and labels. If it is not None, the input specs and
            label specs will be inferred from the custom input specs. The custom
            input specs should be a list containing two sublists: the first
            sublist represents theinput specs, and the second sublist represents
            the label specs. Default: None.

    Returns:
        DistModel: A ``DistModel`` instance converted the input ``layer``.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle import nn
            >>> from paddle.distributed import Replicate, Shard

            >>> BATCH_SIZE = 4
            >>> BATCH_NUM = 4
            >>> IMAGE_SIZE = 16
            >>> CLASS_NUM = 8
            >>> class RandomDataset(paddle.io.Dataset): # type: ignore[type-arg]
            ...     def __init__(self, images, labels, num_samples):
            ...         self.images = images
            ...         self.labels = labels
            ...         self.num_samples = num_samples
            ...     def __getitem__(self, idx):
            ...         return self.images[idx], self.labels[idx]
            ...     def __len__(self):
            ...         return self.num_samples

            >>> class DemoNet(nn.Layer):
            ...     def __init__(self, mesh):
            ...         super().__init__()
            ...         self._mesh = mesh
            ...         self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE)
            ...         self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM)
            ...         self.relu = nn.ReLU()
            ...         # shard the weights of this layer
            ...         self.linear_0.weight = dist.shard_tensor(
            ...             self.linear_0.weight,
            ...             self._mesh,
            ...             [Shard(1)],
            ...             stop_gradient=False,
            ...         )
            ...         self.linear_1.weight = dist.shard_tensor(
            ...             self.linear_1.weight,
            ...             self._mesh,
            ...             [Shard(0)],
            ...             stop_gradient=False,
            ...         )
            ...     def forward(self, x):
            ...         out = self.linear_0(x)
            ...         out = self.relu(out)
            ...         out = self.linear_1(out)
            ...         return out

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
            >>> labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
            >>> dataset = RandomDataset(images, labels, BATCH_SIZE)
            >>> loader = paddle.io.DataLoader(dataset, batch_size=BATCH_SIZE)

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> layer = DemoNet(mesh)
            >>> opt = paddle.optimizer.SGD(
            ...     learning_rate=0.1, parameters=layer.parameters()
            ... )
            >>> loss_fn = nn.MSELoss()
            >>> dist_loader = dist.shard_dataloader(loader, meshes=[mesh])
            >>> dist_model = dist.to_static(
            ...     layer, dist_loader, loss_fn, opt
            ... )
            >>> # training
            >>> dist_model.train()
            >>> for batch_id, (image, label) in enumerate(dist_loader()):
            ...     # in train mode, executing the __call__ method will
            ...     # update the parameters of the model and return the
            ...     # loss
            ...     loss = dist_model(image, label)

            >>> # evaluation
            >>> dist_model.eval()
            >>> for batch_id, (image, label) in enumerate(dist_loader()):
            ...     # in eval mode, executing the __call__ method will
            ...     # return the loss
            ...     loss = dist_model(image, label)

            >>> # prediction
            >>> dist_model.predict()
            >>> for batch_id, (image, label) in enumerate(dist_loader()):
            ...     # in predict mode, executing the __call__ method will
            ...     # return a dict that contains the outputs of the model,
            ...     # where the value of "out0" is the first output.
            ...     outs = dist_model(image)

            >>> # This case need to be executed in multi-card environment
            >>> # export CUDA_VISIBLE_DEVICES=0,1
            >>> # python -m paddle.distributed.launch {test_case}.py
    """
    if isinstance(optimizer, _ShardOptimizer) and not use_pir_api():
        shard_fn = optimizer._shard_fn
        sharding_degree = optimizer._sharding_degree
        optimizer = optimizer._inner_opt

        if shard_fn is not None:
            strategy = dist.Strategy() if strategy is None else strategy

            # Deduce sharding degree for static
            # Note: Because limitation of architecture, we need to ensure that
            # all parameters are sharded by the same mesh axis
            assert (
                sharding_degree is not None
            ), "Sharding degree can not be None."

            if isinstance(shard_fn, ShardingStage1):
                strategy.sharding.enable = True
                strategy.sharding.stage = 1
                strategy.sharding.degree = sharding_degree
            elif isinstance(shard_fn, ShardingStage2):
                strategy.sharding.enable = True
                strategy.sharding.stage = 2
                strategy.sharding.degree = sharding_degree
            elif isinstance(shard_fn, ShardingStage3):
                strategy.sharding.enable = True
                strategy.sharding.stage = 3
                strategy.sharding.degree = sharding_degree
                for param in optimizer._parameter_list:
                    shard_fn._unshard_parameter(param)
            else:
                raise NotImplementedError(
                    "Only sharding stage 1, 2 and 3 can to_static for now. User-defined shard_fn will be supported later."
                )
    if strategy is None or strategy.full_graph:
        dist_model = DistModel(
            layer, loader, loss, optimizer, strategy, input_spec=input_spec
        )
        return dist_model
    else:
        layer = paddle.jit.to_static(layer, full_graph=False)
        return layer


def unshard_dtensor(dist_tensor: Tensor) -> Tensor:
    """
    Converts a distributed tensor to a dense tensor. ``unshard_dtensor``
    first make the ``dist_tensor`` be ``Replicate`` state on all processes and
    then converts it to a dense ``paddle.Tensor``. It can be treated as a
    reverse operation of ``shard_tensor``.

    Args:
        dist_tensor (paddle.Tensor): The distributed tensor which is constructed
            from a dense tensor with ``shard_tensor``.

    Returns:
        paddle.Tensor: The original dense tensor of the input ``dist_tensor``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle.distributed import Replicate, Shard

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> original_tensor = paddle.rand([4, 1024, 512])
            >>> dist_tensor = dist.shard_tensor(original_tensor, mesh, [Shard(0)])
            >>> # dense_tensor's shape is the same as original_tensor
            >>> dense_tensor = dist.unshard_dtensor(dist_tensor)
    """
    if paddle.in_dynamic_mode():
        # if the input is not a distributed
        # tensor, return it directly
        if dist_tensor.is_dist() is False:
            raise ValueError("The input should be a distributed tensor.")

        mesh = dist_tensor.process_mesh
        placements = dist_tensor.placements
        replicate_placements = [dist.Replicate()] * len(placements)
        r_dist_tensor = reshard(dist_tensor, mesh, replicate_placements)

        if isinstance(dist_tensor, EagerParamBase):
            return EagerParamBase.from_tensor(
                r_dist_tensor._local_value(),
                **dist_tensor.__dict__,
            )
        else:
            return paddle.Tensor(r_dist_tensor._local_value())

    elif paddle.framework.in_pir_mode():
        # in pir mode, we define the logic of unshard_tensor as dist_tensor_type --> dense_tensor_type with global shape.
        dense_tensor_type = paddle.pir.create_shaped_type(
            dist_tensor.type(), dist_tensor.shape
        )
        dist_tensor.set_type(dense_tensor_type)

        return dist_tensor

    else:
        raise NotImplementedError(
            "`unshard_dtensor()` only supported in dynamic and pir mode."
        )


class ShardDataloader:
    """
    ShardDataloader converts a dataloader to a new dataloader which provided two capabilities:
    1. split dataloader by shard_dim to do data parallel.
    2. reshard the output of dataloader to distributed tensor.
    if is_dataset_splitted is True, just need to do reshard.

    Args:
        dataloader (paddle.io.DataLoader): The dataloader to be sharded.
        meshes (ProcessMesh|list[ProcessMesh]|tuple[ProcessMesh]): The mesh list of the dataloader.
            Identify which mesh the input is on. if len(meshes) == 1 or type(meshes) == ProcessMesh,
            all the inputs are on the same mesh.
        input_keys (list[str]|tuple[str]): if the iteration result of dataloader is a dict of tensors,
            input_keys is the keys of this dict, identify which tensor is located on which mesh,
            one-to-one correspondence with meshes. i.e. dict[input_keys[i]] is on meshes[i].
            Default: None, which means the outputs is a list, and the i'th input is on meshes[i].
        shard_dims (list|tuple|str|int]): The mesh dimension to shard the dataloader.
            Users can specify the shard_dim of each mesh or specify a single shard_dim for all meshes.
            Default: None, which means the data loader will not be split, i.e. mp.
        is_dataset_splitted (bool): Whether the dataset has been split.
        dense_tensor_idx (list): A paired 2D list specifies the index of the dense_tensor in the output of dataloader.
            It allows users to identify which elements within each output batch are dense_tensor.
            first dense_tensor: the dense_tensor return by dataloader.
            second dense_tensor: num_or_sections specifies how to split first tensor: evenly (if a number) or unevenly (if a list).
            Default: None, meaning all outputs are dist_tensors.
            Note: For dense_tensor_idx settings, the idx must be paired.
    """

    def __init__(
        self,
        dataloader: paddle.io.DataLoader,
        meshes: ProcessMesh | list[ProcessMesh] | tuple[ProcessMesh],
        input_keys: list[str] | tuple[str] | None = None,
        shard_dims: list | tuple | str | int | None = None,
        is_dataset_splitted: bool = False,
        dense_tensor_idx: list[list[int]] | None = None,
    ):
        # do some check
        if is_dataset_splitted is True and shard_dims is None:
            raise ValueError(
                "shard_dims must be set when is_dataset_splitted is True"
            )

        self._meshes = to_list(meshes)
        if self._meshes is None or len(self._meshes) == 0:
            raise ValueError("meshes must be set")

        process_id = dist.get_rank()
        if self._process_id_in_multi_meshes(process_id):
            raise ValueError(
                f"process_id {process_id} is in more than one mesh, the meshes are {self._meshes}"
            )

        self._all_inputs_in_one_mesh = len(self._meshes) == 1
        self._input_keys = input_keys
        self._shard_dims = self._process_shard_dims(shard_dims)

        mesh, shard_dim = self._get_mesh_and_shard_dim(process_id)
        if mesh is None:
            mesh = to_list(self._meshes[0])[0]
            shard_dim = to_list(self._shard_dims[0])[0]
            dp_rank = 0
            dp_world_size = mesh.get_dim_size(shard_dim)
        else:
            dp_rank = mesh.get_rank_by_dim_and_process_id(shard_dim, process_id)
            dp_world_size = mesh.get_dim_size(shard_dim)

        if is_dataset_splitted is True or shard_dims is None:
            self._dataloader = dataloader
            self.batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader.batch_sampler, DistributedBatchSampler):
            self.batch_size = dataloader.batch_sampler.batch_size
            self.batch_sampler = dataloader.batch_sampler
            self._dataloader = dataloader
        else:
            self.batch_size = int(
                dataloader.batch_sampler.batch_size / dp_world_size
            )
            if isinstance(dataloader.batch_sampler, _InfiniteIterableSampler):
                shuffle = False
                drop_last = False
            else:
                shuffle = dataloader.batch_sampler.shuffle
                drop_last = dataloader.batch_sampler.drop_last
            self.batch_sampler = DistributedBatchSampler(
                dataset=dataloader.dataset,
                batch_size=self.batch_size,
                num_replicas=dp_world_size,
                rank=dp_rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            self.batch_sampler._acc_steps = dataloader.batch_sampler._acc_steps
            self._dataloader = paddle.io.DataLoader(
                dataset=dataloader.dataset,
                batch_sampler=self.batch_sampler,
                feed_list=dataloader.feed_list,
                places=dataloader.places,
                return_list=dataloader.return_list,
                collate_fn=dataloader.collate_fn,
                num_workers=dataloader.num_workers,
                use_buffer_reader=dataloader.use_buffer_reader,
                prefetch_factor=dataloader.prefetch_factor,
                use_shared_memory=dataloader.use_shared_memory,
                timeout=dataloader.timeout,
                worker_init_fn=dataloader.worker_init_fn,
                persistent_workers=dataloader._persistent_workers,
            )
        # Note(lizhiyu): In dygraph mode, the flag "pin_memory" is default "True", but it decrease the speed of `AutoParallel`
        self._dataloader.pin_memory = False
        self.iter = None
        self.dense_tensor_idx = dense_tensor_idx

    def _process_shard_dims(self, shard_dims):
        if isinstance(shard_dims, (int, str)) or shard_dims is None:
            res = []
            for i in range(len(self._meshes)):
                if isinstance(self._meshes[i], (list, tuple)):
                    res.append([shard_dims] * len(self._meshes[i]))
                else:
                    res.append(shard_dims)
            return res
        else:
            if len(shard_dims) != len(self._meshes):
                raise ValueError(
                    f"shard_dims must be the same length as meshes, but got {len(shard_dims)} != {len(self._meshes)}"
                )
            return shard_dims

    def _get_mesh_and_shard_dim(self, process_id):
        for i in range(len(self._meshes)):
            if isinstance(self._meshes[i], (list, tuple)):
                for j in range(len(self._meshes[i])):
                    if process_id in self._meshes[i][j]._process_ids:
                        return self._meshes[i][j], self._shard_dims[i][j]
            else:
                if process_id in self._meshes[i]._process_ids:
                    return self._meshes[i], self._shard_dims[i]
        return None, None

    def _process_id_in_multi_meshes(self, process_id):
        count = 0
        flatten_meshes = []
        for mesh in self._meshes:
            if isinstance(mesh, (list, tuple)):
                flatten_meshes.extend(mesh)
            else:
                flatten_meshes.append(mesh)

        # NOTE(zhengzhonghui): User may set the same mesh for different inputs, so we need to unique the meshes
        unique_meshes = list(set(flatten_meshes))
        for mesh in unique_meshes:
            if process_id in mesh._process_ids:
                count += 1
        return count > 1

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self

    def _get_mesh_and_placement(self, index):
        shard_dim = (
            self._shard_dims[0]
            if self._all_inputs_in_one_mesh
            else self._shard_dims[index]
        )
        if shard_dim is not None and not in_auto_dp_mode():
            placements = [dist.Shard(0)]
        else:
            placements = [dist.Replicate()]
        mesh = (
            self._meshes[0]
            if self._all_inputs_in_one_mesh
            else self._meshes[index]
        )
        for _ in range(1, len(mesh._shape)):
            placements.append(dist.Replicate())
        return mesh, placements

    def _get_meshes_and_placements_for_list_input(self, index, length):
        if self._all_inputs_in_one_mesh:
            meshes = [self._meshes[0]] * length
            shard_dims = [self._shard_dims[0]] * length
        else:
            meshes = self._meshes[index]
            if isinstance(meshes, (list, tuple)):
                assert len(meshes) == length
            else:
                meshes = [meshes] * length
            shard_dims = self._shard_dims[index]
            if isinstance(shard_dims, (list, tuple)):
                assert len(shard_dims) == length
            else:
                shard_dims = [shard_dims] * length

        placements = []
        for i in range(length):
            if shard_dims[i] is not None and not in_auto_dp_mode():
                placement = [dist.Shard(0)]
            else:
                placement = [dist.Replicate()]
            for _ in range(1, len(meshes[i]._shape)):
                placement.append(dist.Replicate())
            placements.append(placement)
        return meshes, placements

    def _dtensors_from_list_input(
        self, list_tensors, meshes, placements, dense_tensor_idx=None
    ):
        dist_data = []
        for j in range(len(list_tensors)):
            if dense_tensor_idx is not None and j in dense_tensor_idx:
                dist_data.append(list_tensors[j])
            else:
                dist_data.append(
                    dtensor_from_local(
                        list_tensors[j], meshes[j], placements[j]
                    )
                )
        return dist_data

    def _get_batch(self, batch_data):
        if isinstance(batch_data, (list, tuple)):
            if self._all_inputs_in_one_mesh is False:
                assert len(batch_data) == len(self._meshes)
            dist_batch_data = []
            for i in range(len(batch_data)):
                input_data = batch_data[i]
                if isinstance(input_data, (list, tuple)):
                    (
                        meshes,
                        placements,
                    ) = self._get_meshes_and_placements_for_list_input(
                        i, len(input_data)
                    )
                    _dense_tensor_idx = (
                        None
                        if self.dense_tensor_idx is None
                        else self.dense_tensor_idx[i]
                    )
                    dist_batch_data.append(
                        self._dtensors_from_list_input(
                            input_data, meshes, placements, _dense_tensor_idx
                        )
                    )
                elif isinstance(input_data, paddle.Tensor):
                    if (
                        self.dense_tensor_idx is not None
                        and self.dense_tensor_idx[i] != []
                    ):
                        dist_batch_data.append(input_data)
                    else:
                        mesh, placements = self._get_mesh_and_placement(i)
                        dist_batch_data.append(
                            dtensor_from_local(input_data, mesh, placements)
                        )
                else:
                    raise ValueError(
                        f"Unsupported input_data type {type(input_data)}"
                    )
            return dist_batch_data
        elif isinstance(batch_data, dict):
            input_keys = (
                batch_data.keys()
                if self._input_keys is None
                else self._input_keys
            )
            if self._all_inputs_in_one_mesh is False:
                assert len(input_keys) == len(self._meshes)
            dist_batch_data = {}
            for i, key in enumerate(input_keys):
                input_data = batch_data[key]
                if isinstance(input_data, (list, tuple)):
                    (
                        meshes,
                        placements,
                    ) = self._get_meshes_and_placements_for_list_input(
                        i, len(input_data)
                    )
                    _dense_tensor_idx = (
                        None
                        if self.dense_tensor_idx is None
                        else self.dense_tensor_idx[i]
                    )
                    dist_batch_data[key] = self._dtensors_from_list_input(
                        input_data, meshes, placements, _dense_tensor_idx
                    )
                elif isinstance(input_data, paddle.Tensor):
                    if (
                        self.dense_tensor_idx is not None
                        and self.dense_tensor_idx[i] != []
                    ):
                        dist_batch_data.append(input_data)
                    else:
                        mesh, placements = self._get_mesh_and_placement(i)
                        dist_batch_data[key] = dtensor_from_local(
                            batch_data[key], mesh, placements
                        )
                else:
                    raise ValueError(
                        f"Unsupported input_data type {type(input_data)}"
                    )
            return dist_batch_data
        elif isinstance(batch_data, paddle.Tensor):
            mesh, placements = self._get_mesh_and_placement(0)
            return dtensor_from_local(batch_data, mesh, placements)
        else:
            raise ValueError(f"Unsupported batch_data type {type(batch_data)}")

    def __next__(self):
        if self.iter is None:
            self.iter = self._dataloader.__iter__()
        batch_data = next(self.iter)
        return self._get_batch(batch_data)

    def __call__(self):
        self.iter = self._dataloader.__iter__()
        return self


def shard_dataloader(
    dataloader: DataLoader,
    meshes: ProcessMesh | Sequence[ProcessMesh],
    input_keys: Sequence[str] | None = None,
    shard_dims: Sequence[str] | Sequence[int] | str | int | None = None,
    is_dataset_splitted: bool = False,
    dense_tensor_idx: list[list[int]] | None = None,
) -> ShardDataloader:
    """
    Convert the dataloader to a ShardDataloader which provided two capabilities:
    1. split dataloader by shard_dim to do data parallel if it it not None.
    2. reshard the output of dataloader to distributed tensor.
    if is_dataset_splitted is True, it means that the dataset has been split by users, and just need to do reshard.
    only if is_dataset_splitted is False and shard_dims is not None, it will do split.

    Args:
        dataloader (paddle.io.DataLoader): The dataloader to be sharded. the output of dataloader
            must be a list or dict of paddle.Tensor with 2 elements, i.e. [input_data, label] or
            {"input_data": input_data, "label": label}, input_data and label can be a list to support multiple inputs.
        meshes (ProcessMesh|list[ProcessMesh]|tuple[ProcessMesh]): The mesh list of the dataloader.
            Identify which mesh the input is on. if len(meshes) == 1 or type(meshes) == ProcessMesh,
            all the inputs are on the same mesh.
        input_keys (list[str]|tuple[str]): if the iteration result of dataloader is a dict of tensors,
            input_keys is the keys of this dict, identify which tensor is located on which mesh,
            one-to-one correspondence with meshes. i.e. dict[input_keys[i]] is on meshes[i].
            Default: None, which means the outputs is a list, and the i'th input is on meshes[i].
        shard_dims (list(str)|tuple(str)|list(int)|tuple(int)|str|int]):
            The mesh dimension to shard the dataloader.
            Users can specify the shard_dim of each mesh or specify a single shard_dim for all meshes.
            Default: None, which means the data loader will not be split, i.e. mp.
        is_dataset_splitted (bool): Whether the dataset has been split, Default: False.
        dense_tensor_idx (list): A paired 2D list specifies the index of the dense_tensor in the output of dataloader.
            It allows users to identify which elements within each output batch are dense_tensor.
            first dense_tensor: the dense_tensor return by dataloader.
            second dense_tensor: num_or_sections specifies how to split first tensor: evenly (if a number) or unevenly (if a list).
            Default: None, meaning all outputs are dist_tensors.
            Note: For dense_tensor_idx settings, the idx must be paired.
    Returns:
        ShardDataloader: The sharded dataloader.

    Examples:
        .. code-block:: python
            :name: example-1

            >>> import os
            >>> import numpy as np
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle.io import BatchSampler, DataLoader, Dataset

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
            >>> mesh1 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=['x', 'y'])

            >>> paddle.seed(1024)
            >>> np.random.seed(1024)
            >>> class RandomDataset(Dataset): # type: ignore[type-arg]
            >>>     def __init__(self, seq_len, hidden, num_samples=8):
            ...         super().__init__()
            ...         self.seq_len = seq_len
            ...         self.hidden = hidden
            ...         self.num_samples = num_samples
            ...         self.inputs = [np.random.uniform(size=[self.seq_len, self.hidden]).astype("float32") for _ in range(num_samples)]
            ...         self.labels = [np.array(index, dtype="float32") for index in range(num_samples)]

            ...     def __getitem__(self, index):
            ...         return self.inputs[index], self.labels[index]

            ...     def __len__(self):
            ...         return self.num_samples

            >>> class MlpModel(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super(MlpModel, self).__init__()
            ...         self.w0 = dist.shard_tensor(
            ...             self.create_parameter(shape=[8, 8]),
            ...             mesh0, [dist.Replicate(), dist.Shard(1)])
            ...         self.w1 = dist.shard_tensor(
            ...             self.create_parameter(shape=[8, 8]),
            ...             mesh1, [dist.Replicate(), dist.Shard(0)])

            ...     def forward(self, x):
            ...         y = paddle.matmul(x, self.w0)
            ...         y = dist.reshard(y, mesh1, [dist.Shard(0), dist.Shard(2)])
            ...         z = paddle.matmul(y, self.w1)
            ...         return z

            >>> model = MlpModel()
            >>> dataset = RandomDataset(4, 8)
            >>> sampler = BatchSampler(
            ...     dataset,
            ...     batch_size=2,
            ... )
            >>> dataloader = DataLoader(
            ...     dataset,
            ...     batch_sampler=sampler,
            ... )
            >>> dist_dataloader = dist.shard_dataloader(
            ...     dataloader=dataloader,
            ...     meshes=[mesh0, mesh1],
            ...     shard_dims="x"
            ... )
            >>> opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
            >>> dist_opt = dist.shard_optimizer(opt)
            >>> def loss_fn(logits, label):
            ...     # logits: [bs, seq_len, hidden], label: [bs]
            ...     loss = paddle.nn.MSELoss(reduction="sum")
            ...     logits = paddle.sum(logits, axis=[1, 2])
            ...     return loss(logits, label)

            >>> RUN_STATIC = eval(os.environ['RUN_STATIC'])
            >>> def run_dynamic():
            ...     for step, (input, label) in enumerate(dist_dataloader()):
            ...         logits = model(input)
            ...         loss = loss_fn(logits, label)
            ...         print("step:{}, loss:{}".format(step, loss))
            ...         loss.backward()
            ...         dist_opt.step()
            ...         dist_opt.clear_grad()

            >>> def run_static():
            ...     dist_model = dist.to_static(
            ...         model, dist_dataloader, loss_fn, opt
            ...     )
            ...     dist_model.train()
            ...     for step, (input, label) in enumerate(dist_dataloader()):
            ...         print("label:", label)
            ...         loss = dist_model(input, label)
            ...         print("step:{}, loss:{}".format(step, loss))

            >>> if RUN_STATIC == 0:
            ...     run_dynamic()
            ... else:
            ...     run_static()

            >>> # This case need to be executed in multi-card environment
            >>> # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            >>> # RUN_STATIC=1 python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" {test_case}.py
            >>> # RUN_STATIC=0 python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" {test_case}.py

        .. code-block:: python
            :name: example-2

            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle.io import BatchSampler, DataLoader, Dataset
            >>> import numpy as np
            >>> mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['dp', 'mp'])
            >>> mesh1 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=['dp', 'mp'])
            >>> class RandomDataset(Dataset): # type: ignore[type-arg]
            ...     def __init__(self, seq_len, hidden, num_samples=8):
            ...         super().__init__()
            ...         self.seq_len = seq_len
            ...         self.hidden = hidden
            ...         self.num_samples = num_samples
            ...         self.inputs1 = [
            ...             np.random.uniform(size=[self.seq_len, self.hidden]).astype(
            ...                 "float32"
            ...             )
            ...             for _ in range(num_samples)
            ...         ]
            ...         self.inputs2 = [
            ...             np.random.uniform(size=[self.seq_len, self.hidden]).astype(
            ...                 "float32"
            ...             )
            ...             for _ in range(num_samples)
            ...         ]
            ...         self.labels = [
            ...             np.array(index, dtype="float32") for index in range(num_samples)
            ...         ]
            ...     def __getitem__(self, index):
            ...         return {
            ...             "inputs": [self.inputs1[index], self.inputs2[index]],
            ...             "label": self.labels[index],
            ...         }
            ...     def __len__(self):
            ...         return self.num_samples

            >>> dataset = RandomDataset(4, 8)
            >>> sampler = BatchSampler(
            ...     dataset,
            ...     batch_size=2,
            ... )
            >>> dataloader = DataLoader(
            ...     dataset,
            ...     batch_sampler=sampler,
            ... )
            >>> dist_dataloader = dist.shard_dataloader(
            ...     dataloader=dataloader,
            ...     meshes=[mesh0, mesh1],  # or [[mesh0, mesh0], mesh1]
            ...     shard_dims="dp",
            ...     input_keys=["inputs", "label"],
            ... )
    """

    return ShardDataloader(
        dataloader,
        meshes,
        input_keys,
        shard_dims,
        is_dataset_splitted,
        dense_tensor_idx,
    )


def in_auto_parallel_align_mode():
    return paddle.base.framework.get_flags(
        "FLAGS_enable_auto_parallel_align_mode"
    )["FLAGS_enable_auto_parallel_align_mode"]


def enable_auto_dp():
    """
    Enables an automated Data Parallel (DP) setup for auto-parallel training.

    This function simplifies the process of implementing vanilla (standard) Data
    Parallelism within the auto-parallel framework. By calling ``enable_auto_dp()``,
    users can achieve data parallel training without needing to manually configure
    ``paddle.distributed.shard_dataloader`` (or a similar distributed dataloader
    interface) for DP-specific data sharding or distribution. This mode automates
    the setup required for DP communication and data handling.

    The function works by setting the related environment variable
    to ``1``. This signals to the auto-parallel system that it should
    automatically manage the data parallelism aspects of the training process
    according to a predefined strategy.

    A significant advantage of this automated DP mode is its inherent robustness
    and ability to handle scenarios that can be challenging for manual or other
    standard DP configurations. For instance, it is particularly effective for:

    - Training models where input data may have non-uniform shapes across
      different data parallel ranks (e.g., certain video generation models
      like Wanx). In such cases, where traditional DP might lead to program
      hangs due to shape mismatches during communication, this automated mode
      employs strategies (like adjusting data representation and gradient
      synchronization) to ensure smooth training.

    In essence, ``enable_auto_dp()`` provides two key benefits:

    1. **Simplified DP Setup:** Automates the configuration for basic data
       parallelism, reducing manual setup effort (e.g., no need for manual
       ``shard_dataloader`` DP configuration).
    2. **Robustness for Complex Cases:** Effectively handles advanced scenarios
       like non-uniform input shapes.

    Note:
        This function should typically be called at the very beginning of your
        training script, prior to initializing Paddle's distributed environment
        or any auto-parallel components. The underlying auto-parallel framework,
        including its data loading and optimizer components, must be designed to
        recognize and act upon the environment variable.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> import paddle
            >>> from paddle import nn
            >>> import paddle.distributed as dist
            >>> from paddle.io import Dataset, DataLoader

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> dist.enable_auto_dp()

            >>> BATCH_SIZE = 32
            >>> CLASS_NUM = 10
            >>> INPUT_DIM = 256
            >>> STEPS = 100

            >>> class RandomDataset(Dataset):  # type: ignore[type-arg]
            ...     def __init__(self, num_samples):
            ...         rank = dist.get_rank() if dist.get_world_size() > 1 else 0
            ...         np.random.seed(42 + rank)
            ...         self.num_samples = num_samples
            ...     def __getitem__(self, idx):
            ...         x = np.random.rand(INPUT_DIM).astype('float32')
            ...         y = np.random.randint(0, CLASS_NUM, (1,)).astype('int64')
            ...         return x, y
            ...     def __len__(self):
            ...         return self.num_samples

            >>> class SimpleNet(nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.net = nn.Sequential(
            ...             nn.Linear(INPUT_DIM, 102400),
            ...             nn.Linear(102400, INPUT_DIM),
            ...             nn.Linear(INPUT_DIM, CLASS_NUM),
            ...         )
            ...     def forward(self, x):
            ...         return self.net(x)

            >>> model = SimpleNet()
            >>> optimizer = paddle.optimizer.AdamW(learning_rate=1e-3, parameters=model.parameters())
            >>> dataset = RandomDataset(num_samples=STEPS * BATCH_SIZE)
            >>> loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

            >>> model.train()
            >>> for step, (x, y) in enumerate(loader):
            ...     y.stop_gradient = True
            ...     loss = paddle.mean(model(x))
            ...     loss.backward()
            ...     optimizer.step()
            ...     model.clear_gradients()
            ...     if step % 5 == 0:
            ...         print(f"[step {step}] loss: {loss.item():.4f}")

            >>> # This case need to be executed in multi-card environment
            >>> # export CUDA_VISIBLE_DEVICES=0,1
            >>> # python -m paddle.distributed.launch {test_case}.py

    """
    _enable_auto_dp()
