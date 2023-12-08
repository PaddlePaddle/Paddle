#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict
from typing import Callable

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.base.framework import EagerParamBase
from paddle.distributed.auto_parallel.interface import (
    shard_tensor as shard_tensor_static,
)
from paddle.framework import core

from .placement_type import get_shard_spec

# There are the auto parallel API of the unified version of dynamic and static mode.
# Some APIs have the same name with the previous APIs implementation, which are
# a temporary state, and the APIs here will eventually be used.

# Part1: Shard attributes related APIs


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
    data, mesh, placements, dtype=None, place=None, stop_gradient=True
):
    """
    Constructs a ``paddle.Tensor`` with distributed attributes from ``data``,
    which can scalar, tuple, list, numpy.ndarray, paddle.Tensor.

    If the ``data`` is already a Tensor, transform it to a Distributed Tensor.

    Args:
        data(scalar|tuple|list|ndarray|Tensor): Initial data for the tensor.
            Can be a scalar, list, tuple, numpy.ndarray, paddle.Tensor.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        placements(list[paddle.distributed.Placement]): the placements describe how to place the tensor on ProcessMesh, it can
            be Shard, Replicate and Partial.
        dtype(str|np.dtype, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' ,
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data``
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional): The place to allocate Tensor. Can be
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs.
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.

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
    # `paddle.to_tensor` supports both dynamic and static mode
    tensor = paddle.to_tensor(
        data, dtype=dtype, place=place, stop_gradient=stop_gradient
    )

    if paddle.in_dynamic_mode():
        # here the dist tensor is deep copy constructed
        if isinstance(data, EagerParamBase):
            return EagerParamBase.from_tensor(
                tensor,
                process_mesh=mesh,
                placements=placements,
                **tensor.__dict__,
            )
        else:
            return paddle.Tensor(
                tensor, process_mesh=mesh, placements=placements, place=place
            )
    else:
        # TODO(zhiqiu): we need to refine the static shard_tensor
        sharding_specs = get_shard_spec(mesh, placements, tensor.ndim)
        return shard_tensor_static(tensor, mesh, sharding_specs)


def dtensor_from_fn(fn, mesh, placements, *args, **kwargs):
    """
    Construct a Distributed Tensor from a function of arguments.

    Args:
        fn (callable): A callable function that takes arguments of Distributed Tensor and returns tensor.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        placements(list[paddle.distributed.Placement]): the placements describe how to place the tensor on ProcessMesh, it can
            be Shard, Replicate and Partial.
        *args (tuple): A tuple of arguments to be passed to the ``fn`` function.
        **kwargs (dict): A dict of arguments to be passed to the ``fn`` function.

    Retruns:
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


def reshard(dist_tensor, mesh, placements):
    """
    Reshard a distributed ``paddle.Tensor`` with given distributed attributes.

    Args:
        dist_tensor(Tensor): the distributed tensor to be resharded.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        placements(list[paddle.distributed.Placement]): the placements describe how to place the tensor on ProcessMesh, it can
            be Shard, Replicate and Partial.

    Returns:
        Tensor: A Distributed Tensor reshared with distributed attributes.

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
        sharding_specs = get_shard_spec(mesh, placements, dist_tensor.ndim)
        dist_attr = DistAttr(mesh, sharding_specs)
        partial_dims = []
        for i, p in enumerate(placements):
            if isinstance(p, dist.Partial):
                partial_dims.append(i)
        if len(partial_dims) > 0:
            dist_attr._set_partial_dims(partial_dims)

        return paddle.base.core.reshard(dist_tensor, dist_attr)
    else:
        # TODO(GhostScreaming): Support static DistTensor later.
        raise RuntimeError(
            "paddle.dist.reshard only support dynamic graph now. It will be supported for static graph later."
        )


def shard_layer(
    layer: nn.Layer,
    process_mesh: dist.ProcessMesh,
    shard_fn: Callable = None,
    input_fn: Callable = None,
    output_fn: Callable = None,
) -> nn.Layer:
    """
    Converts all layer's parameters to DistTensor parameters according to
    the `shard_fn` specified. It could also control the conversion of input
    or output of the layer by specifying the `input_fn` and `output_fn`.
    (i.e. convert the input to `paddle.Tensor` with DistTensor, convert output
    back to `paddle.Tensor` with DenseTensor.)

    The `shard_fn` should have the following signature:

        def shard_fn(layer_name, layer, process_mesh) -> None

    The `input_fn` should have the following signature:

        def input_fn(inputs, process_mesh) -> list(paddle.Tensor)

    In general, the type of `input_fn` return value is paddle.Tensor with DistTensor.

    The `output_fn` should have the following signature:

        def output_fn(outputs, process_mesh) -> list(paddle.Tensor)

    In general, the type of `output_fn` return value is paddle.Tensor with DenseTensor.

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
            convert it back to `paddle.Tensor` with DenseTensor.
            The `output_fn` will be registered for the Layer as `forward post-hook`.
            By default we do not shard or convert the output.
    Returns:
        Layer: A layer that contains parameters/buffers
            that are all `paddle.Tensor` with DistTensor

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

            >>> # This case need to be excuted in multi-card environment
            >>> # export CUDA_VISIBLE_DEVICES=0,1
            >>> # python -m paddle.distributed.launch {test_case}.py
    """
    # Ensure that process_mesh is not an empty object
    if process_mesh is None:
        raise ValueError("The argument `process_mesh` cannot be empty.")

    # Check the legality of process_mesh
    if not isinstance(process_mesh, dist.ProcessMesh):
        raise ValueError(
            "The argument `process_mesh` is not `dist.ProcessMesh` type."
        )

    def replicate_layer_params_and_buffers(
        layer: nn.Layer, mesh: dist.ProcessMesh
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
            "`paddle.distributed.shard_layer` only supports dynamic graph mode "
            "now. It will be supported for static graph mode later."
        )


class _ShardOptimizer:
    def __init__(self, optimizer, shard_fn=None):
        assert (
            optimizer is not None
        ), "The argument `optimizer` cannot be empty."
        assert isinstance(
            optimizer, (paddle.optimizer.AdamW, paddle.optimizer.SGD)
        ), "`paddle.distributed.ShardOptimizer` only supports AdamW and SGD optimizer for now."

        self.target_block = (
            paddle.base.framework.default_main_program().global_block()
        )
        optimizer.helper = paddle.base.layer_helper.LayerHelper(
            optimizer.__class__.__name__
        )
        self._inner_opt = optimizer
        self._shard_fn = shard_fn

    def _shard_accumulator(self, param):
        # create the accumulators
        self._inner_opt._create_accumulators(self.target_block, [param])

        target_name = param.name
        if param.name in self._inner_opt._master_weights.keys():
            target_name = self._inner_opt._master_weights[param.name].name

        # shard the accumulators
        for key in self._inner_opt._accumulators.keys():
            accumulator = self._inner_opt._accumulators[key][target_name]
            if accumulator.is_dist():
                continue
            if self._shard_fn is not None:
                self._inner_opt._accumulators[key][
                    target_name
                ] = self._shard_fn(key, param, accumulator)
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
                    self._inner_opt._accumulators[key][
                        target_name
                    ] = shard_tensor(
                        accumulator,
                        mesh=param.process_mesh,
                        placements=placements,
                    )

    def step(self):
        if not isinstance(self._inner_opt._parameter_list[0], dict):
            params_grads = []
            for param in self._inner_opt._parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
            for p, g in params_grads:
                self._shard_accumulator(p)
            self._inner_opt._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads
            )
        else:
            for param_group in self._inner_opt._param_groups:
                params_grads = defaultdict(lambda: [])
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )
                for p, g in params_grads['params']:
                    self._shard_accumulator(p)
                self._inner_opt._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads
                )

    def state_dict(self):
        """
        Create and shard the optimizer states e.g., acumulators and master_weights before load_state_dict.
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

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)


def shard_optimizer(optimizer, shard_fn=None):
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
    return _ShardOptimizer(optimizer, shard_fn)
