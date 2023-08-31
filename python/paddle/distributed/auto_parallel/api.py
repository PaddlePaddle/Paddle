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

from typing import Callable

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.framework import core

# There are the auto parallel API of the unified version of dynamic and static mode.
# Some APIs have the same name with the previous APIs implementation, which are
# a temporary state, and the APIs here will eventually be used.


class DistAttr(core.TensorDistAttr):
    """
    DistAttr specifies how tensors are distributed or sliced on ProcessMesh.

    Args:
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        sharding_specs(list[str|None]): The specification describing how to shard the Tensor.

    Examples:

    .. code-block:: python

        import paddle
        import paddle.distributed as dist

        mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])

        print(dist_attr)
    """

    def __init__(self, mesh, sharding_specs):
        # 1. inputs checking
        if not isinstance(mesh, ProcessMesh):
            raise ValueError(
                "The mesh must be an instance of paddle.distributed.ProcessMesh."
            )
        if not isinstance(sharding_specs, list):
            raise ValueError("The sharding_specs must be an instance of list.")
        assert all(
            isinstance(dim_name, str) or dim_name is None
            for dim_name in sharding_specs
        ), 'The dimension name in sharding_specs must be an instance of str.'

        dims_mapping = [
            mesh.dim_names.index(dim_name) if dim_name is not None else -1
            for dim_name in sharding_specs
        ]

        # 2. init core.TensorDistAttr
        core.TensorDistAttr.__init__(self)
        self.process_mesh = mesh
        self.dims_mapping = dims_mapping


def shard_tensor(
    data, dtype=None, place=None, stop_gradient=True, dist_attr=None
):
    """
    Constructs a ``paddle.Tensor`` with distributed attributes from ``data``,
    which can scalar, tuple, list, numpy.ndarray, paddle.Tensor.

    If the ``data`` is already a Tensor, transform it to a Distributed Tensor.

    Args:
        data(scalar|tuple|list|ndarray|Tensor): Initial data for the tensor.
            Can be a scalar, list, tuple, numpy.ndarray, paddle.Tensor.
        dtype(str|np.dtype, optional): The desired data type of returned tensor. Can be 'bool' , 'float16' ,
            'float32' , 'float64' , 'int8' , 'int16' , 'int32' , 'int64' , 'uint8',
            'complex64' , 'complex128'. Default: None, infers dtype from ``data``
            except for python float number which gets dtype from ``get_default_type`` .
        place(CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional): The place to allocate Tensor. Can be
            CPUPlace, CUDAPinnedPlace, CUDAPlace. Default: None, means global place. If ``place`` is
            string, It can be ``cpu``, ``gpu:x`` and ``gpu_pinned``, where ``x`` is the index of the GPUs.
        stop_gradient(bool, optional): Whether to block the gradient propagation of Autograd. Default: True.
        dist_attr(paddle.distributed.DistAttr): Specify how tensors are distributed or sliced on ProcessMesh.

    Returns:
        Tensor: A Tensor constructed from ``data`` with distributed attributes.

    Examples:

    .. code-block:: python

        import paddle
        import paddle.distributed as dist

        mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])

        # dense tensor
        a = paddle.to_tensor([[1,2,3],
                              [5,6,7]])
        # distributed tensor
        d_tensor = dist.shard_tensor(a, dist_attr=dist_attr)

        print(d_tensor)
    """
    # 1. create dense tensor
    # `paddle.to_tensor` supports both dynamic and static mode
    data = paddle.to_tensor(data)

    # 2. create dist tensor
    assert len(dist_attr.dims_mapping) == len(
        list(data.shape)
    ), "The length of sharding_specs must be same as the shape of the input tensor."

    if paddle.in_dynamic_mode():
        return paddle.Tensor(data, dist_attr=dist_attr)
    else:
        raise NotImplementedError(
            "The `paddle.distributed.shard_tensor` for static mode will be implemented later."
        )


def dtensor_from_fn(fn, dist_attr, *args, **kwargs):
    """
    Construct a Distributed Tensor from a function of arguments.

    Args:
        fn (callable): A callable function that takes arguments of Distributed Tensor and returns tensor.
        dist_attr (paddle.distributed.DistAttr): Specify how tensors are distributed or sliced on ProcessMesh.
        *args (tuple): A tuple of arguments to be passed to the ``fn`` function.
        **kwargs (dict): A dict of arguments to be passed to the ``fn`` function.

    Retruns:
        Tensor: A Tensor constructed from ``fn`` with distributed attributes.

    Examples:

    .. code-block:: python

        >>> import paddle
        >>> import paddle.distribute as dist
        >>> # Create a distributed attribute
        >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        >>> dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])
        >>> # Call the function dtensor_from_fn with dist_attr parameter
        >>> d_tensor = dist.dtensor_from_fn(paddle.ones, dist_attr=dist_attr, shape=[1])
        >>> print(d_tensor)
    """
    tensor = fn(*args, **kwargs)
    return shard_tensor(tensor, dist_attr=dist_attr)


def shard_layer(
    model: nn.Layer,
    process_mesh: dist.ProcessMesh,
    shard_fn: Callable = None,
    input_fn: Callable = None,
    output_fn: Callable = None,
) -> nn.Layer:
    """
    This function converts all model parameters to DistTensor parameters
    according to the `shard_fn` specified. It could also control the input or
    output of the module by specifying the `input_fn` and `output_fn`.

    Args:
        model(nn.Layer): Model constructed by users using paddle.nn.Layer.
        process_mesh(ProcessMesh): ProcessMesh information to be placed in this model.
        shard_fn(Callable): Function for splitting model parameters. If not specified, by default we copy all parameters of the model across ProcessMesh.
        input_fn(Callable): Specify the partition distribution of the input, input_fn will serve for the Layer as forward_pre_hook.
        ouput_fn(Callable): Specify the partition distribution of the output, output_fn will serve for the Layer as forward_post_hook. By default we do not do any partitioning.

    Returns:
        model:model with DistTensor parameters

    Examples:

    ..code-block:: python

        >>> import paddle
        >>> import paddle.distributed as dist
        >>> mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

        >>> class MLP(paddle.nn.Layer):
        ...     def __init__(self, ):
        ...         super.__init__()
        ...         self.fc1 = nn.Linear(8, 8)
        ...         elf.fc2 = nn.Linear(8, 8)

        ...     def forward(self, input):
        ...         return self.fc2(self.fc1(input))

        >>> def shard_params_func(model_name, model):
        ...     dist_attr = dist.TensorDistAttr(shard_spec==['x', 'y'], mesh=mesh)
        ...     if model_name == 'fc1':
        ...         model.weight = dist.shard_tensor(model.weight, dist_attr)

        >>> model = MLP()
        >>> model = dist.shard_layer(model, shard_params_func)

        >>> print(model)
    """
    process_mesh = process_mesh or process_mesh.get_current_process_mesh()

    def replicate_model_params_buffers(m: nn.model, mesh: process_mesh) -> None:
        full_replicate = [paddle.dtensor.Replicate()] * mesh.ndim
        for key, param in m._parameters.items():
            if param is not None and not isinstance(
                param, paddle.dtensor.DTensor
            ):
                m.register_parameter(
                    key,
                    nn.Parameter(
                        paddle.dtensor.distribute_tensor(
                            param.data, mesh, full_replicate
                        )
                    ),
                )
        for key, buffer in m._buffers.items():
            if buffer is not None and not isinstance(
                buffer, paddle.dtensor.DTensor
            ):
                m._buffers[key] = paddle.dtensor.distribute_tensor(
                    buffer, mesh, full_replicate
                )

    if shard_fn is None:
        # if shard_fn not specified, by default replicate
        # all model params
        for name, submod in model.named_modules():
            replicate_model_params_buffers(submod, process_mesh)
    else:
        # apply shard_fn to submodules
        for name, submod in model.named_modules():
            shard_fn(name, submod, process_mesh)
            replicate_model_params_buffers(submod, process_mesh)

    # register input_fn as model forward pre hook
    if input_fn is not None:
        model.register_forward_pre_hook(
            lambda _, inputs: input_fn(inputs, process_mesh)
        )
    # register input_fn as model forward hook
    if output_fn is not None:
        model.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(outputs, process_mesh)
        )

    return model
