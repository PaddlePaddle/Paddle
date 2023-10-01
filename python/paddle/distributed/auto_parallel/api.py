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


import paddle
from paddle.base.framework import EagerParamBase
from paddle.distributed.auto_parallel.interface import (
    shard_tensor as shard_tensor_static,
)
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

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
            >>> dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])

            >>> # dense tensor
            >>> a = paddle.to_tensor([[1,2,3],
            ...                       [5,6,7]])

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # distributed tensor
            >>> d_tensor = dist.shard_tensor(a, dist_attr=dist_attr)

            >>> print(d_tensor)

    """
    # 1. create dense tensor
    # `paddle.to_tensor` supports both dynamic and static mode
    tensor = paddle.to_tensor(
        data, dtype=dtype, place=place, stop_gradient=stop_gradient
    )

    # 2. create dist tensor
    assert len(dist_attr.dims_mapping) == len(
        list(tensor.shape)
    ), "The length of sharding_specs must be same as the shape of the input tensor."

    if paddle.in_dynamic_mode():
        # here the dist tensor is deep copy constructed
        if isinstance(data, EagerParamBase):
            return EagerParamBase.from_tensor(
                tensor, dist_attr=dist_attr, **tensor.__dict__
            )
        else:
            return paddle.Tensor(tensor, dist_attr=dist_attr)
    else:
        # TODO(zhiqiu): we need to refine the static shard_tensor
        return shard_tensor_static(
            tensor, dist_attr.process_mesh, dist_attr.sharding_specs
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
            >>> import paddle.distributed as dist
            >>> # Create a distributed attribute
            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])
            >>> # Call the function dtensor_from_fn with dist_attr parameter
            >>> d_tensor = dist.dtensor_from_fn(paddle.ones, dist_attr=dist_attr, shape=[1])
            >>> print(d_tensor)

    """
    tensor = fn(*args, **kwargs)
    return shard_tensor(tensor, dist_attr=dist_attr)


def reshard(dist_tensor, dist_attr):
    """
    Reshard a distributed ``paddle.Tensor`` with given distributed attributes.

    Args:
        dist_tensor(Tensor): the distributed tensor to be resharded.
        dist_attr(paddle.distributed.DistAttr): Specify how tensors are distributed or sliced on ProcessMesh.

    Returns:
        Tensor: A Distributed Tensor reshared with distributed attributes.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
            >>> dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])

            >>> out_mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=['x', 'y'])
            >>> out_dist_attr = dist.DistAttr(mesh=out_mesh, sharding_specs=[None, None])

            >>> # dense tensor
            >>> a = paddle.to_tensor([[1,2,3],
            ...                       [5,6,7]])

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # distributed tensor
            >>> d_tensor = dist.shard_tensor(a, dist_attr=dist_attr)

            >>> out_d_tensor = dist.reshard(d_tensor, out_dist_attr)

            >>> print(d_tensor)
            >>> print(out_d_tensor)

    """

    if paddle.framework.in_dynamic_mode():
        return paddle.base.core.reshard(dist_tensor, dist_attr)
    else:
        # TODO(GhostScreaming): Support static DistTensor later.
        raise RuntimeError(
            "paddle.dist.reshard only support dynamic graph now. It will be supported for static graph later."
        )
