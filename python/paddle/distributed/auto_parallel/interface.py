#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import copy
import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import Variable
from paddle.fluid.framework import in_dygraph_mode
from .dist_context import get_default_distributed_context
from .dist_tensor import DistributedTensor
from .dist_op import DistributedModule
from .dist_attribute import TensorDistributedAttribute
from .dist_attribute import OperatorDistributedAttribute


def _static_mode_check():
    if in_dygraph_mode():
        raise RuntimeError("Auto-parallel only supports static mode for now, "
                           "please use paddle.enable_static() first.")


def shard_tensor(x, dist_attr=None):
    """
    Add distributed attributes for a tensors.

    Args:
        x (Tensor): the tensor to process.
        mesh (ProcessMesh): an instance of ProcessMesh to describe the topology of logical processes.
        dim_mapping (list): a list to describe the mapping between `x` and `mesh`,
            the dimension `i` of `x` is split across the dimension `dims_mapping[i]`, where -1 means
            without parition along the corresponding dimension.

    Returns:
        Tensor: the tensor `x` itself.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            
            paddle.enable_static()

            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            x = paddle.ones([4, 6])
            dist.shard_tensor(x, mesh, [0, -1])

    """
    _static_mode_check()
    assert isinstance(dist_attr, (dict, TensorDistributedAttribute)), \
        "The type of dist_attr must be dict or TensorDistributedAttribute."
    dist_tensor = DistributedTensor(x, dist_attr)
    dist_tensor.mark_as_annotated()
    default_dist_ctx = get_default_distributed_context()
    default_dist_ctx.add_dist_tensor_for_program(dist_tensor)
    return x


def shard_op(op_fn, dist_attr=None):
    """
    Call a functioin and add distributed attributes for ops added by the function.

    Args:
        op_fn (callable): a callable object of an API.
        mesh (ProcessMesh): an instance of ProcessMesh specifies the topology of logical processes.
        dim_mapping_dict (dict): a mapping from tensor's name to its dims_mapping.
            The dim_mapping is a list to describe the mapping between a tensor and `mesh`,
            the dimension `i` of the tensor is split across the dimension `dim_mapping[i]`,
            where -1 means without parition along the corresponding dimension.
        kwargs (dict): a dict of parameter passed to the function `op_fn`.

    Returns:
        list: the outputs of the function `op_fn`.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()
            
            mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]])
            x = paddle.ones([4, 6])
            y = paddle.zeros([4, 6])
            kwargs = {'x': x, 'y': y}
            dist.shard_op(paddle.add, mesh, None, **kwargs)

    """
    _static_mode_check()
    assert isinstance(dist_attr, (dict, OperatorDistributedAttribute)), \
        "The type of dist_attr must be dict or OperatorDistributedAttribute."
    dist_module = DistributedModule(op_fn, dist_attr)
    return dist_module
