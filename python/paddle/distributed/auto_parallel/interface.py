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
from paddle.fluid.framework import _non_static_mode
from .dist_context import get_default_distributed_context
from .dist_tensor import DistributedTensor
from .dist_op import DistributedModule
from .dist_attribute import TensorDistributedAttribute
from .dist_attribute import OperatorDistributedAttribute


def _static_mode_check():
    if _non_static_mode():
        raise RuntimeError("Auto-parallel only supports static mode for now, "
                           "please use paddle.enable_static() first.")


def shard_tensor(x, dist_attr=None):
    """
    Add distributed attributes for a tensors.

    Args:
        x (Tensor): the tensor to be sharded.
        dist_attr (dict): the tensor distributed attributes. The accepted attributes are as follow:
            "process_mesh": a nested list an to describe the mesh topology of logical processes.
            "dims_mapping": a list to describe the mapping between `x` and `process_mesh`, the dimension
                `i` of `x` is split across the dimension `dims_mapping[i]` of `process_mesh`,
                where -1 means that tensor dimension is not split.
            Both process_mesh and dims_mapping are optional and users can specify as need.

    Returns:
        Tensor: the tensor `x` annotated with distributed attributes.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()

            x = paddle.ones([4, 6])
            dist.shard_tensor(x, dist_attr={"process_mesh": [[0, 1], [2, 3]],
                                            "dims_mapping": [0, -1]})

    """
    _static_mode_check()
    assert dist_attr is None or isinstance(dist_attr, (dict, TensorDistributedAttribute)), \
        "The type of dist_attr must be None, dict or TensorDistributedAttribute."
    dist_tensor = DistributedTensor(x, dist_attr)
    dist_tensor.dist_attr.mark_annotated_as(dist_attr)
    default_dist_ctx = get_default_distributed_context()
    default_dist_ctx.add_dist_tensor_for_program(dist_tensor)
    return x


def shard_op(op_fn, dist_attr=None):
    """
    Call a functioin and add distributed attributes for ops added by the function.

    Args:
        op_fn (callable): a callable operator or module to be sharded.
        dist_attr (dict): the operator distributed attributes. The accepted attributes are classified into
            two categories. The first category decsribes the distributed attributes shared by all inputs and
            outputs, and only `process_mesh` can be specified now. The second category describes distributed
            attributes for inputs or outputs same as the `dist_attr` of `shard_tensor`. All of them are
            optional and users can specify them as need. Note that `process_mesh` for operators must be the
            same as these process_meshes for inputs and outputs.

    Returns:
        list: the outputs of the function `op_fn`, which are annotated with distributed attributes.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            paddle.enable_static()

            x = paddle.ones([4, 6])
            y = paddle.zeros([4, 6])
            dist_add = dist.shard_op(paddle.add,
                                     dist_attr={
                                         "process_mesh": [[2, 3, 1], [0, 4, 5]],
                                         x: {"dims_mapping": [-1, 0]},
                                         y: {"dims_mapping": [0, -1]}
                                     })
            dist_add(x, y)

    """
    _static_mode_check()
    assert dist_attr is None or isinstance(dist_attr, (dict, OperatorDistributedAttribute)), \
        "The type of dist_attr must be dict or OperatorDistributedAttribute."
    dist_module = DistributedModule(op_fn, dist_attr)
    return dist_module
