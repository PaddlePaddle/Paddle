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

from paddle.fluid import core
from .process_mesh import ProcessMesh
from .process_mesh import get_current_process_mesh
from .process_mesh import set_current_process_mesh
from .process_mesh import reset_current_process_mesh
from .dist_context import get_default_distributed_context
from .dist_tensor import DistributedTensor
from .dist_op import DistributedOperatorHelper
from .utils import verify_shard_spec, convert_to_dims_mapping


def shard_tensor(x, process_mesh=None, shard_spec=None):
    """
    Add distributed attributes for a tensors.

    Args:
        x (Tensor): the tensor to be sharded.
        dist_attr (dict): the tensor distributed attributes. The accepted attributes are as follow:
            "process_mesh": a nested list an to describe the mesh topology of logical processes.
            "shard_spec": a list to describe the mapping between `x` and `process_mesh`, the dimension 
                `i` of `x` is split across the dimension `shard_spec[i]` of `process_mesh`, 
                where -1 means that tensor dimension is not split.
            Both process_mesh and shard_spec are optional and users can specify as need.

    Returns:
        Tensor: the tensor `x` annotated with distributed attributes.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist
            
            paddle.enable_static()

            x = paddle.ones([4, 6])
            dist.shard_tensor(x, dist_attr={"process_mesh": [[0, 1], [2, 3]],
                                            "shard_spec": [0, -1]})

    """

    if process_mesh is not None:
        assert isinstance(process_mesh, ProcessMesh), \
            "Argument process_mesh {} is not an instance of ProcessMesh".format(process_mesh)
    else:
        process_mesh = get_current_process_mesh()
        assert  process_mesh is not None, \
            "Specify the process mesh argument or use ProcessMesh context manager first."
    assert isinstance(shard_spec, list), \
        "Argument shard_spec {} is not an instance of list".format(shard_spec)
    dist_tensor = DistributedTensor(x)
    serial_tensor = dist_tensor.serial_tensor
    dist_tensor.dist_attr.process_mesh = process_mesh
    if serial_tensor.type == core.VarDesc.VarType.READER \
        or serial_tensor.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY \
        or serial_tensor.type == core.VarDesc.VarType.STEP_SCOPES:
        tensor_shape = []
    else:
        tensor_shape = serial_tensor.shape
    assert len(tensor_shape) == len(shard_spec), \
        "The length of shard_spec {} does not match the length of tensor shape {}.".format(shard_spec, tensor_shape)
    if shard_spec is not None:
        assert verify_shard_spec(shard_spec, process_mesh), \
            "shard_spec {} is invalid".format(shard_spec)
        dist_tensor.dist_attr.dims_mapping = convert_to_dims_mapping(
            shard_spec, process_mesh)
    if process_mesh is not None:
        dist_tensor.dist_attr.mark_annotated("process_mesh")
    if shard_spec is not None:
        dist_tensor.dist_attr.mark_annotated("dims_mapping")
    default_dist_ctx = get_default_distributed_context()
    default_dist_ctx.add_dist_tensor_for_program(dist_tensor)
    dist_tensor = default_dist_ctx.get_dist_tensor_for_program(x)
    return x


def shard_op(op, process_mesh=None, in_shard_specs=None, out_shard_specs=None):
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
                                         x: {"shard_spec": [-1, 0]},
                                         y: {"shard_spec": [0, -1]}
                                     })
            dist_add(x, y)

    """

    if process_mesh is not None:
        assert isinstance(process_mesh, ProcessMesh), \
            "Argument process_mesh {} is not an instance of ProcessMesh".format(process_mesh)
    else:
        process_mesh = get_current_process_mesh()
        assert  process_mesh is not None, \
            "Specify the process mesh argument or use ProcessMesh context manager first."
    in_dims_mappings = []
    if in_shard_specs is not None:
        assert all((isinstance(shard_spec, list) or shard_spec is None) for shard_spec in in_shard_specs), \
            "in_shard_spec {} is not a list of list or None".format(in_shard_specs)
        for shard_spec in in_shard_specs:
            if shard_spec is not None:
                assert verify_shard_spec(shard_spec, process_mesh), \
                    "shard_spec {} of in_shard_specs {} is invalid".format(
                        shard_spec, in_shard_specs)
                in_dims_mappings.append(
                    convert_to_dims_mapping(shard_spec, process_mesh))
            else:
                in_dims_mappings.append(None)
    out_dims_mappings = []
    if out_shard_specs is not None:
        assert all((isinstance(shard_spec, list) or shard_spec is None) for shard_spec in out_shard_specs), \
            "out_shard_spec {} is not a list of list or None".format(out_shard_specs)
        for shard_spec in out_shard_specs:
            if shard_spec is not None:
                assert verify_shard_spec(shard_spec, process_mesh), \
                    "shard_spec {} of in_shard_specs {} is invalid".format(
                        shard_spec, out_shard_specs)
                out_dims_mappings.append(
                    convert_to_dims_mapping(shard_spec, process_mesh))
            else:
                out_dims_mappings.append(None)
    op = DistributedOperatorHelper(op, process_mesh, in_dims_mappings,
                                   out_dims_mappings)
    return op
