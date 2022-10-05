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

from collections import defaultdict

import paddle
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
    Shard a tensor on a process mesh according to the shard specification.

    Args:
        x (Tensor): the tensor to be sharded.
        process_mesh (ProcessMesh, optional): An instance of ProcessMesh describes a mesh
            topology of the used logical processes where the tensor is sharded. If it is None,
            the found current process mesh will be used. And an error will be raised if the
            current process mesh cannot be found. Default: None.
        shard_spec (list, optional): a list to describe the sharding mapping between `x` and `process_mesh`,
            which means the dimension `i` of `x` is split across the dimension `shard_spec[i]` of `process_mesh`,
            where `None` means that tensor dimension is not split. For example, given a tensor wih
            the shape [6, 12] and a process mesh with the shape [2, 3] and the dimension names ["x", "y"]:
                If `shard_spec=["x", "y"]`, each shard of the tensor will have a shape [3, 4];
                If `shard_spec=["y", "x"]`, each shard of the tensor will have a shape [2, 6];
                If `shard_spec=["x", None]`, each shard of the tensor will have a shape [3, 12];
                If `shard_spec=[None, "x"]`, each shard of the tensor will have a shape [6, 4];
                If `shard_spec=["y", None]`, each shard of the tensor will have a shape [2, 12];
                If `shard_spec=[None, "y"]`, each shard of the tensor will have a shape [6, 4];
                If `shard_spec=[None, None]`, each shard of the tensor will have a shape [6, 12];
        If the `shard_spec` is None, the tensor will be replicated across all the processes of `process_mesh`.
        In the above example, the `shard_spec=None` is same as 'shard_spec=[None, None]'. Defaults: None.

    Returns:
        Tensor: the tensor `x` annotated with sharding information.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed.fleet import auto

            mesh = auto.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
            x = paddle.ones([4, 6])
            shard_spec = ["x", "y"]
            auto.shard_tensor(x, mesh, shard_spec)

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
    if shard_spec is not None:
        assert verify_shard_spec(shard_spec, tensor_shape, process_mesh), \
            "For tensor {}, shard_spec {} is invalid with tensor_shape {} and process_mesh {}.".format(
                serial_tensor.name, shard_spec, tensor_shape, process_mesh)
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
    Shard an operation on a process mesh according to its input and output shard specification.

    Args:
        op (Callable): a callable operator or module to be sharded.
        process_mesh (ProcessMesh, optional): An instance of ProcessMesh describes a mesh
            topology of the used logical processes where the op is sharded. All of its inputs and
            outputs are sharded by this process mesh. If it is None, the found current process mesh
            will be used. And an error will be raised if the current process mesh cannot be found.
            Default: None.
        in_shard_specs (list of list, optional): a list of list to describe the sharding specifications
            for the inputs. Each item of `in_shard_specs` is a `shard_spec` between the correspoinding input
            and `process_mesh`. If one item is None, the cooresponding input is replicated across all processes
            If it is None, all inputs are replicated accross all processes. Note that the lenght of the
            `in_shard_specs` should be equal to the actual number of inputs when calling this operation.
            Default: None.
        out_shard_specs (list of list, optional): a list of list to describe the sharding specifications
            for the outputs. Each item of `out_shard_specs` is a `shard_spec` between the correspoinding output
            and `process_mesh`. If one item is None, the cooresponding output is replicated across all processes
            If it is None, all outputs are replicated accross all processes. Note that the lenght of the
            `in_shard_specs` should be equal to the actual number of inputs when calling this operation.
            Default: None. Default: None.

    Returns:
        Outputs of `op`, each of which is annotated with sharding information.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed.fleet import auto

            x = paddle.ones([4, 6])
            y = paddle.zeros([4, 6])
            mesh = auto.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
            dist_add = auto.shard_op(paddle.add,
                                     in_shard_specs=[["x", "y"], ["y", None]],
                                     out_shard_specs=[[None, "x"]])
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
                out_dims_mappings.append(
                    convert_to_dims_mapping(shard_spec, process_mesh))
            else:
                out_dims_mappings.append(None)
    op = DistributedOperatorHelper(op, process_mesh, in_dims_mappings,
                                   out_dims_mappings)
    return op


def recompute(op):

    class RecomputeOperator:

        def __init__(self, op):
            self._op = op

        def __call__(self, *args, **kwargs):
            default_prog = paddle.fluid.default_main_program()
            cur_block = default_prog.current_block()
            op_size = len(cur_block.ops)
            output = self._op(*args, **kwargs)
            new_op_size = len(cur_block.ops)

            for idx in range(op_size, new_op_size):
                op = cur_block.ops[idx]
                op._set_attr("is_recompute@auto_parallel", True)

            return output

    return RecomputeOperator(op)


# _g_fetched_tensors = {}

# def fetch(tensor, name=None):
#     if name is None:
#         _g_fetched_tensors[tensor.name] = tensor
#     else:
#         _g_fetched_tensors[name] = tensor

# def _get_fetches():
#     return _g_fetched_tensors

_g_collections = {}


class CollectionNames(object):
    FEEDS = "feeds"
    FETCHES = "fetches"


def get_collection(name):
    collection = _g_collections.get(name, None)
    if collection is None:
        collection = []
        _g_collections[name] = collection
    return _g_collections[name]


def add_to_collection(collection_name, value, value_name=None):
    if collection_name not in _g_collections:
        _g_collections[collection_name] = []
    else:
        if value_name is not None:
            _g_collections[collection_name].append((value_name, value))
        else:
            _g_collections[collection_name].append((None, value))


def fetch(tensor, name=None):
    add_to_collection(CollectionNames.FETCHES, tensor, name)
