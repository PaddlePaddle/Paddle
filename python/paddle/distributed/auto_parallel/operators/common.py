# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import abc
<<<<<<< HEAD

from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole

from ..dist_attribute import OperatorDistAttr
from ..process_group import new_process_group
from ..utils import _get_comm_group, _get_corresponding_rank, is_optimize_op
=======
import paddle
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..dist_attribute import OperatorDistributedAttribute
from ..utils import _get_comm_group, _get_corresponding_rank, is_optimize_op
from ..process_group import new_process_group
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

_g_distributed_operator_impl_containers = {}

_g_elementwise_ops = [
<<<<<<< HEAD
    "elementwise",
    "gelu",
    "dropout",
    "cast",
    "gather",
    "concat",
    "fused_softmax_mask_upper_triangle",
=======
    "elementwise", "gelu", "dropout", "cast", "gather", "concat",
    "fused_softmax_mask_upper_triangle"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
]
BACKWARD_ONLY_DIST_OPS = {'check_finite_and_unscale', 'update_loss_scaling'}


<<<<<<< HEAD
class ParallelMode:
    """
    the parallel mode for communication or auxiliary operator
    """

=======
class ParallelMode():
    """
    the parallel mode for communication or auxiliary operator
    """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    DataParallel = "auto_parallel/data_parallel"
    ModelParallel = "auto_parallel/model_parallel"
    PipelineParalel = "auto_parallel/pipeline_paralel"
    MoEParallel = "auto_parallel/moe_parallel"


<<<<<<< HEAD
class SyncMode:
    """
    the synchorization mode for communication or auxiliary operator
    """

    AmpFlagSync = "auto_parallel/amp_flag_synchorization"
    GlobalNormSync = "auto_parallel/global_norm_synchorization"


=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
def is_elementwise_op(op_type):
    if op_type in _g_elementwise_ops:
        return True
    if "elementwise" in op_type:
        return True
    return False


class DistributedOperatorImplContainer:
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, op_type):
        self._type = op_type
        self._impls = []

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, op_type):
        self._type = op_type

    @property
    def impls(self):
        return self._impls

    def register_impl(self, dist_impl):
<<<<<<< HEAD
        assert (
            self.type == dist_impl.type
        ), "Op type of container must be same as that of the implementation."
=======
        assert self.type == dist_impl.type, \
            "Op type of container must be same as that of the implementation."
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        impl_idx = len(self.impls)
        dist_impl.idx = impl_idx
        self._impls.append(dist_impl)

    def get_impl(self, impl_idx):
        return self._impls[impl_idx]

    def get_input_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_input_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls

    def get_output_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_output_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls

    def get_compatible_impls(self, dist_op):
        compatible_impls = []
        for impl in self.impls:
            if impl.is_auto_compatible(dist_op):
                compatible_impls.append(impl)
        return compatible_impls


class DistributedOperatorImpl(abc.ABC):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, name):
        self._name = name
        self._type = None
        self._idx = None
        self._forward_implemented = False
        self._backward_implemented = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, op_type):
        self._type = op_type

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, impl_idx):
        self._idx = impl_idx

    @abc.abstractmethod
    def is_input_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @abc.abstractmethod
    def is_output_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @abc.abstractmethod
    def is_auto_compatible(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @staticmethod
    @abc.abstractmethod
    def forward(dist_ctx, *args, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    @staticmethod
    @abc.abstractmethod
    def backward(dist_ctx, *grad_outputs, **kwargs):
        raise NotImplementedError("Please Implement this method in Subclass.")

    def update_dims_mapping(self, dist_op):
        raise NotImplementedError("Please Implement this method in Subclass.")


def register_distributed_operator_impl_container(container):
    global _g_distributed_operator_impl_containers
    _g_distributed_operator_impl_containers[container.type] = container


def get_distributed_operator_impl_container(op_type):
    global _g_distributed_operator_impl_containers
    return _g_distributed_operator_impl_containers.get(op_type, None)


def register_distributed_operator_impl(op_type, dist_impl):
    dist_op_impl_container = get_distributed_operator_impl_container(op_type)
    if dist_op_impl_container is not None:
        dist_impl.type = op_type
        dist_op_impl_container.register_impl(dist_impl)
    else:
        assert False, "Must register distributed operator registry first."


def find_compatible_distributed_operator_impls(dist_op, fwd=True, partial=True):
    """
<<<<<<< HEAD
    Here just return the first compatible implemention.
=======
    Here just return the first compatible implemention. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    This will be improved by cost model in the future.
    """
    op_type = dist_op.serial_op.type
    dist_op_impl_container = get_distributed_operator_impl_container(op_type)
    dist_op_eltwise_impl_container = get_distributed_operator_impl_container(
<<<<<<< HEAD
        "elementwise"
    )
    dist_op_default_impl_container = get_distributed_operator_impl_container(
        "default"
    )
=======
        "elementwise")
    dist_op_default_impl_container = get_distributed_operator_impl_container(
        "default")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    compatible_impls = []
    if partial:
        if fwd:
            # First, find impls in the corresponding container
            if dist_op_impl_container:
                compatible_impls.extend(
<<<<<<< HEAD
                    dist_op_impl_container.get_input_compatible_impls(dist_op)
                )
=======
                    dist_op_impl_container.get_input_compatible_impls(dist_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            # Second, find impls in the elementwise container
            if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
                compatible_impls.extend(
                    dist_op_eltwise_impl_container.get_input_compatible_impls(
<<<<<<< HEAD
                        dist_op
                    )
                )
=======
                        dist_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            # Third, find impls in the default container
            if dist_op_default_impl_container:
                compatible_impls.extend(
                    dist_op_default_impl_container.get_input_compatible_impls(
<<<<<<< HEAD
                        dist_op
                    )
                )
=======
                        dist_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        else:
            # First, find impls in the corresponding container
            if dist_op_impl_container:
                compatible_impls.extend(
<<<<<<< HEAD
                    dist_op_impl_container.get_output_compatible_impls(dist_op)
                )
=======
                    dist_op_impl_container.get_output_compatible_impls(dist_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            # Second, find impls in the elementwise container
            if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
                compatible_impls.extend(
                    dist_op_eltwise_impl_container.get_output_compatible_impls(
<<<<<<< HEAD
                        dist_op
                    )
                )
=======
                        dist_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            # Third, find impls in the default container
            if dist_op_default_impl_container:
                compatible_impls.extend(
                    dist_op_default_impl_container.get_output_compatible_impls(
<<<<<<< HEAD
                        dist_op
                    )
                )
=======
                        dist_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    else:
        # First, find impls in the corresponding container
        if dist_op_impl_container:
            compatible_impls.extend(
<<<<<<< HEAD
                dist_op_impl_container.get_compatible_impls(dist_op)
            )
        # Second, find impls in the elementwise container
        if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
            compatible_impls.extend(
                dist_op_eltwise_impl_container.get_compatible_impls(dist_op)
            )
        # Third, find impls in the default container
        if dist_op_default_impl_container:
            compatible_impls.extend(
                dist_op_default_impl_container.get_compatible_impls(dist_op)
            )
=======
                dist_op_impl_container.get_compatible_impls(dist_op))
        # Second, find impls in the elementwise container
        if dist_op_eltwise_impl_container and is_elementwise_op(op_type):
            compatible_impls.extend(
                dist_op_eltwise_impl_container.get_compatible_impls(dist_op))
        # Third, find impls in the default container
        if dist_op_default_impl_container:
            compatible_impls.extend(
                dist_op_default_impl_container.get_compatible_impls(dist_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    if compatible_impls:
        # For now, just return the first compatible impl
        # best_compatible_impl = compatible_impls[0]
        best_compatible_impl = compatible_impls
    else:
        best_compatible_impl = None
    return best_compatible_impl


def is_parameter_related(varname, block):
    if ".subprog_" in varname:
<<<<<<< HEAD
        varname = varname[: varname.index(".subprog_")]
    if ".cast_fp" in varname:
        varname = varname[: varname.index(".cast_fp")]
    if ".cast_bf" in varname:
        varname = varname[: varname.index(".cast_bf")]
    if ".quantized" in varname:
        varname = varname[: varname.index(".quantized")]
    # if "@RESHARD" in varname:
    #     varname = varname[: varname.index("@RESHARD")]
    assert block._find_var_recursive(varname)
    var = block._var_recursive(varname)
=======
        varname = varname[:varname.index(".subprog_")]
    if ".cast_fp" in varname:
        varname = varname[:varname.index(".cast_fp")]
    if ".quantized" in varname:
        varname = varname[:varname.index(".quantized")]
    assert block.has_var(varname)
    var = block.var(varname)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return var.is_parameter


def infer_shape(block, src_var, src_var_dist_attr, op_input_dist_attr):
<<<<<<< HEAD
    var_shape = block._var_recursive(src_var.name).shape
    var_topoloy = src_var_dist_attr.process_mesh.shape
=======
    var_shape = block.var(src_var.name).shape
    var_topoloy = src_var_dist_attr.process_mesh.topology
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    var_dims_mapping = src_var_dist_attr.dims_mapping

    complete_shape = []
    for idx, shape in enumerate(var_shape):
        if var_dims_mapping[idx] == -1:
            complete_shape.append(shape)
        else:
            new_shape = shape * var_topoloy[var_dims_mapping[idx]]
            complete_shape.append(new_shape)

    exact_shape = []
<<<<<<< HEAD
    input_topology = op_input_dist_attr.process_mesh.shape
=======
    input_topology = op_input_dist_attr.process_mesh.topology
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    input_dims_mapping = op_input_dist_attr.dims_mapping
    for idx, shape in enumerate(complete_shape):
        if input_dims_mapping[idx] == -1:
            exact_shape.append(shape)
        else:
            new_shape = shape // input_topology[input_dims_mapping[idx]]
            exact_shape.append(new_shape)

    return exact_shape


<<<<<<< HEAD
def set_comm_op_dist_attr_for_program(
    new_op, process_mesh, tensor_dist_attr, ctx
):
    assert process_mesh is not None
    assert tensor_dist_attr is not None

    new_op_dist_attr = OperatorDistAttr()
=======
def set_comm_op_dist_attr_for_program(new_op, process_mesh, tensor_dist_attr,
                                      ctx):
    assert process_mesh is not None
    assert tensor_dist_attr is not None

    new_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    new_op_dist_attr.process_mesh = process_mesh
    for input_varname in new_op.desc.input_arg_names():
        new_op_dist_attr.set_input_dist_attr(input_varname, tensor_dist_attr)
    for output_varname in new_op.desc.output_arg_names():
        new_op_dist_attr.set_output_dist_attr(output_varname, tensor_dist_attr)
    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def naive_copy_op_dist_attr_for_program(new_op, ref_op, ctx):

    ref_dist_attr = ctx.get_op_dist_attr_for_program(ref_op)
<<<<<<< HEAD
    new_op_dist_attr = OperatorDistAttr()
=======
    new_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    new_op_dist_attr.process_mesh = ref_dist_attr.process_mesh

    for input_name in ref_op.input_names:
        assert input_name in new_op.input_names
        assert len(ref_op.input(input_name)) == 1
        assert len(new_op.input(input_name)) == 1

        ref_tensor_dist_attr = ref_dist_attr.get_input_dist_attr(
<<<<<<< HEAD
            ref_op.input(input_name)[0]
        )
        new_op_dist_attr.set_input_dist_attr(
            new_op.input(input_name)[0], ref_tensor_dist_attr
        )
=======
            ref_op.input(input_name)[0])
        new_op_dist_attr.set_input_dist_attr(
            new_op.input(input_name)[0], ref_tensor_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    for output_name in ref_op.output_names:
        assert output_name in new_op.output_names
        assert len(ref_op.output(output_name)) == 1
        assert len(new_op.output(output_name)) == 1

        ref_tensor_dist_attr = ref_dist_attr.get_output_dist_attr(
<<<<<<< HEAD
            ref_op.output(output_name)[0]
        )
        new_op_dist_attr.set_output_dist_attr(
            new_op.output(output_name)[0], ref_tensor_dist_attr
        )
=======
            ref_op.output(output_name)[0])
        new_op_dist_attr.set_output_dist_attr(
            new_op.output(output_name)[0], ref_tensor_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    ctx.set_op_dist_attr_for_program(new_op, new_op_dist_attr)


def get_data_parallel_group(dist_ctx, op, act_grad_names, rank):
    """
    deduce the data parallel communication group for current operator.

    Args:
        dist_ctx (DistributedContext): dist context.
<<<<<<< HEAD
        op (Operator): the current (backward) operator which might need.
        act_grad_names (list): list of input activation grads variable name to the current operator.
        out_grad_names (list): list of the output parameter's grads variable name of the current operator.
=======
        op (Operator): the current (backward) operator which might need. 
        act_grad_names (list): list of input activation grads variable name to the current operator. 
        out_grad_names (list): list of the output parameter's grads variable name of the current operator. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        rank (int): global ranks index for current process.
    """
    dp_group = None

    op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
    process_mesh = op_dist_attr.process_mesh
<<<<<<< HEAD
    mesh_shape = process_mesh.shape
    # FIXME Hack for Pipeline Parallelism where the current operator
    # not belong to the mesh the current rank belong to.
    if rank not in process_mesh.process_ids:
=======
    mesh_shape = process_mesh.topology
    # FIXME Hack for Pipeline Parallelism where the current operator
    # not belong to the mesh the current rank belong to.
    if rank not in process_mesh.processes:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        rank = _get_corresponding_rank(dist_ctx, process_mesh, rank)

    for var_name in act_grad_names:
        var_dim_mapping = op_dist_attr.get_input_dims_mapping(var_name)
        # consider that the variable's shape is None
        # TODO utilize the batch_dim attr instead of "0" in future
        batch_size_axis = var_dim_mapping[0] if len(var_dim_mapping) > 0 else -1

        if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
<<<<<<< HEAD
            group_ranks = _get_comm_group(
                process_mesh.process_ids,
                process_mesh.shape,
                batch_size_axis,
                rank,
            )
=======
            group_ranks = _get_comm_group(process_mesh.processes,
                                          process_mesh.topology,
                                          batch_size_axis, rank)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dp_group = new_process_group(group_ranks)
            break

    return dp_group


def sync_and_scale_gradients(dist_ctx, op, dp_group, allreduce_var_names):
    """
<<<<<<< HEAD
    insert the allreudce and scale ops for gradients of model
=======
    insert the allreudce and scale ops for gradients of model 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    parameters for operator in data parallelism.

    Args:
        dist_ctx (DistributedContext): dist context.
<<<<<<< HEAD
        op (Operator): the current (backward) operator which might need.
        allreduce_var_names (list): list of the parameter's grads variable name in the current operator output.
=======
        op (Operator): the current (backward) operator which might need. 
        allreduce_var_names (list): list of the parameter's grads variable name in the current operator output. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """

    op_dist_attr = dist_ctx.get_op_dist_attr_for_program(op)
    process_mesh = op_dist_attr.process_mesh
    dist_op_context = dist_ctx.dist_op_context
    main_block = dist_op_context.work_block
    dp_degree = len(dp_group.ranks)

    for var_name in allreduce_var_names:
        added_ops = []
        grad_var = main_block.var(var_name)
<<<<<<< HEAD
        allreduce_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': [grad_var]},
            outputs={'Out': [grad_var]},
            attrs={
                'ring_id': dp_group.id,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Backward,
            },
        )
        allreduce_op._set_attr(
            'op_namescope', str('/') + ParallelMode.DataParallel
        )
        added_ops.append(allreduce_op)

        if dist_ctx.gradient_scale:
            scale_op = main_block.append_op(
                type='scale',
                inputs={'X': grad_var},
                outputs={'Out': grad_var},
                attrs={'scale': 1.0 / dp_degree, OP_ROLE_KEY: OpRole.Backward},
            )
            scale_op._set_attr(
                'op_namescope', str('/') + ParallelMode.DataParallel
            )
            added_ops.append(scale_op)

        dims_mapping = op_dist_attr.get_output_dims_mapping(grad_var.name)
        assert (
            dims_mapping is not None
        ), "Unexpected: dims_mapping of output [{}] of op [{}] is None".format(
            grad_var.name, op_dist_attr.op_type
        )
        # NOTE auxiliary op's dist attr should follow dist_op not dist_tensor
        for new_op in added_ops:
            new_op_attr = OperatorDistAttr()
=======
        allreduce_op = main_block.append_op(type='c_allreduce_sum',
                                            inputs={'X': [grad_var]},
                                            outputs={'Out': [grad_var]},
                                            attrs={
                                                'ring_id': dp_group.id,
                                                'use_calc_stream': True,
                                                OP_ROLE_KEY: OpRole.Backward
                                            })
        allreduce_op._set_attr('op_namescope',
                               str('/') + ParallelMode.DataParallel)
        added_ops.append(allreduce_op)

        if dist_ctx.gradient_scale:
            scale_op = main_block.append_op(type='scale',
                                            inputs={'X': grad_var},
                                            outputs={'Out': grad_var},
                                            attrs={
                                                'scale': 1.0 / dp_degree,
                                                OP_ROLE_KEY: OpRole.Backward
                                            })
            scale_op._set_attr('op_namescope',
                               str('/') + ParallelMode.DataParallel)
            added_ops.append(scale_op)

        dims_mapping = op_dist_attr.get_output_dims_mapping(grad_var.name)
        assert dims_mapping is not None, "Unexception: dims_mapping of output [{}] of op [{}] is None".format(
            grad_var.name, op_dist_attr.op_type)
        # NOTE auxiliary op's dist attr should follow dist_op not dist_tensor
        for new_op in added_ops:
            new_op_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            new_op_attr.process_mesh = process_mesh
            new_op_attr.set_output_dims_mapping(grad_var.name, dims_mapping)
            new_op_attr.set_input_dims_mapping(grad_var.name, dims_mapping)
            dist_ctx.set_op_dist_attr_for_program(new_op, new_op_attr)


<<<<<<< HEAD
def gradient_synchronization(
    dist_ctx, op, act_grad_names, out_grad_names, rank
):
    """
    conduct the allreudce and scaling（dp size）for gradients of model
=======
def gradient_synchronization(dist_ctx, op, act_grad_names, out_grad_names,
                             rank):
    """
    conduct the allreudce and scaling（dp size）for gradients of model 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    parameters for operator in data parallelism.

    Args:
        dist_ctx (DistributedContext): dist context.
<<<<<<< HEAD
        op (Operator): the current (backward) operator which might need.
        act_grad_names (list): list of input activation grads variable name to the current operator.
        out_grad_names (list): list of the output parameter's grads variable name of the current operator.
=======
        op (Operator): the current (backward) operator which might need. 
        act_grad_names (list): list of input activation grads variable name to the current operator. 
        out_grad_names (list): list of the output parameter's grads variable name of the current operator. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        rank (int): global ranks index for current process.
    """

    if not is_in_backward_phase(dist_ctx):
        return

<<<<<<< HEAD
    if (
        is_optimize_op(op)
        or len(act_grad_names) == 0
        or len(out_grad_names) == 0
    ):
=======
    if is_optimize_op(op) or len(act_grad_names) == 0 or len(
            out_grad_names) == 0:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return

    dp_group = get_data_parallel_group(dist_ctx, op, act_grad_names, rank)

    if not dp_group:
        return

    sync_and_scale_gradients(dist_ctx, op, dp_group, out_grad_names)


def is_data_parallel_scale_op(op):
<<<<<<< HEAD
    return (
        op.type == "scale"
        and op.desc.has_attr("op_namescope")
        and ParallelMode.DataParallel in op.desc.attr("op_namescope")
    )


def is_data_parallel_reduce_op(op):
    return (
        op.type in ["c_reduce_sum", "c_allreduce_sum"]
        and op.desc.has_attr("op_namescope")
        and ParallelMode.DataParallel in op.desc.attr("op_namescope")
    )


def is_amp_flag_sync_op(op):
    return (
        op.type == "c_allreduce_max"
        and op.desc.has_attr("op_namescope")
        and SyncMode.AmpFlagSync in op.desc.attr("op_namescope")
    )


def is_global_norm_sync_op(op):
    return (
        op.type == "c_allreduce_sum"
        and op.desc.has_attr("op_namescope")
        and SyncMode.GlobalNormSync in op.desc.attr("op_namescope")
    )
=======
    return op.type == "scale" and op.desc.has_attr("op_namescope") \
            and ParallelMode.DataParallel in op.desc.attr("op_namescope")


def is_data_parallel_reduce_op(op):
    return op.type in ["c_reduce_sum", "c_allreduce_sum"] and op.desc.has_attr("op_namescope") \
            and ParallelMode.DataParallel in op.desc.attr("op_namescope")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def is_in_backward_phase(dist_ctx):
    # NOTE currently high-order differential in Paddle dose NOT distinguish gradient computation operators
    # in Forward phase and operators in Backward phase (both with op_role=1), which will mislead
    # auto parallel to add gradient synchronization for gradient computation operators in Forward phase.
    # we use this FLAG to distinguish these two phases temporarily.

    return dist_ctx.dist_op_context.in_backward_phase()
