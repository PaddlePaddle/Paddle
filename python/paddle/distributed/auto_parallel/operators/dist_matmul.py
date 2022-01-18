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

import copy
from .common import infer_shape
from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from .common import set_comm_op_dist_attr_for_program, naive_copy_op_dist_attr_for_program, is_parameter_related
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from ..utils import set_dist_op_desc_original_id
from ..dist_attribute import OperatorDistributedAttribute
from paddle.fluid import core, unique_name
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..process_group import new_process_group
from ..utils import _get_comm_group, _get_corresponding_rank
from .dist_default import DistributedDefaultImpl0


def copy_op_with_new_input_output(ctx, block, src_op, **kwargs):
    dist_op_desc = block.desc.append_op()
    dist_op_desc.copy_from(src_op.desc)
    set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
    for input_name in src_op.desc.input_names():
        assert input_name in kwargs
        dist_op_desc.set_input(input_name, kwargs[input_name])
    for output_name in src_op.desc.output_names():
        assert input_name in kwargs
        dist_op_desc.set_output(output_name, kwargs[output_name])

    block._sync_with_cpp()
    return dist_op_desc


def _update_dims_mapping_for_matmul(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    op_dist_attr = dist_op.dist_attr
    x_name = op_desc.input('X')[0]
    y_name = op_desc.input('Y')[0]
    out_name = op_desc.output('Out')[0]
    x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
    y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
    out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
    x_dims_mapping_len = len(x_dims_mapping)
    y_dims_mapping_len = len(y_dims_mapping)
    out_dims_mapping_len = len(out_dims_mapping)

    # Add dim mapping to Make sure the length dims_mapping be at least 2
    if x_dims_mapping_len == 1:
        x_dims_mapping.insert(0, -1)
    if y_dims_mapping_len == 1:
        y_dims_mapping.insert(1, -1)

    # Deal with dim > 2 and take care of broadcasting
    if out_dims_mapping_len > 2:
        broadcast_x_dims_mapping = []
        broadcast_y_dims_mapping = []
        broadcast_out_dims_mapping = []

        for i in range(out_dims_mapping_len - x_dims_mapping_len):
            broadcast_x_dims_mapping.append(out_dims_mapping[i])
        for i in range(x_dims_mapping_len - 2):
            broadcast_x_dims_mapping.append(x_dims_mapping[i])

        for i in range(out_dims_mapping_len - y_dims_mapping_len):
            broadcast_y_dims_mapping.append(out_dims_mapping[i])
        for i in range(y_dims_mapping_len - 2):
            broadcast_y_dims_mapping.append(y_dims_mapping[i])

        for i in range(out_dims_mapping_len - 2):
            broadcast_out_dims_mapping.append(out_dims_mapping[i])

        compatible_dims_mapping = compute_compatible_dims_mapping([
            broadcast_x_dims_mapping, broadcast_y_dims_mapping,
            broadcast_out_dims_mapping
        ])
        assert compatible_dims_mapping is not None, "There is no compatible dim mapping."

        for i in range(x_dims_mapping_len - 2):
            new_idx = i + (out_dims_mapping_len - x_dims_mapping_len)
            if x_dims_mapping[i] != compatible_dims_mapping[new_idx]:
                x_dims_mapping[i] = compatible_dims_mapping[new_idx]
                changed = True

        for i in range(y_dims_mapping_len - 2):
            new_idx = i + (out_dims_mapping_len - y_dims_mapping_len)
            if y_dims_mapping[i] != compatible_dims_mapping[new_idx]:
                y_dims_mapping[i] = compatible_dims_mapping[new_idx]
                changed = True

        for i in range(out_dims_mapping_len - 2):
            if out_dims_mapping[i] != compatible_dims_mapping[i]:
                out_dims_mapping[i] = compatible_dims_mapping[i]
                changed = True

    # The following which uses negative index can be work
    # when len(out_dims_mapping) > 2 and len(out_dims_mapping) <=2
    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, y_dims_mapping], [-1, -2])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, out_dims_mapping], [-2, -2])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
        [y_dims_mapping, out_dims_mapping], [-1, -1])
    if dim_changed:
        changed = True

    # Remove unnecessary dim mapping to make sure the length of dims_mapping is same as its tensor
    if x_dims_mapping_len == 1:
        x_dims_mapping.pop(0)
    if y_dims_mapping_len == 1:
        y_dims_mapping.pop(1)

    assert len(x_dims_mapping) == x_dims_mapping_len
    assert len(y_dims_mapping) == y_dims_mapping_len
    assert len(out_dims_mapping) == out_dims_mapping_len

    return changed


def _is_auto_compatible_for_matmul(dist_op):
    op_desc = dist_op.serial_op.desc
    op_dist_attr = dist_op.dist_attr
    x_name = op_desc.input('X')[0]
    y_name = op_desc.input('Y')[0]
    out_name = op_desc.output('Out')[0]
    # Deep copy these dims_mappings for keeping them unchanged.
    x_dims_mapping = copy.deepcopy(op_dist_attr.get_input_dims_mapping(x_name))
    y_dims_mapping = copy.deepcopy(op_dist_attr.get_input_dims_mapping(y_name))
    out_dims_mapping = copy.deepcopy(
        op_dist_attr.get_output_dims_mapping(out_name))
    x_dims_mapping_len = len(x_dims_mapping)
    y_dims_mapping_len = len(y_dims_mapping)
    out_dims_mapping_len = len(out_dims_mapping)

    # Add dim mapping to Make sure the length dims_mapping be at least 2
    if x_dims_mapping_len == 1:
        x_dims_mapping.insert(0, -1)
    if y_dims_mapping_len == 1:
        y_dims_mapping.insert(1, -1)

    # Deal with dim > 2 and take care of broadcasting
    if out_dims_mapping_len > 2:
        broadcast_x_dims_mapping = []
        broadcast_y_dims_mapping = []
        broadcast_out_dims_mapping = []

        for i in range(out_dims_mapping_len - x_dims_mapping_len):
            broadcast_x_dims_mapping.append(out_dims_mapping[i])
        for i in range(x_dims_mapping_len - 2):
            broadcast_x_dims_mapping.append(x_dims_mapping[i])

        for i in range(out_dims_mapping_len - y_dims_mapping_len):
            broadcast_y_dims_mapping.append(out_dims_mapping[i])
        for i in range(y_dims_mapping_len - 2):
            broadcast_y_dims_mapping.append(y_dims_mapping[i])

        for i in range(out_dims_mapping_len - 2):
            broadcast_out_dims_mapping.append(out_dims_mapping[i])

        is_same = ((broadcast_x_dims_mapping == broadcast_y_dims_mapping) and
                   (broadcast_x_dims_mapping == broadcast_out_dims_mapping))
        if not is_same:
            return False

    # The following which uses negative index can be work
    # when len(out_dims_mapping) > 2 and len(out_dims_mapping) <=2
    is_same = (x_dims_mapping[-1] == y_dims_mapping[-2])
    if not is_same:
        return False

    is_same = (x_dims_mapping[-2] == out_dims_mapping[-2])
    if not is_same:
        return False

    is_same = (y_dims_mapping[-1] == out_dims_mapping[-1])
    if not is_same:
        return False

    return True


def _right_operand_parameter_matmul_backward(ctx, *args, **kwargs):

    # by now the backward function only insert the gradient allreduce for dist op itself

    dist_op_context = ctx.dist_op_context
    main_block = dist_op_context.get_dst_main_program().global_block()
    backward_op = dist_op_context.get_cur_src_op()
    rank_id = dist_op_context.get_rank_id()
    dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
    assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
        str(backward_op))

    # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
    if rank_id not in dist_attr.process_mesh.processes:
        rank_id = _get_corresponding_rank(ctx, dist_attr.process_mesh, rank_id)

    assert 'Y' in kwargs, "input [{}] is not given".format('Y')
    assert 'X' in kwargs, "input [{}] is not given".format('X')
    assert 'Out@GRAD' in kwargs, "input [{}] is not given".format('Out@GRAD')
    assert 'Y@GRAD' in kwargs, "output [{}] is not given".format('Y@GRAD')
    assert 'X@GRAD' in kwargs, "output [{}] is not given".format('X@GRAD')
    assert len(
        kwargs['Y']
    ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
        kwargs['Y'])
    assert len(
        kwargs['X']
    ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
        kwargs['X'])
    assert len(
        kwargs['Out@GRAD']
    ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
        kwargs['Out'])
    assert len(
        kwargs['Y@GRAD']
    ) == 1, "row_parallel_embedding output Ids take 1 variable but got {}".format(
        kwargs['Y@GRAD'])

    X_var = main_block.var(kwargs['X'][0])
    Y_var = main_block.var(kwargs['Y'][0])
    Out_grad = main_block.var(kwargs['Out@GRAD'][0])
    Y_grad = main_block.var(kwargs['Y@GRAD'][0])

    assert not is_parameter_related(
        X_var.name, main_block
    ), "left operand(X) [{}] of dist matmul should not be parameter".format(
        X_var.name)

    Y_var_dim_mapping = dist_attr.get_input_dims_mapping(Y_var.name)
    process_mesh_shape = dist_attr.process_mesh.topology
    process_mesh_group = dist_attr.process_mesh.processes
    # assert len(
    #     Y_var_dim_mapping
    # ) == 2, "dist matmual only support Y operand with 2 dims now but Y({})'s dim is [{}]".format(
    #     Y_var.name, Y_var_dim_mapping)
    Y_var_partitioned = False
    for dim in Y_var_dim_mapping:
        if dim >= 0 and process_mesh_shape[dim] > 0:
            Y_var_partitioned = True
            break

    if is_parameter_related(Y_var.name, main_block) and Y_var_partitioned:

        if Y_var_dim_mapping[0] >= 0:
            # row parallel: c_identity + matmul
            assert Y_var_dim_mapping[1] < 0
            parallel_axis = Y_var_dim_mapping[0]

            check_variable_and_dtype(
                Out_grad, 'tensor',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                '_c_identity')

            intermediate_var_0 = main_block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    ["c_identity", 'tmp'])) + "@GRAD",
                dtype=Out_grad.dtype,
                shape=Out_grad.shape,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=Out_grad.stop_gradient)

            # copy X_var's dist_attr to intermediate_var_0's dist_attr
            out_grad_dist_attr = dist_attr.get_input_dist_attr(Out_grad.name)
            assert out_grad_dist_attr is not None
            ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                                 out_grad_dist_attr)

            group_ranks = _get_comm_group(
                process_mesh_group, process_mesh_shape, parallel_axis, rank_id)
            group = new_process_group(group_ranks)
            c_identity_op = main_block.append_op(
                type='c_identity',
                inputs={'X': [Out_grad]},
                outputs={'Out': intermediate_var_0},
                attrs={
                    'ring_id': group.id,
                    'use_calc_stream': True,
                    'use_model_parallel': True,
                    OP_ROLE_KEY: OpRole.Backward,
                })
            check_variable_and_dtype(intermediate_var_0, 'x',
                                     ['float16', 'float32', 'float64'],
                                     'linear')
            check_dtype(intermediate_var_0.dtype, 'dtype',
                        ['float16', 'float32', 'float64'], 'linear')
            set_comm_op_dist_attr_for_program(
                c_identity_op, dist_attr.process_mesh, out_grad_dist_attr, ctx)

            new_kwargs = copy.deepcopy(kwargs)
            new_kwargs['Out@GRAD'] = [intermediate_var_0.name]
            matmul_op_desc = copy_op_with_new_input_output(
                ctx, main_block, backward_op, **new_kwargs)
        else:
            # col parallel: matmul + allreduce
            assert Y_var_dim_mapping[0] < 0
            parallel_axis = Y_var_dim_mapping[1]
            new_kwargs = copy.deepcopy(kwargs)

            # NOTE (JZ-LIANG) should allow left operand be empty for matmul grad
            has_x_grad = len(kwargs['X@GRAD']) > 0
            if has_x_grad:
                assert len(kwargs['X@GRAD']) == 1
                X_grad = main_block.var(kwargs['X@GRAD'][0])
                intermediate_var_0 = main_block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        ["c_identity", 'tmp'])) + "@GRAD",
                    dtype=X_grad.dtype,
                    shape=X_grad.shape,
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    persistable=False,
                    stop_gradient=X_grad.stop_gradient)

                X_grad_dist_attr = dist_attr.get_output_dist_attr(X_grad.name)
                assert X_grad_dist_attr is not None
                ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                                     X_grad_dist_attr)
                new_kwargs['X@GRAD'] = [intermediate_var_0.name]

            matmul_op_desc = copy_op_with_new_input_output(
                ctx, main_block, backward_op, **new_kwargs)

            # NOTE (JZ-LIANG) trick to skip one allreduce if left operand has not grad
            if has_x_grad:
                group_ranks = _get_comm_group(process_mesh_group,
                                              process_mesh_shape, parallel_axis,
                                              rank_id)
                group = new_process_group(group_ranks)
                c_allreduce_sum_op = main_block.append_op(
                    type='c_allreduce_sum',
                    inputs={'X': [intermediate_var_0.name]},
                    outputs={'Out': kwargs['X@GRAD']},
                    attrs={
                        'ring_id': group.id,
                        'use_calc_stream': True,
                        'use_model_parallel': True,
                        OP_ROLE_KEY: OpRole.Backward
                    })
                set_comm_op_dist_attr_for_program(c_allreduce_sum_op,
                                                  dist_attr.process_mesh,
                                                  X_grad_dist_attr, ctx)
    else:
        # replicate
        matmul_op_desc = copy_op_with_new_input_output(ctx, main_block,
                                                       backward_op, **kwargs)

    main_block._sync_with_cpp()

    # check if need gradient allreduce
    need_gradient_allreduce = False

    process_mesh = dist_attr.process_mesh
    var_dim_mapping = dist_attr.get_input_dims_mapping(X_var.name)
    mesh_shape = process_mesh.topology
    batch_size_axis = var_dim_mapping[0]
    if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
        need_gradient_allreduce = True
        group_ranks = _get_comm_group(process_mesh.processes,
                                      process_mesh.topology, batch_size_axis,
                                      rank_id)
        dp_degree = len(group_ranks)
        dp_group = new_process_group(group_ranks)

    if need_gradient_allreduce and is_parameter_related(Y_var.name, main_block):
        Y_Grad_var = main_block.var(kwargs['Y@GRAD'][0])
        allreduce_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': [Y_Grad_var]},
            outputs={'Out': [Y_Grad_var]},
            attrs={
                'ring_id': dp_group.id,
                'use_calc_stream': True,
                OP_ROLE_KEY: OpRole.Backward
            })
        scale_op = main_block.append_op(
            type='scale',
            inputs={'X': Y_Grad_var},
            outputs={'Out': Y_Grad_var},
            attrs={'scale': 1.0 / dp_degree,
                   OP_ROLE_KEY: OpRole.Backward})
        main_block._sync_with_cpp()

        dims_mapping = ctx.get_tensor_dist_attr_for_program(
            Y_Grad_var).dims_mapping
        process_mesh = dist_attr.process_mesh
        for op in [allreduce_op, scale_op]:
            op_attr = OperatorDistributedAttribute()
            op_attr.process_mesh = process_mesh
            op_attr.set_output_dims_mapping(Y_Grad_var.name, dims_mapping)
            op_attr.set_input_dims_mapping(Y_Grad_var.name, dims_mapping)
            ctx.set_op_dist_attr_for_program(op, op_attr)


def _init_param_sync(Weight_var, dist_op_context, startup_block, ctx, rank_id):

    assert Weight_var.name not in dist_op_context.already_init_sync_vars
    assert startup_block.has_var(Weight_var.name)
    dist_op_context.already_init_sync_vars.add(Weight_var.name)
    param = startup_block.var(Weight_var.name)
    param_dist_attr = ctx.get_tensor_dist_attr_for_program(param)
    process_mesh = param_dist_attr.process_mesh
    dim_mapping = param_dist_attr.dims_mapping

    for axis, size in enumerate(process_mesh.topology):
        if size <= 1 or axis in dim_mapping:
            pass
        else:
            group_ranks = _get_comm_group(process_mesh.processes,
                                          process_mesh.topology, axis, rank_id)
            sync_group = new_process_group(group_ranks)

            startup_block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': sync_group.id,
                    'root': 0,
                    'use_calc_stream': True,
                    OP_ROLE_KEY: OpRole.Forward
                })
    startup_block._sync_with_cpp()


class DistributedMatmul(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super(DistributedMatmul, self).__init__(op_type)


register_distributed_operator_impl_container(DistributedMatmul("matmul"))


# ColumnParallel
class DistributedMatmulImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl0, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[-2]) or is_dim_replicate(y_dims_mapping[
                -1]):
            return False
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_replicate(out_dims_mapping[-1]):
            return False
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False
        if not _is_auto_compatible_for_matmul(dist_op):
            return False
        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(dist_op)
        if dim_changed:
            changed = True
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.get_dst_main_program().global_block()
        startup_block = dist_op_context.get_dst_startup_program().global_block()
        src_op = dist_op_context.get_cur_src_op()
        rank_id = dist_op_context.get_rank_id()
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name)
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block.var(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name)[1]
        assert matmul_col_dim_mapping >= 0, "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        # infer new var shape with op dist attr
        x_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(X_var)
        assert x_tensor_dist_attr is not None
        identity_var_dist_attr = op_dist_attr.get_input_dist_attr(X_var.name)
        assert identity_var_dist_attr is not None
        ref_shape_x = infer_shape(main_block, X_var, x_tensor_dist_attr,
                                  identity_var_dist_attr)
        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
        ref_shape_out = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                    out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_identity", 'tmp'])),
            dtype=X_var.dtype,
            shape=X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=X_var.stop_gradient)
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             identity_var_dist_attr)

        check_variable_and_dtype(
            X_var, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], '_c_identity')

        c_identity_op = main_block.append_op(
            type='c_identity',
            inputs={'X': [X_var]},
            outputs={'Out': intermediate_var_0},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
            })
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(intermediate_var_0, 'x',
                                 ['float16', 'float32', 'float64'], 'linear')
        check_dtype(intermediate_var_0.dtype, 'dtype',
                    ['float16', 'float32', 'float64'], 'linear')
        attrs = {
            'transpose_X': False,
            'transpose_Y': False,
            'alpha': 1,
        }
        inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
        matmul_op = main_block.append_op(
            type='matmul', inputs=inputs, outputs={'Out': Out_var}, attrs=attrs)
        if Out_var.shape != ref_shape_out:
            Out_var.desc.set_shape(ref_shape_out)

        # set dist op's dist_attr with serial op's dist_attr
        # c_identity
        identity_op_dist_attr = OperatorDistributedAttribute()
        identity_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        identity_op_dist_attr.impl_type = op_dist_attr.impl_type
        identity_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        # input
        input_varname = c_identity_op.desc.input_arg_names()[0]
        input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
        assert input_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr)
        identity_op_dist_attr.set_input_dist_attr(input_varname,
                                                  input_dist_attr)
        # output
        output_varname = c_identity_op.desc.output_arg_names()[0]
        identity_op_dist_attr.set_output_dist_attr(output_varname,
                                                   input_dist_attr)
        # set op dist attr
        ctx.set_op_dist_attr_for_program(c_identity_op, identity_op_dist_attr)

        # matmul
        matmul_op_dist_attr = OperatorDistributedAttribute()
        matmul_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmul_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmul_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        # input
        for input_varname in matmul_op.desc.input_arg_names():
            if input_varname in src_op.desc.input_arg_names():
                input_dist_attr = op_dist_attr.get_input_dist_attr(
                    input_varname)
                assert input_dist_attr is not None, "dist_attr is {}".format(
                    op_dist_attr)
                matmul_op_dist_attr.set_input_dist_attr(input_varname,
                                                        input_dist_attr)
            else:
                input_var = main_block.var(input_varname)
                tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    input_var)
                matmul_op_dist_attr.set_input_dist_attr(input_varname,
                                                        tensor_dist_attr)
        # output
        output_varname = matmul_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr)
        matmul_op_dist_attr.set_output_dist_attr(output_varname,
                                                 output_dist_attr)
        # set op dist attr
        ctx.set_op_dist_attr_for_program(matmul_op, matmul_op_dist_attr)

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            _init_param_sync(Weight_var, dist_op_context, startup_block, ctx,
                             rank_id)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# RowParallel
class DistributedMatmulImpl1(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl1, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[-2]) or is_dim_shard(y_dims_mapping[
                -1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_shard(out_dims_mapping[-1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        if not _is_auto_compatible_for_matmul(dist_op):
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(dist_op)
        if dim_changed:
            changed = True
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.get_dst_main_program().global_block()
        startup_block = dist_op_context.get_dst_startup_program().global_block()
        src_op = dist_op_context.get_cur_src_op()
        rank_id = dist_op_context.get_rank_id()
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name)
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block.var(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name)[0]
        assert matmul_row_dim_mapping >= 0, "row_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_row_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = matmul_row_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        check_variable_and_dtype(X_var, 'x', ['float16', 'float32', 'float64'],
                                 'linear')
        check_dtype(X_var.dtype, 'dtype', ['float16', 'float32', 'float64'],
                    'linear')
        attrs = {
            'transpose_X': False,
            'transpose_Y': False,
            'alpha': 1,
        }
        inputs = {'X': X_var, 'Y': Weight_var}

        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
        ref_shape = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            shape=Out_var.shape,
            dtype=Out_var.dtype,
            type=Out_var.type,
            lod_level=Out_var.lod_level,
            persistable=False,
            is_data=False,
            need_check_feed=Out_var.desc.need_check_feed())
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             out_var_dist_attr)

        matmul_op = main_block.append_op(
            type='matmul',
            inputs=inputs,
            outputs={'Out': intermediate_var_0},
            attrs=attrs)
        if intermediate_var_0.shape != ref_shape:
            intermediate_var_0.desc.set_shape(ref_shape)

        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': intermediate_var_0},
            outputs={'Out': Out_var},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True
            })
        if Out_var.shape != ref_shape:
            Out_var.desc.set_shape(ref_shape)

        # set dist op's dist_attr with serial op's dist_attr
        # matmul
        matmul_op_dist_attr = OperatorDistributedAttribute()
        matmul_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmul_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmul_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in matmul_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr)
            matmul_op_dist_attr.set_input_dist_attr(input_varname,
                                                    input_dist_attr)
        output_varname = matmul_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr)
        matmul_op_dist_attr.set_output_dist_attr(output_varname,
                                                 output_dist_attr)
        ctx.set_op_dist_attr_for_program(matmul_op, matmul_op_dist_attr)

        # allreduce
        allreduce_op_dist_attr = OperatorDistributedAttribute()
        allreduce_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        allreduce_op_dist_attr.impl_type = op_dist_attr.impl_type
        allreduce_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in c_allreduce_sum_op.desc.input_arg_names():
            input_var = main_block.var(input_varname)
            tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(input_var)
            assert tensor_dist_attr is not None
            allreduce_op_dist_attr.set_input_dist_attr(input_varname,
                                                       tensor_dist_attr)
        for output_varname in c_allreduce_sum_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr)
            allreduce_op_dist_attr.set_output_dist_attr(output_varname,
                                                        output_dist_attr)
        ctx.set_op_dist_attr_for_program(c_allreduce_sum_op,
                                         allreduce_op_dist_attr)

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            _init_param_sync(Weight_var, dist_op_context, startup_block, ctx,
                             rank_id)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# ReplicateParallel
class DistributedMatmulImpl2(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl2, self).__init__(name)

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping,
                               -2) and is_dim_shard(x_dims_mapping[-2]):
            return False

        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping,
                               -2) and is_dim_shard(y_dims_mapping[-2]):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if is_dim_shard(out_dims_mapping[-1]):
            return False
        if is_valid_list_index(out_dims_mapping,
                               -2) and is_dim_shard(out_dims_mapping[-2]):
            return False

        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        if not _is_auto_compatible_for_matmul(dist_op):
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(dist_op)
        if dim_changed:
            changed = True
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl0("column_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl1("row_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl2("replicate_parallel"))


class DistributedMatmulV2(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super(DistributedMatmulV2, self).__init__(op_type)


register_distributed_operator_impl_container(DistributedMatmulV2("matmul_v2"))


# ColumnParallel
class DistributedMatmulV2Impl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulV2Impl0, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[-2]) or is_dim_replicate(y_dims_mapping[
                -1]):
            return False
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_replicate(out_dims_mapping[-1]):
            return False
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        if not _is_auto_compatible_for_matmul(dist_op):
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(dist_op)
        if dim_changed:
            changed = True
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.get_dst_main_program().global_block()
        startup_block = dist_op_context.get_dst_startup_program().global_block()
        src_op = dist_op_context.get_cur_src_op()
        rank_id = dist_op_context.get_rank_id()
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name)
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block.var(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name)[1]
        assert matmul_col_dim_mapping >= 0, "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        # infer new var shape with op dist attr
        x_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(X_var)
        assert x_tensor_dist_attr is not None
        identity_var_dist_attr = op_dist_attr.get_input_dist_attr(X_var.name)
        assert identity_var_dist_attr is not None
        ref_shape_x = infer_shape(main_block, X_var, x_tensor_dist_attr,
                                  identity_var_dist_attr)
        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
        ref_shape_out = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                    out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_identity", 'tmp'])),
            dtype=X_var.dtype,
            shape=X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=X_var.stop_gradient)
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             identity_var_dist_attr)

        check_variable_and_dtype(
            X_var, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], '_c_identity')
        c_identity_op = main_block.append_op(
            type='c_identity',
            inputs={'X': [X_var]},
            outputs={'Out': intermediate_var_0},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
            })
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(intermediate_var_0, 'x',
                                 ['float16', 'float32', 'float64'], 'linear')
        check_dtype(intermediate_var_0.dtype, 'dtype',
                    ['float16', 'float32', 'float64'], 'linear')
        attrs = {'trans_x': False, 'trans_y': False}
        inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
        matmul_v2_op = main_block.append_op(
            type='matmul_v2',
            inputs=inputs,
            outputs={'Out': Out_var},
            attrs=attrs)
        if Out_var.shape != ref_shape_out:
            Out_var.desc.set_shape(ref_shape_out)

        # set dist op's dist_attr with serial op's dist_attr
        # c_identity
        identity_op_dist_attr = OperatorDistributedAttribute()
        identity_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        identity_op_dist_attr.impl_type = op_dist_attr.impl_type
        identity_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        # input
        input_varname = c_identity_op.desc.input_arg_names()[0]
        input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
        assert input_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr)
        identity_op_dist_attr.set_input_dist_attr(input_varname,
                                                  input_dist_attr)
        # output
        output_varname = c_identity_op.desc.output_arg_names()[0]
        identity_op_dist_attr.set_output_dist_attr(output_varname,
                                                   input_dist_attr)
        ctx.set_op_dist_attr_for_program(c_identity_op, identity_op_dist_attr)

        # matmulv2
        matmulv2_op_dist_attr = OperatorDistributedAttribute()
        matmulv2_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmulv2_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmulv2_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in matmul_v2_op.desc.input_arg_names():
            if input_varname in src_op.desc.input_arg_names():
                input_dist_attr = op_dist_attr.get_input_dist_attr(
                    input_varname)
                assert input_dist_attr is not None, "dist_attr is {}".format(
                    op_dist_attr)
                matmulv2_op_dist_attr.set_input_dist_attr(input_varname,
                                                          input_dist_attr)
            else:
                input_var = main_block.var(input_varname)
                tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    input_var)
                matmulv2_op_dist_attr.set_input_dist_attr(input_varname,
                                                          tensor_dist_attr)
        for output_varname in matmul_v2_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr)
            matmulv2_op_dist_attr.set_output_dist_attr(output_varname,
                                                       output_dist_attr)
        ctx.set_op_dist_attr_for_program(matmul_v2_op, matmulv2_op_dist_attr)

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            _init_param_sync(Weight_var, dist_op_context, startup_block, ctx,
                             rank_id)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# RowParallel
class DistributedMatmulV2Impl1(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulV2Impl1, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[-2]) or is_dim_shard(y_dims_mapping[
                -1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_shard(out_dims_mapping[-1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        if not _is_auto_compatible_for_matmul(dist_op):
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(dist_op)
        if dim_changed:
            changed = True
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.get_dst_main_program().global_block()
        startup_block = dist_op_context.get_dst_startup_program().global_block()
        src_op = dist_op_context.get_cur_src_op()
        rank_id = dist_op_context.get_rank_id()
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
                input_name)
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block.var(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name)[0]
        assert matmul_row_dim_mapping >= 0, "row_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_row_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = matmul_row_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        check_variable_and_dtype(X_var, 'x', ['float16', 'float32', 'float64'],
                                 'linear')
        check_dtype(X_var.dtype, 'dtype', ['float16', 'float32', 'float64'],
                    'linear')
        attrs = {'trans_x': False, 'trans_y': False}
        inputs = {'X': X_var, 'Y': Weight_var}

        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
        ref_shape = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            shape=Out_var.shape,
            dtype=Out_var.dtype,
            type=Out_var.type,
            lod_level=Out_var.lod_level,
            persistable=False,
            is_data=False,
            need_check_feed=Out_var.desc.need_check_feed())
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             out_var_dist_attr)

        matmul_v2_op = main_block.append_op(
            type='matmul_v2',
            inputs=inputs,
            outputs={'Out': intermediate_var_0},
            attrs=attrs)
        if intermediate_var_0.shape != ref_shape:
            intermediate_var_0.desc.set_shape(ref_shape)

        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': intermediate_var_0},
            outputs={'Out': Out_var},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True
            })
        if Out_var.shape != ref_shape:
            Out_var.desc.set_shape(ref_shape)

        # set dist op's dist_attr with serial op's dist_attr
        # matmulv2
        matmulv2_op_dist_attr = OperatorDistributedAttribute()
        matmulv2_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmulv2_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmulv2_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in matmul_v2_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr)
            matmulv2_op_dist_attr.set_input_dist_attr(input_varname,
                                                      input_dist_attr)
        output_varname = matmul_v2_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr)
        matmulv2_op_dist_attr.set_output_dist_attr(output_varname,
                                                   output_dist_attr)
        ctx.set_op_dist_attr_for_program(matmul_v2_op, matmulv2_op_dist_attr)

        # allreduce
        allreduce_op_dist_attr = OperatorDistributedAttribute()
        allreduce_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        allreduce_op_dist_attr.impl_type = op_dist_attr.impl_type
        allreduce_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in c_allreduce_sum_op.desc.input_arg_names():
            input_var = main_block.var(input_varname)
            tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(input_var)
            assert tensor_dist_attr is not None
            allreduce_op_dist_attr.set_input_dist_attr(input_varname,
                                                       tensor_dist_attr)
        for output_varname in c_allreduce_sum_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr)
            allreduce_op_dist_attr.set_output_dist_attr(output_varname,
                                                        output_dist_attr)
        ctx.set_op_dist_attr_for_program(c_allreduce_sum_op,
                                         allreduce_op_dist_attr)

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            _init_param_sync(Weight_var, dist_op_context, startup_block, ctx,
                             rank_id)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# ReplicateParallel
class DistributedMatmulV2Impl2(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulV2Impl2, self).__init__(name)

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping,
                               -2) and is_dim_shard(x_dims_mapping[-2]):
            return False

        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping,
                               -2) and is_dim_shard(y_dims_mapping[-2]):
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if is_dim_shard(out_dims_mapping[-1]):
            return False
        if is_valid_list_index(out_dims_mapping,
                               -2) and is_dim_shard(out_dims_mapping[-2]):
            return False

        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        if not _is_auto_compatible_for_matmul(dist_op):
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(dist_op)
        if dim_changed:
            changed = True
        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


register_distributed_operator_impl("matmul_v2",
                                   DistributedMatmulV2Impl0("column_parallel"))
register_distributed_operator_impl("matmul_v2",
                                   DistributedMatmulV2Impl1("row_parallel"))
register_distributed_operator_impl(
    "matmul_v2", DistributedMatmulV2Impl2("replicate_parallel"))
