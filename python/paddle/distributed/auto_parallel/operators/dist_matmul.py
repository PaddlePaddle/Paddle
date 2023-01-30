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

<<<<<<< HEAD
from paddle.common_ops_import import check_dtype, check_variable_and_dtype
from paddle.distributed.auto_parallel.cost.comm_op_cost import (
    AllreduceSumOpCost,
    IdentityOpCost,
)
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole
from paddle.framework import core
from paddle.utils import unique_name

from ..cost import (
    MatmulGradOpCost,
    MatmulOpCost,
    MatmulV2GradOpCost,
    MatmulV2OpCost,
    MulGradOpCost,
    MulOpCost,
    build_comm_costs_from_descs,
    build_comm_desc_from_dist_op,
    build_comp_costs_from_descs,
    build_comp_desc_from_dist_op,
    build_dp_costs,
)
from ..dist_attribute import OperatorDistAttr
from ..process_group import new_process_group
from ..utils import (
    _get_comm_group,
    _get_corresponding_rank,
    compute_compatible_and_update_dim_mapping,
    compute_compatible_dims_mapping,
    is_dim_replicate,
    is_dim_shard,
    is_valid_list_index,
    set_dist_op_desc_original_id,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    gradient_synchronization,
    infer_shape,
    is_parameter_related,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    set_comm_op_dist_attr_for_program,
)
from .dist_default import DistributedDefaultImpl0
=======
from .common import infer_shape
from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from .common import gradient_synchronization
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
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..process_group import new_process_group
from ..utils import _get_comm_group, _get_corresponding_rank
from .dist_default import DistributedDefaultImpl0
from ..cost import build_comp_desc_from_dist_op, build_comm_desc_from_dist_op, build_dp_costs
from ..cost import build_comm_costs_from_descs, build_comp_costs_from_descs
from ..cost import MatmulV2OpCost, MatmulOpCost, MulOpCost
from ..cost import MatmulV2GradOpCost, MatmulGradOpCost, MulGradOpCost
from paddle.distributed.auto_parallel.cost.comm_op_cost import AllreduceSumOpCost, IdentityOpCost
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping):
    if trans_x:
<<<<<<< HEAD
        x_dims_mapping[-1], x_dims_mapping[-2] = (
            x_dims_mapping[-2],
            x_dims_mapping[-1],
        )
    if trans_y:
        y_dims_mapping[-1], y_dims_mapping[-2] = (
            y_dims_mapping[-2],
            y_dims_mapping[-1],
        )


def copy_op_with_new_input_output(ctx, block, src_op, **kwargs):
    pass

    src_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
    dist_attr = copy.deepcopy(src_dist_attr)
    dist_op = block.append_op(type='nop')
    dist_op_desc = dist_op.desc
=======
        x_dims_mapping[-1], x_dims_mapping[-2] = x_dims_mapping[
            -2], x_dims_mapping[-1]
    if trans_y:
        y_dims_mapping[-1], y_dims_mapping[-2] = y_dims_mapping[
            -2], y_dims_mapping[-1]


def copy_op_with_new_input_output(ctx, block, src_op, **kwargs):
    dist_op_desc = block.append_op(type='nop').desc
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    dist_op_desc.copy_from(src_op.desc)
    set_dist_op_desc_original_id(dist_op_desc, src_op.desc, ctx)
    for input_name in src_op.desc.input_names():
        assert input_name in kwargs
        dist_op_desc.set_input(input_name, kwargs[input_name])
<<<<<<< HEAD
        dist_attr.rename_input(
            src_op.desc.input(input_name)[0], kwargs[input_name][0]
        )
    for output_name in src_op.desc.output_names():
        assert output_name in kwargs
        dist_op_desc.set_output(output_name, kwargs[output_name])
        dist_attr.rename_output(
            src_op.desc.output(output_name)[0], kwargs[output_name][0]
        )
    # TODO: this call leads to a deepcopy when we init the dist op
    ctx.set_op_dist_attr_for_program(dist_op, dist_attr)
=======
    for output_name in src_op.desc.output_names():
        assert input_name in kwargs
        dist_op_desc.set_output(output_name, kwargs[output_name])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return dist_op_desc


def _update_dims_mapping_for_matmul(dist_op):
    changed = False
    op_desc = dist_op.serial_op.desc
    op_dist_attr = dist_op.dist_attr
    x_name = op_desc.input('X')[0]
    y_name = op_desc.input('Y')[0]
    out_name = op_desc.output('Out')[0]
    trans_x = None
    trans_y = None
    if op_desc.type() == "matmul_v2":
        trans_x = op_desc.attr('trans_x')
        trans_y = op_desc.attr('trans_y')
    elif op_desc.type() == "matmul":
        trans_x = op_desc.attr('transpose_X')
        trans_y = op_desc.attr('transpose_Y')
    x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
    y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
    out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
    x_dims_mapping_len = len(x_dims_mapping)
    y_dims_mapping_len = len(y_dims_mapping)
    out_dims_mapping_len = len(out_dims_mapping)

    # Add dim mapping to Make sure the length dims_mapping be at least 2
    if x_dims_mapping_len == 1:
        assert trans_x is False
        x_dims_mapping.insert(0, -1)
        out_dims_mapping.insert(out_dims_mapping_len - 1, 0)
    if y_dims_mapping_len == 1:
        assert trans_y is False
        y_dims_mapping.insert(1, -1)
        out_dims_mapping.insert(out_dims_mapping_len, 0)

    trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping)

    new_x_dims_mapping_len = len(x_dims_mapping)
    new_y_dims_mapping_len = len(y_dims_mapping)
    new_out_dims_mapping_len = len(out_dims_mapping)
    # Deal with dim > 2 and take care of broadcasting
    if new_out_dims_mapping_len > 2:
        broadcast_x_dims_mapping = []
        broadcast_y_dims_mapping = []
        broadcast_out_dims_mapping = []

        for i in range(new_out_dims_mapping_len - new_x_dims_mapping_len):
            broadcast_x_dims_mapping.append(out_dims_mapping[i])
        for i in range(new_x_dims_mapping_len - 2):
            broadcast_x_dims_mapping.append(x_dims_mapping[i])

        for i in range(new_out_dims_mapping_len - new_y_dims_mapping_len):
            broadcast_y_dims_mapping.append(out_dims_mapping[i])
        for i in range(new_y_dims_mapping_len - 2):
            broadcast_y_dims_mapping.append(y_dims_mapping[i])

        for i in range(new_out_dims_mapping_len - 2):
            broadcast_out_dims_mapping.append(out_dims_mapping[i])

<<<<<<< HEAD
        compatible_dims_mapping = compute_compatible_dims_mapping(
            [
                broadcast_x_dims_mapping,
                broadcast_y_dims_mapping,
                broadcast_out_dims_mapping,
            ]
        )
        if compatible_dims_mapping is None:
            trans_x_y_dims_mapping(
                trans_x, trans_y, x_dims_mapping, y_dims_mapping
            )
=======
        compatible_dims_mapping = compute_compatible_dims_mapping([
            broadcast_x_dims_mapping, broadcast_y_dims_mapping,
            broadcast_out_dims_mapping
        ])
        if compatible_dims_mapping is None:
            trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping,
                                   y_dims_mapping)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        for i in range(new_x_dims_mapping_len - 2):
            new_idx = i + (out_dims_mapping_len - new_x_dims_mapping_len)
            if x_dims_mapping[i] != compatible_dims_mapping[new_idx]:
                x_dims_mapping[i] = compatible_dims_mapping[new_idx]
                changed = True

        for i in range(new_y_dims_mapping_len - 2):
            new_idx = i + (out_dims_mapping_len - new_y_dims_mapping_len)
            if y_dims_mapping[i] != compatible_dims_mapping[new_idx]:
                y_dims_mapping[i] = compatible_dims_mapping[new_idx]
                changed = True

        for i in range(new_out_dims_mapping_len - 2):
            if out_dims_mapping[i] != compatible_dims_mapping[i]:
                out_dims_mapping[i] = compatible_dims_mapping[i]
                changed = True

    # The following which uses negative index can be work
    # when len(out_dims_mapping) > 2 and len(out_dims_mapping) <=2
    dim_changed = compute_compatible_and_update_dim_mapping(
<<<<<<< HEAD
        [x_dims_mapping, y_dims_mapping], [-1, -2]
    )
=======
        [x_dims_mapping, y_dims_mapping], [-1, -2])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
<<<<<<< HEAD
        [x_dims_mapping, out_dims_mapping], [-2, -2]
    )
=======
        [x_dims_mapping, out_dims_mapping], [-2, -2])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
<<<<<<< HEAD
        [y_dims_mapping, out_dims_mapping], [-1, -1]
    )
=======
        [y_dims_mapping, out_dims_mapping], [-1, -1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if dim_changed:
        changed = True

    trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping)

    # Remove unnecessary dim mapping to make sure the length of dims_mapping is same as its tensor
    if x_dims_mapping_len == 1:
        x_dims_mapping.pop(0)
        out_dims_mapping.pop(out_dims_mapping_len - 1)
    if y_dims_mapping_len == 1:
        y_dims_mapping.pop(1)
        out_dims_mapping.pop(out_dims_mapping_len)

    assert len(x_dims_mapping) == x_dims_mapping_len
    assert len(y_dims_mapping) == y_dims_mapping_len
    assert len(out_dims_mapping) == out_dims_mapping_len

<<<<<<< HEAD
    if changed:
        op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)
        op_dist_attr.set_input_dims_mapping(y_name, y_dims_mapping)
        op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return changed


def _is_auto_compatible_for_matmul(dist_op):
    op_desc = dist_op.serial_op.desc
    op_dist_attr = dist_op.dist_attr
    x_name = op_desc.input('X')[0]
    y_name = op_desc.input('Y')[0]
    out_name = op_desc.output('Out')[0]
    trans_x = None
    trans_y = None
    if op_desc.type() == "matmul_v2":
        trans_x = op_desc.attr('trans_x')
        trans_y = op_desc.attr('trans_y')
    elif op_desc.type() == "matmul":
        trans_x = op_desc.attr('transpose_X')
        trans_y = op_desc.attr('transpose_Y')

    # Deep copy these dims_mappings for keeping them unchanged.
    x_dims_mapping = copy.deepcopy(op_dist_attr.get_input_dims_mapping(x_name))
    y_dims_mapping = copy.deepcopy(op_dist_attr.get_input_dims_mapping(y_name))
    out_dims_mapping = copy.deepcopy(
<<<<<<< HEAD
        op_dist_attr.get_output_dims_mapping(out_name)
    )
=======
        op_dist_attr.get_output_dims_mapping(out_name))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    x_dims_mapping_len = len(x_dims_mapping)
    y_dims_mapping_len = len(y_dims_mapping)
    out_dims_mapping_len = len(out_dims_mapping)

    # Add dim mapping to Make sure the length dims_mapping be at least 2
    if x_dims_mapping_len == 1:
        x_dims_mapping.insert(0, -1)
    if y_dims_mapping_len == 1:
        y_dims_mapping.insert(1, -1)

    trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping)

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

<<<<<<< HEAD
        is_same = (broadcast_x_dims_mapping == broadcast_y_dims_mapping) and (
            broadcast_x_dims_mapping == broadcast_out_dims_mapping
        )
=======
        is_same = ((broadcast_x_dims_mapping == broadcast_y_dims_mapping)
                   and (broadcast_x_dims_mapping == broadcast_out_dims_mapping))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not is_same:
            return False

    # The following which uses negative index can be work
    # when len(out_dims_mapping) > 2 and len(out_dims_mapping) <=2
<<<<<<< HEAD
    is_same = x_dims_mapping[-1] == y_dims_mapping[-2]
    if not is_same:
        return False

    is_same = x_dims_mapping[-2] == out_dims_mapping[-2]
    if not is_same:
        return False

    is_same = y_dims_mapping[-1] == out_dims_mapping[-1]
=======
    is_same = (x_dims_mapping[-1] == y_dims_mapping[-2])
    if not is_same:
        return False

    is_same = (x_dims_mapping[-2] == out_dims_mapping[-2])
    if not is_same:
        return False

    is_same = (y_dims_mapping[-1] == out_dims_mapping[-1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if not is_same:
        return False

    return True


def _right_operand_parameter_matmul_backward(ctx, *args, **kwargs):

    # by now the backward function only insert the gradient allreduce for dist op itself

    dist_op_context = ctx.dist_op_context
    main_block = dist_op_context.work_block
    backward_op = dist_op_context.cur_src_op
    rank_id = dist_op_context.rank_id
    dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
<<<<<<< HEAD
    assert (
        dist_attr is not None
    ), "backward op [{}] don't have dist attribute !".format(str(backward_op))

    # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
    if rank_id not in dist_attr.process_mesh.process_ids:
=======
    assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
        str(backward_op))

    # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
    if rank_id not in dist_attr.process_mesh.processes:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        rank_id = _get_corresponding_rank(ctx, dist_attr.process_mesh, rank_id)

    assert 'Y' in kwargs, "input [{}] is not given".format('Y')
    assert 'X' in kwargs, "input [{}] is not given".format('X')
    assert 'Out@GRAD' in kwargs, "input [{}] is not given".format('Out@GRAD')
    assert 'Y@GRAD' in kwargs, "output [{}] is not given".format('Y@GRAD')
    assert 'X@GRAD' in kwargs, "output [{}] is not given".format('X@GRAD')
<<<<<<< HEAD
    assert (
        len(kwargs['Y']) == 1
    ), "row_parallel_embedding input Ids take 1 variable but got {}".format(
        kwargs['Y']
    )
    assert (
        len(kwargs['X']) == 1
    ), "row_parallel_embedding input Ids take 1 variable but got {}".format(
        kwargs['X']
    )
    assert (
        len(kwargs['Out@GRAD']) == 1
    ), "row_parallel_embedding input Ids take 1 variable but got {}".format(
        kwargs['Out']
    )
    assert (
        len(kwargs['Y@GRAD']) == 1
    ), "row_parallel_embedding output Ids take 1 variable but got {}".format(
        kwargs['Y@GRAD']
    )

    X_var = main_block._var_recursive(kwargs['X'][0])
    Y_var = main_block._var_recursive(kwargs['Y'][0])
    Out_grad = main_block._var_recursive(kwargs['Out@GRAD'][0])
    Y_grad = main_block._var_recursive(kwargs['Y@GRAD'][0])
=======
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
    Y_var = main_block._var_recursive(kwargs['Y'][0])
    Out_grad = main_block.var(kwargs['Out@GRAD'][0])
    Y_grad = main_block.var(kwargs['Y@GRAD'][0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    assert not is_parameter_related(
        X_var.name, main_block
    ), "left operand(X) [{}] of dist matmul should not be parameter".format(
<<<<<<< HEAD
        X_var.name
    )

    X_var_dims_mapping = dist_attr.get_input_dims_mapping(X_var.name)
    Y_var_dim_mapping = dist_attr.get_input_dims_mapping(Y_var.name)
    process_mesh_shape = dist_attr.process_mesh.shape
    process_mesh_group = dist_attr.process_mesh.process_ids
=======
        X_var.name)

    X_var_dims_mapping = dist_attr.get_input_dims_mapping(X_var.name)
    Y_var_dim_mapping = dist_attr.get_input_dims_mapping(Y_var.name)
    process_mesh_shape = dist_attr.process_mesh.topology
    process_mesh_group = dist_attr.process_mesh.processes
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    trans_x = None
    trans_y = None
    if backward_op.desc.type() == "matmul_v2_grad":
        trans_x = backward_op.desc.attr('trans_x')
        trans_y = backward_op.desc.attr('trans_y')
    elif backward_op.desc.type() == "matmul_grad":
        trans_x = backward_op.desc.attr('transpose_X')
        trans_y = backward_op.desc.attr('transpose_Y')

    if trans_y:
        trans_x_y_dims_mapping(False, True, None, Y_var_dim_mapping)

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
<<<<<<< HEAD
                Out_grad,
                'tensor',
                ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
                '_c_identity',
            )

            intermediate_var_0 = main_block.create_var(
                name=unique_name.generate_with_ignorable_key(
                    ".".join(["c_identity", 'tmp'])
                )
                + "@GRAD",
=======
                Out_grad, 'tensor',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                '_c_identity')

            intermediate_var_0 = main_block.create_var(
                name=unique_name.generate_with_ignorable_key(".".join(
                    ["c_identity", 'tmp'])) + "@GRAD",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                dtype=Out_grad.dtype,
                shape=Out_grad.shape,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
<<<<<<< HEAD
                stop_gradient=Out_grad.stop_gradient,
            )
=======
                stop_gradient=Out_grad.stop_gradient)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            # copy X_var's dist_attr to intermediate_var_0's dist_attr
            out_grad_dist_attr = dist_attr.get_input_dist_attr(Out_grad.name)
            assert out_grad_dist_attr is not None
<<<<<<< HEAD
            ctx.set_tensor_dist_attr_for_program(
                intermediate_var_0, out_grad_dist_attr
            )

            group_ranks = _get_comm_group(
                process_mesh_group, process_mesh_shape, parallel_axis, rank_id
            )
=======
            ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                                 out_grad_dist_attr)

            group_ranks = _get_comm_group(process_mesh_group,
                                          process_mesh_shape, parallel_axis,
                                          rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                },
            )
            check_variable_and_dtype(
                intermediate_var_0,
                'x',
                ['float16', 'float32', 'float64', 'uint16'],
                'linear',
            )
            check_dtype(
                intermediate_var_0.dtype,
                'dtype',
                ['float16', 'float32', 'float64', 'uint16'],
                'linear',
            )
            set_comm_op_dist_attr_for_program(
                c_identity_op, dist_attr.process_mesh, out_grad_dist_attr, ctx
            )
=======
                })
            check_variable_and_dtype(intermediate_var_0, 'x',
                                     ['float16', 'float32', 'float64'],
                                     'linear')
            check_dtype(intermediate_var_0.dtype, 'dtype',
                        ['float16', 'float32', 'float64'], 'linear')
            set_comm_op_dist_attr_for_program(c_identity_op,
                                              dist_attr.process_mesh,
                                              out_grad_dist_attr, ctx)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            new_kwargs = copy.deepcopy(kwargs)
            new_kwargs['Out@GRAD'] = [intermediate_var_0.name]
            matmul_op_desc = copy_op_with_new_input_output(
<<<<<<< HEAD
                ctx, main_block, backward_op, **new_kwargs
            )
=======
                ctx, main_block, backward_op, **new_kwargs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        else:
            # col parallel: matmul + allreduce
            assert Y_var_dim_mapping[0] < 0
            parallel_axis = Y_var_dim_mapping[1]
            new_kwargs = copy.deepcopy(kwargs)

            # NOTE (JZ-LIANG) should allow left operand be empty for matmul grad
            has_x_grad = len(kwargs['X@GRAD']) > 0
            if has_x_grad:
                assert len(kwargs['X@GRAD']) == 1
<<<<<<< HEAD
                X_grad = main_block._var_recursive(kwargs['X@GRAD'][0])
                intermediate_var_0 = main_block.create_var(
                    name=unique_name.generate_with_ignorable_key(
                        ".".join(["c_identity", 'tmp'])
                    )
                    + "@GRAD",
=======
                X_grad = main_block.var(kwargs['X@GRAD'][0])
                intermediate_var_0 = main_block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        ["c_identity", 'tmp'])) + "@GRAD",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    dtype=X_grad.dtype,
                    shape=X_grad.shape,
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    persistable=False,
<<<<<<< HEAD
                    stop_gradient=X_grad.stop_gradient,
                )

                X_grad_dist_attr = dist_attr.get_output_dist_attr(X_grad.name)
                assert X_grad_dist_attr is not None
                ctx.set_tensor_dist_attr_for_program(
                    intermediate_var_0, X_grad_dist_attr
                )
                new_kwargs['X@GRAD'] = [intermediate_var_0.name]

            matmul_op_desc = copy_op_with_new_input_output(
                ctx, main_block, backward_op, **new_kwargs
            )

            # NOTE (JZ-LIANG) trick to skip one allreduce if left operand has not grad
            if has_x_grad:
                group_ranks = _get_comm_group(
                    process_mesh_group,
                    process_mesh_shape,
                    parallel_axis,
                    rank_id,
                )
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                group = new_process_group(group_ranks)
                c_allreduce_sum_op = main_block.append_op(
                    type='c_allreduce_sum',
                    inputs={'X': [intermediate_var_0.name]},
                    outputs={'Out': kwargs['X@GRAD']},
                    attrs={
                        'ring_id': group.id,
                        'use_calc_stream': True,
                        'use_model_parallel': True,
<<<<<<< HEAD
                        OP_ROLE_KEY: OpRole.Backward,
                    },
                )
                set_comm_op_dist_attr_for_program(
                    c_allreduce_sum_op,
                    dist_attr.process_mesh,
                    X_grad_dist_attr,
                    ctx,
                )
    else:
        # replicate
        matmul_op_desc = copy_op_with_new_input_output(
            ctx, main_block, backward_op, **kwargs
        )
=======
                        OP_ROLE_KEY: OpRole.Backward
                    })
                set_comm_op_dist_attr_for_program(c_allreduce_sum_op,
                                                  dist_attr.process_mesh,
                                                  X_grad_dist_attr, ctx)
    else:
        # replicate
        matmul_op_desc = copy_op_with_new_input_output(ctx, main_block,
                                                       backward_op, **kwargs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # data parallel gradient synchronization
    act_grad_names = [X_var.name]

    out_grad_names = []
    if is_parameter_related(Y_var.name, main_block):
        out_grad_names = [kwargs['Y@GRAD'][0]]

    if trans_x:
        trans_x_y_dims_mapping(True, False, X_var_dims_mapping, None)

<<<<<<< HEAD
    gradient_synchronization(
        ctx, backward_op, act_grad_names, out_grad_names, rank_id
    )
=======
    gradient_synchronization(ctx, backward_op, act_grad_names, out_grad_names,
                             rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    if trans_x:
        trans_x_y_dims_mapping(True, False, X_var_dims_mapping, None)
    if trans_y:
        trans_x_y_dims_mapping(False, True, None, Y_var_dim_mapping)


def _init_param_sync(Weight_var, dist_op_context, startup_block, ctx, rank_id):

    if Weight_var.name in dist_op_context.already_init_sync_vars:
        return
    assert startup_block.has_var(Weight_var.name)
    dist_op_context.already_init_sync_vars.add(Weight_var.name)
    param = startup_block.var(Weight_var.name)
    param_dist_attr = ctx.get_tensor_dist_attr_for_program(param)
    process_mesh = param_dist_attr.process_mesh
    dim_mapping = param_dist_attr.dims_mapping

<<<<<<< HEAD
    for axis, size in enumerate(process_mesh.shape):
        if size <= 1 or axis in dim_mapping:
            pass
        else:
            group_ranks = _get_comm_group(
                process_mesh.process_ids, process_mesh.shape, axis, rank_id
            )
            sync_group = new_process_group(group_ranks)

            startup_block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': sync_group.id,
                    'root': 0,
                    'use_calc_stream': True,
                    OP_ROLE_KEY: OpRole.Forward,
                },
            )


class DistributedMatmul(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)
=======
    for axis, size in enumerate(process_mesh.topology):
        if size <= 1 or axis in dim_mapping:
            pass
        else:
            group_ranks = _get_comm_group(process_mesh.processes,
                                          process_mesh.topology, axis, rank_id)
            sync_group = new_process_group(group_ranks)

            startup_block.append_op(type='c_broadcast',
                                    inputs={'X': param},
                                    outputs={'Out': param},
                                    attrs={
                                        'ring_id': sync_group.id,
                                        'root': 0,
                                        'use_calc_stream': True,
                                        OP_ROLE_KEY: OpRole.Forward
                                    })


class DistributedMatmul(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        super(DistributedMatmul, self).__init__(op_type)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


register_distributed_operator_impl_container(DistributedMatmul("matmul"))


# ColumnParallel
class DistributedMatmulImpl0(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMatmulImpl0, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0]
        )
=======
        vars = main_block.vars
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # col parallel: matmul + allreduce
        assert Y_var_dim_mapping[0] < 0
        parallel_axis = Y_var_dim_mapping[1]

        has_x_grad = len(backward_op.output("X@GRAD")) > 0
        if has_x_grad:
            assert len(backward_op.output("X@GRAD")) == 1

        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulGradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulGradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # calc comm op cost
        if has_x_grad:
            attrs = {"use_calc_stream": True, "use_model_parallel": True}
            var_names = backward_op.output("X@GRAD")
            c_allreduce_sum_desc_mapping = build_comm_desc_from_dist_op(
                "c_allreduce_sum",
                dist_op,
                ctx,
                var_names,
                attrs=attrs,
<<<<<<< HEAD
                parallel_axis=parallel_axis,
            )
            comm_op_cost_list = build_comm_costs_from_descs(
                AllreduceSumOpCost,
                ctx,
                processes,
                c_allreduce_sum_desc_mapping,
                cluster,
            )
=======
                parallel_axis=parallel_axis)
            comm_op_cost_list = build_comm_costs_from_descs(
                AllreduceSumOpCost, ctx, processes,
                c_allreduce_sum_desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            res.append(comm_op_cost_list)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulOpCost, ctx, processes, desc_mapping, cluster
        )

        # calc comm op cost
        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0]
        )[-1]
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulOpCost, ctx, processes,
                                                   desc_mapping, cluster)

        # calc comm op cost
        serial_op = dist_op.serial_op
        vars = serial_op.block.vars
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0])[-1]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        var_names = serial_op.input("X")
        c_identity_desc_mapping = build_comm_desc_from_dist_op(
            "c_identity",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )

        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster
        )
=======
            parallel_axis=parallel_axis)

        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res_cost = [comm_op_cost_list, cost_mapping]

        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = copy.deepcopy(
<<<<<<< HEAD
            op_dist_attr.get_input_dims_mapping(x_name)
        )
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name)
        )
=======
            op_dist_attr.get_input_dims_mapping(x_name))
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = op_desc.attr('transpose_X')
        trans_y = op_desc.attr('transpose_Y')
        trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[-2]) or is_dim_replicate(
<<<<<<< HEAD
            y_dims_mapping[-1]
        ):
=======
                y_dims_mapping[-1]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
<<<<<<< HEAD
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )
=======
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                input_name
            )
=======
                input_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block._var_recursive(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
=======
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block.var(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = src_op.attr("transpose_X")
        trans_y = src_op.attr("transpose_Y")

        # TODO infer logic comm presentation
        matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            Weight_var.name
        )[-1]
        if trans_y:
            matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name
            )[-2]
        assert (
            matmul_col_dim_mapping >= 0
        ), "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
=======
            Weight_var.name)[-1]
        if trans_y:
            matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name)[-2]
        assert matmul_col_dim_mapping >= 0, "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        group = new_process_group(group_ranks)

        # infer new var shape with op dist attr
        x_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(X_var)
        assert x_tensor_dist_attr is not None
        identity_var_dist_attr = op_dist_attr.get_input_dist_attr(X_var.name)
        assert identity_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape_x = infer_shape(
            main_block, X_var, x_tensor_dist_attr, identity_var_dist_attr
        )
=======
        ref_shape_x = infer_shape(main_block, X_var, x_tensor_dist_attr,
                                  identity_var_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape_out = infer_shape(
            main_block, Out_var, out_tensor_dist_attr, out_var_dist_attr
        )

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(["c_identity", 'tmp'])
            ),
=======
        ref_shape_out = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                    out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_identity", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dtype=X_var.dtype,
            shape=X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
<<<<<<< HEAD
            stop_gradient=X_var.stop_gradient,
        )
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(
            intermediate_var_0, identity_var_dist_attr
        )

        check_variable_and_dtype(
            X_var,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            '_c_identity',
        )
=======
            stop_gradient=X_var.stop_gradient)
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             identity_var_dist_attr)

        check_variable_and_dtype(
            X_var, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], '_c_identity')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        c_identity_op = main_block.append_op(
            type='c_identity',
            inputs={'X': [X_var]},
            outputs={'Out': intermediate_var_0},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
<<<<<<< HEAD
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(
            intermediate_var_0,
            'x',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
        check_dtype(
            intermediate_var_0.dtype,
            'dtype',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
=======
                OP_ROLE_KEY: src_op.attr('op_role')
            })
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(intermediate_var_0, 'x',
                                 ['float16', 'float32', 'float64'], 'linear')
        check_dtype(intermediate_var_0.dtype, 'dtype',
                    ['float16', 'float32', 'float64'], 'linear')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {
            'transpose_X': trans_x,
            'transpose_Y': trans_y,
            'alpha': 1,
<<<<<<< HEAD
            OP_ROLE_KEY: src_op.attr('op_role'),
        }
        inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
        matmul_op = main_block.append_op(
            type='matmul', inputs=inputs, outputs={'Out': Out_var}, attrs=attrs
        )
=======
            OP_ROLE_KEY: src_op.attr('op_role')
        }
        inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
        matmul_op = main_block.append_op(type='matmul',
                                         inputs=inputs,
                                         outputs={'Out': Out_var},
                                         attrs=attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if Out_var.shape != ref_shape_out:
            Out_var.desc.set_shape(ref_shape_out)

        # set dist op's dist_attr with serial op's dist_attr
        # c_identity
<<<<<<< HEAD
        identity_op_dist_attr = OperatorDistAttr()
=======
        identity_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        identity_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        identity_op_dist_attr.impl_type = op_dist_attr.impl_type
        identity_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        # input
        input_varname = c_identity_op.desc.input_arg_names()[0]
        input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
        assert input_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
            op_dist_attr
        )
        identity_op_dist_attr.set_input_dist_attr(
            input_varname, input_dist_attr
        )
        # output
        output_varname = c_identity_op.desc.output_arg_names()[0]
        identity_op_dist_attr.set_output_dist_attr(
            output_varname, input_dist_attr
        )
=======
            op_dist_attr)
        identity_op_dist_attr.set_input_dist_attr(input_varname,
                                                  input_dist_attr)
        # output
        output_varname = c_identity_op.desc.output_arg_names()[0]
        identity_op_dist_attr.set_output_dist_attr(output_varname,
                                                   input_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # set op dist attr
        ctx.set_op_dist_attr_for_program(c_identity_op, identity_op_dist_attr)

        # matmul
<<<<<<< HEAD
        matmul_op_dist_attr = OperatorDistAttr()
=======
        matmul_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        matmul_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmul_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmul_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        # input
        for input_varname in matmul_op.desc.input_arg_names():
            if input_varname in src_op.desc.input_arg_names():
                input_dist_attr = op_dist_attr.get_input_dist_attr(
<<<<<<< HEAD
                    input_varname
                )
                assert input_dist_attr is not None, "dist_attr is {}".format(
                    op_dist_attr
                )
                matmul_op_dist_attr.set_input_dist_attr(
                    input_varname, input_dist_attr
                )
            else:
                input_var = main_block._var_recursive(input_varname)
                tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    input_var
                )
                matmul_op_dist_attr.set_input_dist_attr(
                    input_varname, tensor_dist_attr
                )
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # output
        output_varname = matmul_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
        assert output_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
            op_dist_attr
        )
        matmul_op_dist_attr.set_output_dist_attr(
            output_varname, output_dist_attr
        )
=======
            op_dist_attr)
        matmul_op_dist_attr.set_output_dist_attr(output_varname,
                                                 output_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # set op dist attr
        ctx.set_op_dist_attr_for_program(matmul_op, matmul_op_dist_attr)

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
<<<<<<< HEAD
            _init_param_sync(
                Weight_var, dist_op_context, startup_block, ctx, rank_id
            )
=======
            _init_param_sync(Weight_var, dist_op_context, startup_block, ctx,
                             rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# RowParallel
class DistributedMatmulImpl1(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMatmulImpl1, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0]
        )
=======
        vars = main_block.vars
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        assert Y_var_dim_mapping[1] < 0
        parallel_axis = Y_var_dim_mapping[0]

        # calc comm op cost
        var_names = [backward_op.input("Out@GRAD")[0]]
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        c_identity_desc_mapping = build_comm_desc_from_dist_op(
            "c_identity",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster
        )
        res.append(comm_op_cost_list)

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        cost_mapping = build_comp_costs_from_descs(
            MatmulGradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
            parallel_axis=parallel_axis)
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster)
        res.append(comm_op_cost_list)

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        cost_mapping = build_comp_costs_from_descs(MatmulGradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulOpCost, ctx, processes, desc_mapping, cluster
        )

        # calc comm op cost
        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0]
        )[-2]
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulOpCost, ctx, processes,
                                                   desc_mapping, cluster)

        # calc comm op cost
        serial_op = dist_op.serial_op
        vars = serial_op.block.vars

        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0])[-2]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {"use_calc_stream": True, "use_model_parallel": True}

        var_names = serial_op.output("Out")
        c_allreduce_sum_desc_mapping = build_comm_desc_from_dist_op(
            "c_allreduce_sum",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )

        comm_op_cost_list = build_comm_costs_from_descs(
            AllreduceSumOpCost,
            ctx,
            processes,
            c_allreduce_sum_desc_mapping,
            cluster,
        )
=======
            parallel_axis=parallel_axis)

        comm_op_cost_list = build_comm_costs_from_descs(
            AllreduceSumOpCost, ctx, processes, c_allreduce_sum_desc_mapping,
            cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        res_cost = [cost_mapping, comm_op_cost_list]

        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = copy.deepcopy(
<<<<<<< HEAD
            op_dist_attr.get_input_dims_mapping(x_name)
        )
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name)
        )
=======
            op_dist_attr.get_input_dims_mapping(x_name))
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = op_desc.attr('transpose_X')
        trans_y = op_desc.attr('transpose_Y')
        trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[-2]) or is_dim_shard(
<<<<<<< HEAD
            y_dims_mapping[-1]
        ):
=======
                y_dims_mapping[-1]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
<<<<<<< HEAD
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )
=======
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                input_name
            )
=======
                input_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block._var_recursive(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
=======
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block.var(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = src_op.attr('transpose_X')
        trans_y = src_op.attr('transpose_Y')

        # TODO infer logic comm presentation
        matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            Weight_var.name
        )[-2]
        if trans_y:
            matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name
            )[-1]
        assert (
            matmul_row_dim_mapping >= 0
        ), "row_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_row_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = matmul_row_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        check_variable_and_dtype(
            X_var, 'x', ['float16', 'float32', 'float64', 'uint16'], 'linear'
        )
        check_dtype(
            X_var.dtype,
            'dtype',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
=======
            Weight_var.name)[-2]
        if trans_y:
            matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name)[-1]
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {
            'transpose_X': trans_x,
            'transpose_Y': trans_y,
            'alpha': 1,
<<<<<<< HEAD
            OP_ROLE_KEY: src_op.attr('op_role'),
=======
            OP_ROLE_KEY: src_op.attr('op_role')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        inputs = {'X': X_var, 'Y': Weight_var}

        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape = infer_shape(
            main_block, Out_var, out_tensor_dist_attr, out_var_dist_attr
        )

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(["c_allreduce_sum", 'tmp'])
            ),
=======
        ref_shape = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_allreduce_sum", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            shape=Out_var.shape,
            dtype=Out_var.dtype,
            type=Out_var.type,
            lod_level=Out_var.lod_level,
            persistable=False,
            is_data=False,
<<<<<<< HEAD
            need_check_feed=Out_var.desc.need_check_feed(),
        )
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(
            intermediate_var_0, out_var_dist_attr
        )

        matmul_op = main_block.append_op(
            type='matmul',
            inputs=inputs,
            outputs={'Out': intermediate_var_0},
            attrs=attrs,
        )
=======
            need_check_feed=Out_var.desc.need_check_feed())
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             out_var_dist_attr)

        matmul_op = main_block.append_op(type='matmul',
                                         inputs=inputs,
                                         outputs={'Out': intermediate_var_0},
                                         attrs=attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if intermediate_var_0.shape != ref_shape:
            intermediate_var_0.desc.set_shape(ref_shape)

        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': intermediate_var_0},
            outputs={'Out': Out_var},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
<<<<<<< HEAD
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )
=======
                OP_ROLE_KEY: src_op.attr('op_role')
            })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if Out_var.shape != ref_shape:
            Out_var.desc.set_shape(ref_shape)

        # set dist op's dist_attr with serial op's dist_attr
        # matmul
<<<<<<< HEAD
        matmul_op_dist_attr = OperatorDistAttr()
=======
        matmul_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        matmul_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmul_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmul_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in matmul_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
                op_dist_attr
            )
            matmul_op_dist_attr.set_input_dist_attr(
                input_varname, input_dist_attr
            )
        output_varname = matmul_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr
        )
        matmul_op_dist_attr.set_output_dist_attr(
            output_varname, output_dist_attr
        )
        ctx.set_op_dist_attr_for_program(matmul_op, matmul_op_dist_attr)

        # allreduce
        allreduce_op_dist_attr = OperatorDistAttr()
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        allreduce_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        allreduce_op_dist_attr.impl_type = op_dist_attr.impl_type
        allreduce_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in c_allreduce_sum_op.desc.input_arg_names():
<<<<<<< HEAD
            input_var = main_block._var_recursive(input_varname)
            tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(input_var)
            assert tensor_dist_attr is not None
            allreduce_op_dist_attr.set_input_dist_attr(
                input_varname, tensor_dist_attr
            )
        for output_varname in c_allreduce_sum_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr
            )
            allreduce_op_dist_attr.set_output_dist_attr(
                output_varname, output_dist_attr
            )
        ctx.set_op_dist_attr_for_program(
            c_allreduce_sum_op, allreduce_op_dist_attr
        )

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            _init_param_sync(
                Weight_var, dist_op_context, startup_block, ctx, rank_id
            )
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# ReplicateParallel
class DistributedMatmulImpl2(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMatmulImpl2, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulGradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        vars = main_block.vars

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulGradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulOpCost, ctx, processes,
                                                   desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        res_cost = [cost_mapping]
        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            x_dims_mapping[-2]
        ):
=======
                x_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            y_dims_mapping[-2]
        ):
=======
                y_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if is_dim_shard(out_dims_mapping[-1]):
            return False
        if is_valid_list_index(out_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            out_dims_mapping[-2]
        ):
=======
                out_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        return True

    def is_auto_compatible(self, dist_op):
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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


<<<<<<< HEAD
register_distributed_operator_impl(
    "matmul", DistributedMatmulImpl0("column_parallel")
)
register_distributed_operator_impl(
    "matmul", DistributedMatmulImpl1("row_parallel")
)
register_distributed_operator_impl(
    "matmul", DistributedMatmulImpl2("replicate_parallel")
)


class DistributedMatmulV2(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)
=======
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl0("column_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl1("row_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl2("replicate_parallel"))


class DistributedMatmulV2(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        super(DistributedMatmulV2, self).__init__(op_type)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


register_distributed_operator_impl_container(DistributedMatmulV2("matmul_v2"))


# ColumnParallel
class DistributedMatmulV2Impl0(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMatmulV2Impl0, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0]
        )
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
=======
        vars = main_block.vars
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0])
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # col parallel: matmul + allreduce
        if backward_op.attr("trans_y"):
            Y_var_dim_mapping.reverse()
        assert Y_var_dim_mapping[0] < 0
        parallel_axis = Y_var_dim_mapping[1]

        has_x_grad = len(backward_op.output("X@GRAD")) > 0
        if has_x_grad:
            assert len(backward_op.output("X@GRAD")) == 1

        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )

        cost_mapping = build_comp_costs_from_descs(
            MatmulV2GradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)

        cost_mapping = build_comp_costs_from_descs(MatmulV2GradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # calc comm op cost
        if has_x_grad:
            attrs = {"use_calc_stream": True, "use_model_parallel": True}
            var_names = backward_op.output("X@GRAD")
            c_allreduce_sum_desc_mapping = build_comm_desc_from_dist_op(
                "c_allreduce_sum",
                dist_op,
                ctx,
                var_names,
                attrs=attrs,
<<<<<<< HEAD
                parallel_axis=parallel_axis,
            )
            comm_op_cost_list = build_comm_costs_from_descs(
                AllreduceSumOpCost,
                ctx,
                processes,
                c_allreduce_sum_desc_mapping,
                cluster,
            )
=======
                parallel_axis=parallel_axis)
            comm_op_cost_list = build_comm_costs_from_descs(
                AllreduceSumOpCost, ctx, processes,
                c_allreduce_sum_desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            res.append(comm_op_cost_list)

        # need gradient allreduce
        process_mesh = dist_attr.process_mesh
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
        # TODO: trans shape if trans_x or trans_y is True
<<<<<<< HEAD
        comp_desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        comp_cost_mapping = build_comp_costs_from_descs(
            MatmulV2OpCost, ctx, processes, comp_desc_mapping, cluster
        )

        # calc comm op cost
        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0]
        )[-1]
=======
        comp_desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                         dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        comp_cost_mapping = build_comp_costs_from_descs(MatmulV2OpCost, ctx,
                                                        processes,
                                                        comp_desc_mapping,
                                                        cluster)

        # calc comm op cost
        serial_op = dist_op.serial_op
        vars = serial_op.block.vars

        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0])[-1]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {"use_calc_stream": True, "use_model_parallel": True}

        var_names = serial_op.input("X")
        c_identity_desc_mapping = build_comm_desc_from_dist_op(
            "c_identity",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster
        )
=======
            parallel_axis=parallel_axis)
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        res_cost = [comm_op_cost_list, comp_cost_mapping]
        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = copy.deepcopy(
<<<<<<< HEAD
            op_dist_attr.get_input_dims_mapping(x_name)
        )
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name)
        )
=======
            op_dist_attr.get_input_dims_mapping(x_name))
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = op_desc.attr('trans_x')
        trans_y = op_desc.attr('trans_y')
        trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[-2]) or is_dim_replicate(
<<<<<<< HEAD
            y_dims_mapping[-1]
        ):
=======
                y_dims_mapping[-1]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
<<<<<<< HEAD
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )
=======
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                input_name
            )
=======
                input_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block._var_recursive(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
=======
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = src_op.attr('trans_x')
        trans_y = src_op.attr('trans_y')

        # TODO infer logic comm presentation
        matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            Weight_var.name
        )[-1]
        if trans_y:
            matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name
            )[-2]
        assert (
            matmul_col_dim_mapping >= 0
        ), "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
=======
            Weight_var.name)[-1]
        if trans_y:
            matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name)[-2]
        assert matmul_col_dim_mapping >= 0, "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        group = new_process_group(group_ranks)

        # infer new var shape with op dist attr
        x_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(X_var)
        assert x_tensor_dist_attr is not None
        identity_var_dist_attr = op_dist_attr.get_input_dist_attr(X_var.name)
        assert identity_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape_x = infer_shape(
            main_block, X_var, x_tensor_dist_attr, identity_var_dist_attr
        )
=======
        ref_shape_x = infer_shape(main_block, X_var, x_tensor_dist_attr,
                                  identity_var_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape_out = infer_shape(
            main_block, Out_var, out_tensor_dist_attr, out_var_dist_attr
        )

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(["c_identity", 'tmp'])
            ),
=======
        ref_shape_out = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                    out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_identity", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dtype=X_var.dtype,
            shape=X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
<<<<<<< HEAD
            stop_gradient=X_var.stop_gradient,
        )
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(
            intermediate_var_0, identity_var_dist_attr
        )

        check_variable_and_dtype(
            X_var,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            '_c_identity',
        )
=======
            stop_gradient=X_var.stop_gradient)
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             identity_var_dist_attr)

        check_variable_and_dtype(
            X_var, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], '_c_identity')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        c_identity_op = main_block.append_op(
            type='c_identity',
            inputs={'X': [X_var]},
            outputs={'Out': intermediate_var_0},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
                OP_ROLE_KEY: src_op.attr('op_role'),
<<<<<<< HEAD
            },
        )
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(
            intermediate_var_0,
            'x',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
        check_dtype(
            intermediate_var_0.dtype,
            'dtype',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
        attrs = {
            'trans_x': trans_x,
            'trans_y': trans_y,
            OP_ROLE_KEY: src_op.attr('op_role'),
        }
        inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
        matmul_v2_op = main_block.append_op(
            type='matmul_v2',
            inputs=inputs,
            outputs={'Out': Out_var},
            attrs=attrs,
        )
=======
            })
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(intermediate_var_0, 'x',
                                 ['float16', 'float32', 'float64'], 'linear')
        check_dtype(intermediate_var_0.dtype, 'dtype',
                    ['float16', 'float32', 'float64'], 'linear')
        attrs = {
            'trans_x': trans_x,
            'trans_y': trans_y,
            OP_ROLE_KEY: src_op.attr('op_role')
        }
        inputs = {'X': [intermediate_var_0], 'Y': [Weight_var]}
        matmul_v2_op = main_block.append_op(type='matmul_v2',
                                            inputs=inputs,
                                            outputs={'Out': Out_var},
                                            attrs=attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if Out_var.shape != ref_shape_out:
            Out_var.desc.set_shape(ref_shape_out)

        # set dist op's dist_attr with serial op's dist_attr
        # c_identity
<<<<<<< HEAD
        identity_op_dist_attr = OperatorDistAttr()
=======
        identity_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        identity_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        identity_op_dist_attr.impl_type = op_dist_attr.impl_type
        identity_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        # input
        input_varname = c_identity_op.desc.input_arg_names()[0]
        input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
        assert input_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
            op_dist_attr
        )
        identity_op_dist_attr.set_input_dist_attr(
            input_varname, input_dist_attr
        )
        # output
        output_varname = c_identity_op.desc.output_arg_names()[0]
        identity_op_dist_attr.set_output_dist_attr(
            output_varname, input_dist_attr
        )
        ctx.set_op_dist_attr_for_program(c_identity_op, identity_op_dist_attr)

        # matmulv2
        matmulv2_op_dist_attr = OperatorDistAttr()
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        matmulv2_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmulv2_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmulv2_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in matmul_v2_op.desc.input_arg_names():
            if input_varname in src_op.desc.input_arg_names():
                input_dist_attr = op_dist_attr.get_input_dist_attr(
<<<<<<< HEAD
                    input_varname
                )
                assert input_dist_attr is not None, "dist_attr is {}".format(
                    op_dist_attr
                )
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, input_dist_attr
                )
            else:
                input_var = main_block._var_recursive(input_varname)
                tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    input_var
                )
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, tensor_dist_attr
                )
        for output_varname in matmul_v2_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr
            )
            matmulv2_op_dist_attr.set_output_dist_attr(
                output_varname, output_dist_attr
            )
=======
                    input_varname)
                assert input_dist_attr is not None, "dist_attr is {}".format(
                    op_dist_attr)
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, input_dist_attr)
            else:
                input_var = main_block.var(input_varname)
                tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    input_var)
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, tensor_dist_attr)
        for output_varname in matmul_v2_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr)
            matmulv2_op_dist_attr.set_output_dist_attr(output_varname,
                                                       output_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ctx.set_op_dist_attr_for_program(matmul_v2_op, matmulv2_op_dist_attr)

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
<<<<<<< HEAD
            _init_param_sync(
                Weight_var, dist_op_context, startup_block, ctx, rank_id
            )
=======
            _init_param_sync(Weight_var, dist_op_context, startup_block, ctx,
                             rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# RowParallel
class DistributedMatmulV2Impl1(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMatmulV2Impl1, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD

        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0]
        )
=======
        vars = main_block.vars
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        assert Y_var_dim_mapping[1] < 0
        parallel_axis = Y_var_dim_mapping[0]

        process_mesh = dist_attr.process_mesh
<<<<<<< HEAD
        processes = process_mesh.process_ids
=======
        processes = process_mesh.processes
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # calc comm op cost
        var_names = [backward_op.input("Out@GRAD")[0]]
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        c_identity_desc_mapping = build_comm_desc_from_dist_op(
            "c_identity",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster
        )
        res.append(comm_op_cost_list)

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        cost_mapping = build_comp_costs_from_descs(
            MatmulV2GradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
            parallel_axis=parallel_axis)
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster)
        res.append(comm_op_cost_list)

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        cost_mapping = build_comp_costs_from_descs(MatmulV2GradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # need gradient allreduce
        process_mesh = dist_attr.process_mesh
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulV2OpCost, ctx, processes, desc_mapping, cluster
        )

        # calc comm op cost
        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0]
        )[-2]
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulV2OpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)

        # calc comm op cost
        serial_op = dist_op.serial_op
        vars = serial_op.block.vars

        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0])[-2]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {"use_calc_stream": True, "use_model_parallel": True}

        var_names = serial_op.output("Out")
        c_allreduce_sum_desc_mapping = build_comm_desc_from_dist_op(
            "c_allreduce_sum",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )

        comm_op_cost_list = build_comm_costs_from_descs(
            AllreduceSumOpCost,
            ctx,
            processes,
            c_allreduce_sum_desc_mapping,
            cluster,
        )
=======
            parallel_axis=parallel_axis)

        comm_op_cost_list = build_comm_costs_from_descs(
            AllreduceSumOpCost, ctx, processes, c_allreduce_sum_desc_mapping,
            cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res_cost = [cost_mapping, comm_op_cost_list]

        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = copy.deepcopy(
<<<<<<< HEAD
            op_dist_attr.get_input_dims_mapping(x_name)
        )
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name)
        )
=======
            op_dist_attr.get_input_dims_mapping(x_name))
        y_dims_mapping = copy.deepcopy(
            op_dist_attr.get_input_dims_mapping(y_name))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = op_desc.attr('trans_x')
        trans_y = op_desc.attr('trans_y')
        trans_x_y_dims_mapping(trans_x, trans_y, x_dims_mapping, y_dims_mapping)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[-2]) or is_dim_shard(
<<<<<<< HEAD
            y_dims_mapping[-1]
        ):
=======
                y_dims_mapping[-1]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
<<<<<<< HEAD
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )
=======
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                input_name
            )
=======
                input_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block._var_recursive(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
=======
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        trans_x = src_op.attr('trans_x')
        trans_y = src_op.attr('trans_y')

        # TODO infer logic comm presentation
        matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            Weight_var.name
        )[-2]
        if trans_y:
            matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name
            )[-1]
        assert (
            matmul_row_dim_mapping >= 0
        ), "row_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_row_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = matmul_row_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        check_variable_and_dtype(
            X_var, 'x', ['float16', 'float32', 'float64', 'uint16'], 'linear'
        )
        check_dtype(
            X_var.dtype,
            'dtype',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
        attrs = {
            'trans_x': trans_x,
            'trans_y': trans_y,
            OP_ROLE_KEY: src_op.attr('op_role'),
=======
            Weight_var.name)[-2]
        if trans_y:
            matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
                Weight_var.name)[-1]
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
            'trans_x': trans_x,
            'trans_y': trans_y,
            OP_ROLE_KEY: src_op.attr('op_role')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        inputs = {'X': X_var, 'Y': Weight_var}

        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape = infer_shape(
            main_block, Out_var, out_tensor_dist_attr, out_var_dist_attr
        )

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(["c_allreduce_sum", 'tmp'])
            ),
=======
        ref_shape = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_allreduce_sum", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            shape=Out_var.shape,
            dtype=Out_var.dtype,
            type=Out_var.type,
            lod_level=Out_var.lod_level,
            persistable=False,
            is_data=False,
<<<<<<< HEAD
            need_check_feed=Out_var.desc.need_check_feed(),
        )
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(
            intermediate_var_0, out_var_dist_attr
        )

        matmul_v2_op = main_block.append_op(
            type='matmul_v2',
            inputs=inputs,
            outputs={'Out': intermediate_var_0},
            attrs=attrs,
        )
=======
            need_check_feed=Out_var.desc.need_check_feed())
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             out_var_dist_attr)

        matmul_v2_op = main_block.append_op(type='matmul_v2',
                                            inputs=inputs,
                                            outputs={'Out': intermediate_var_0},
                                            attrs=attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if intermediate_var_0.shape != ref_shape:
            intermediate_var_0.desc.set_shape(ref_shape)

        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': intermediate_var_0},
            outputs={'Out': Out_var},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
<<<<<<< HEAD
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )
=======
                OP_ROLE_KEY: src_op.attr('op_role')
            })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if Out_var.shape != ref_shape:
            Out_var.desc.set_shape(ref_shape)

        # set dist op's dist_attr with serial op's dist_attr
        # matmulv2
<<<<<<< HEAD
        matmulv2_op_dist_attr = OperatorDistAttr()
=======
        matmulv2_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        matmulv2_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmulv2_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmulv2_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in matmul_v2_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
                op_dist_attr
            )
            matmulv2_op_dist_attr.set_input_dist_attr(
                input_varname, input_dist_attr
            )
        output_varname = matmul_v2_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr
        )
        matmulv2_op_dist_attr.set_output_dist_attr(
            output_varname, output_dist_attr
        )
        ctx.set_op_dist_attr_for_program(matmul_v2_op, matmulv2_op_dist_attr)

        # allreduce
        allreduce_op_dist_attr = OperatorDistAttr()
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        allreduce_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        allreduce_op_dist_attr.impl_type = op_dist_attr.impl_type
        allreduce_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in c_allreduce_sum_op.desc.input_arg_names():
<<<<<<< HEAD
            input_var = main_block._var_recursive(input_varname)
            tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(input_var)
            assert tensor_dist_attr is not None
            allreduce_op_dist_attr.set_input_dist_attr(
                input_varname, tensor_dist_attr
            )
        for output_varname in c_allreduce_sum_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr
            )
            allreduce_op_dist_attr.set_output_dist_attr(
                output_varname, output_dist_attr
            )
        ctx.set_op_dist_attr_for_program(
            c_allreduce_sum_op, allreduce_op_dist_attr
        )

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            _init_param_sync(
                Weight_var, dist_op_context, startup_block, ctx, rank_id
            )
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# ReplicateParallel
class DistributedMatmulV2Impl2(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMatmulV2Impl2, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD
        process_mesh = dist_attr.process_mesh

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulV2GradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        vars = main_block.vars
        process_mesh = dist_attr.process_mesh

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulV2GradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MatmulV2OpCost, ctx, processes, desc_mapping, cluster
        )
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MatmulV2OpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        res_cost = [cost_mapping]

        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            x_dims_mapping[-2]
        ):
=======
                x_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            y_dims_mapping[-2]
        ):
=======
                y_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        if is_valid_list_index(out_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            out_dims_mapping[-2]
        ):
=======
                out_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        return True

    def is_auto_compatible(self, dist_op):
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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


<<<<<<< HEAD
register_distributed_operator_impl(
    "matmul_v2", DistributedMatmulV2Impl0("column_parallel")
)
register_distributed_operator_impl(
    "matmul_v2", DistributedMatmulV2Impl1("row_parallel")
)
register_distributed_operator_impl(
    "matmul_v2", DistributedMatmulV2Impl2("replicate_parallel")
)


class DistributedMul(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)
=======
register_distributed_operator_impl("matmul_v2",
                                   DistributedMatmulV2Impl0("column_parallel"))
register_distributed_operator_impl("matmul_v2",
                                   DistributedMatmulV2Impl1("row_parallel"))
register_distributed_operator_impl(
    "matmul_v2", DistributedMatmulV2Impl2("replicate_parallel"))


class DistributedMul(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        super(DistributedMul, self).__init__(op_type)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


register_distributed_operator_impl_container(DistributedMul("mul"))


# ColumnParallel
class DistributedMulImpl0(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMulImpl0, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0]
        )
=======
        vars = main_block.vars
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # col parallel: matmul + allreduce
        assert Y_var_dim_mapping[0] < 0
        parallel_axis = Y_var_dim_mapping[1]

        has_x_grad = len(backward_op.output("X@GRAD")) > 0
        if has_x_grad:
            assert len(backward_op.output("X@GRAD")) == 1

        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MulGradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MulGradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # calc comm op cost
        if has_x_grad:
            attrs = {"use_calc_stream": True, "use_model_parallel": True}
            var_names = backward_op.output("X@GRAD")
            c_allreduce_sum_desc_mapping = build_comm_desc_from_dist_op(
                "c_allreduce_sum",
                dist_op,
                ctx,
                var_names,
                attrs=attrs,
<<<<<<< HEAD
                parallel_axis=parallel_axis,
            )
            comm_op_cost_list = build_comm_costs_from_descs(
                AllreduceSumOpCost,
                ctx,
                processes,
                c_allreduce_sum_desc_mapping,
                cluster,
            )
=======
                parallel_axis=parallel_axis)
            comm_op_cost_list = build_comm_costs_from_descs(
                AllreduceSumOpCost, ctx, processes,
                c_allreduce_sum_desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            res.append(comm_op_cost_list)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MulOpCost, ctx, processes, desc_mapping, cluster
        )

        # calc comm op cost
        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0]
        )[-1]
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MulOpCost, ctx, processes,
                                                   desc_mapping, cluster)

        # calc comm op cost
        serial_op = dist_op.serial_op
        vars = serial_op.block.vars
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0])[-1]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        var_names = serial_op.input("X")
        c_identity_desc_mapping = build_comm_desc_from_dist_op(
            "c_identity",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )

        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster
        )
=======
            parallel_axis=parallel_axis)

        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res_cost = [comm_op_cost_list, cost_mapping]

        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[-2]) or is_dim_replicate(
<<<<<<< HEAD
            y_dims_mapping[-1]
        ):
=======
                y_dims_mapping[-1]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
<<<<<<< HEAD
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )
=======
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                input_name
            )
=======
                input_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block._var_recursive(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name
        )[-1]
        assert (
            matmul_col_dim_mapping >= 0
        ), "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
=======
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name)[-1]
        assert matmul_col_dim_mapping >= 0, "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_col_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = matmul_col_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        group = new_process_group(group_ranks)

        # infer new var shape with op dist attr
        x_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(X_var)
        assert x_tensor_dist_attr is not None
        identity_var_dist_attr = op_dist_attr.get_input_dist_attr(X_var.name)
        assert identity_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape_x = infer_shape(
            main_block, X_var, x_tensor_dist_attr, identity_var_dist_attr
        )
=======
        ref_shape_x = infer_shape(main_block, X_var, x_tensor_dist_attr,
                                  identity_var_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape_out = infer_shape(
            main_block, Out_var, out_tensor_dist_attr, out_var_dist_attr
        )

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(["c_identity", 'tmp'])
            ),
=======
        ref_shape_out = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                    out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_identity", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dtype=X_var.dtype,
            shape=X_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
<<<<<<< HEAD
            stop_gradient=X_var.stop_gradient,
        )
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(
            intermediate_var_0, identity_var_dist_attr
        )

        check_variable_and_dtype(
            X_var,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            '_c_identity',
        )
=======
            stop_gradient=X_var.stop_gradient)
        # set intermediate_var_0's dist_attr with X_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             identity_var_dist_attr)

        check_variable_and_dtype(
            X_var, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], '_c_identity')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        c_identity_op = main_block.append_op(
            type='c_identity',
            inputs={'X': [X_var]},
            outputs={'Out': intermediate_var_0},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
<<<<<<< HEAD
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(
            intermediate_var_0,
            'x',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
        check_dtype(
            intermediate_var_0.dtype,
            'dtype',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
=======
                OP_ROLE_KEY: src_op.attr('op_role')
            })
        if intermediate_var_0.shape != ref_shape_x:
            intermediate_var_0.desc.set_shape(ref_shape_x)

        check_variable_and_dtype(intermediate_var_0, 'x',
                                 ['float16', 'float32', 'float64'], 'linear')
        check_dtype(intermediate_var_0.dtype, 'dtype',
                    ['float16', 'float32', 'float64'], 'linear')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # attrs = {'trans_x': False, 'trans_y': False}
        attrs = {
            "x_num_col_dims": src_op.desc.attr("x_num_col_dims"),
            "y_num_col_dims": src_op.desc.attr("y_num_col_dims"),
<<<<<<< HEAD
            OP_ROLE_KEY: src_op.attr('op_role'),
=======
            OP_ROLE_KEY: src_op.attr('op_role')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        inputs = {'X': intermediate_var_0, 'Y': Weight_var}

        inputs_ref_shape = {}
        inputs_original_shape = {}
        for var_name in inputs:
            if var_name == "X":
                var = X_var
            else:
                var = inputs[var_name]
            inputs_original_shape[var_name] = var.shape
            input_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(var)
            input_var_dist_attr = op_dist_attr.get_input_dist_attr(var.name)
<<<<<<< HEAD
            input_ref_shape = infer_shape(
                main_block, var, input_tensor_dist_attr, input_var_dist_attr
            )
            inputs_ref_shape[var_name] = input_ref_shape
            var.desc.set_shape(input_ref_shape)

        mul_op = main_block.append_op(
            type='mul', inputs=inputs, outputs={'Out': Out_var}, attrs=attrs
        )
=======
            input_ref_shape = infer_shape(main_block, var,
                                          input_tensor_dist_attr,
                                          input_var_dist_attr)
            inputs_ref_shape[var_name] = input_ref_shape
            var.desc.set_shape(input_ref_shape)

        mul_op = main_block.append_op(type='mul',
                                      inputs=inputs,
                                      outputs={'Out': Out_var},
                                      attrs=attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if Out_var.shape != ref_shape_out:
            Out_var.desc.set_shape(ref_shape_out)

        for var_name in inputs:
            var = inputs[var_name]
            original_shape = inputs_original_shape[var_name]
            var.desc.set_shape(original_shape)

        # set dist op's dist_attr with serial op's dist_attr
        # c_identity
<<<<<<< HEAD
        identity_op_dist_attr = OperatorDistAttr()
=======
        identity_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        identity_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        identity_op_dist_attr.impl_type = op_dist_attr.impl_type
        identity_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        # input
        input_varname = c_identity_op.desc.input_arg_names()[0]
        input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
        assert input_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
            op_dist_attr
        )
        identity_op_dist_attr.set_input_dist_attr(
            input_varname, input_dist_attr
        )
        # output
        output_varname = c_identity_op.desc.output_arg_names()[0]
        identity_op_dist_attr.set_output_dist_attr(
            output_varname, input_dist_attr
        )
        ctx.set_op_dist_attr_for_program(c_identity_op, identity_op_dist_attr)

        # matmulv2
        matmulv2_op_dist_attr = OperatorDistAttr()
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        matmulv2_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmulv2_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmulv2_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in mul_op.desc.input_arg_names():
            if input_varname in src_op.desc.input_arg_names():
                input_dist_attr = op_dist_attr.get_input_dist_attr(
<<<<<<< HEAD
                    input_varname
                )
                assert input_dist_attr is not None, "dist_attr is {}".format(
                    op_dist_attr
                )
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, input_dist_attr
                )
            else:
                input_var = main_block._var_recursive(input_varname)
                tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    input_var
                )
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, tensor_dist_attr
                )
        for output_varname in mul_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr
            )
            matmulv2_op_dist_attr.set_output_dist_attr(
                output_varname, output_dist_attr
            )
=======
                    input_varname)
                assert input_dist_attr is not None, "dist_attr is {}".format(
                    op_dist_attr)
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, input_dist_attr)
            else:
                input_var = main_block.var(input_varname)
                tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(
                    input_var)
                matmulv2_op_dist_attr.set_input_dist_attr(
                    input_varname, tensor_dist_attr)
        for output_varname in mul_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr)
            matmulv2_op_dist_attr.set_output_dist_attr(output_varname,
                                                       output_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ctx.set_op_dist_attr_for_program(mul_op, matmulv2_op_dist_attr)

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
<<<<<<< HEAD
            _init_param_sync(
                Weight_var, dist_op_context, startup_block, ctx, rank_id
            )
=======
            _init_param_sync(Weight_var, dist_op_context, startup_block, ctx,
                             rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# RowParallel
class DistributedMulImpl1(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMulImpl1, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        process_mesh = dist_attr.process_mesh
        main_block = backward_op.block
<<<<<<< HEAD
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0]
        )
=======
        vars = main_block.vars
        Y_var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Y")[0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        assert Y_var_dim_mapping[1] < 0
        parallel_axis = Y_var_dim_mapping[0]

        # calc comm op cost
        var_names = [backward_op.input("Out@GRAD")[0]]
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        c_identity_desc_mapping = build_comm_desc_from_dist_op(
            "c_identity",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )
        processes = process_mesh.process_ids
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster
        )
        res.append(comm_op_cost_list)

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        cost_mapping = build_comp_costs_from_descs(
            MulGradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
            parallel_axis=parallel_axis)
        processes = process_mesh.processes
        comm_op_cost_list = build_comm_costs_from_descs(
            IdentityOpCost, ctx, processes, c_identity_desc_mapping, cluster)
        res.append(comm_op_cost_list)

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        cost_mapping = build_comp_costs_from_descs(MulGradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MulOpCost, ctx, processes, desc_mapping, cluster
        )

        # calc comm op cost
        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0]
        )[-2]
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MulOpCost, ctx, processes,
                                                   desc_mapping, cluster)

        # calc comm op cost
        serial_op = dist_op.serial_op
        vars = serial_op.block.vars

        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("Y")[0])[-2]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = {"use_calc_stream": True, "use_model_parallel": True}

        var_names = serial_op.output("Out")
        c_allreduce_sum_desc_mapping = build_comm_desc_from_dist_op(
            "c_allreduce_sum",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
<<<<<<< HEAD
            parallel_axis=parallel_axis,
        )

        comm_op_cost_list = build_comm_costs_from_descs(
            AllreduceSumOpCost,
            ctx,
            processes,
            c_allreduce_sum_desc_mapping,
            cluster,
        )
=======
            parallel_axis=parallel_axis)

        # print("dist_matmul.py dist_op: ", dist_op)
        comm_op_cost_list = build_comm_costs_from_descs(
            AllreduceSumOpCost, ctx, processes, c_allreduce_sum_desc_mapping,
            cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        res_cost = [cost_mapping, comm_op_cost_list]

        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[-2]) or is_dim_shard(
<<<<<<< HEAD
            y_dims_mapping[-1]
        ):
=======
                y_dims_mapping[-1]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
<<<<<<< HEAD
        assert (
            op_dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )
=======
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # check validation of inputs / outputs
        for input_name in src_op.desc.input_names():
            assert input_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                input_name
            )
=======
                input_name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            assert len(kwargs[input_name]) == len(
                src_op.desc.input(input_name)
            ), "number of tensor for input [{}] is not match".format(input_name)
        for output_name in src_op.desc.output_names():
            assert output_name in kwargs, "input [{}] is not given".format(
<<<<<<< HEAD
                output_name
            )
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name
            )

        X_var = main_block._var_recursive(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name
        )[-2]
        assert (
            matmul_row_dim_mapping >= 0
        ), "row_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            matmul_row_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = matmul_row_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        check_variable_and_dtype(
            X_var, 'x', ['float16', 'float32', 'float64', 'uint16'], 'linear'
        )
        check_dtype(
            X_var.dtype,
            'dtype',
            ['float16', 'float32', 'float64', 'uint16'],
            'linear',
        )
=======
                output_name)
            assert len(kwargs[output_name]) == len(
                src_op.desc.output(output_name)
            ), "number of tensor for input [{}] is not match".format(
                output_name)

        X_var = main_block.var(kwargs['X'][0])
        Weight_var = main_block._var_recursive(kwargs['Y'][0])
        Out_var = main_block.var(kwargs['Out'][0])

        # TODO infer logic comm presentation
        matmul_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name)[-2]
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # attrs = {'trans_x': False, 'trans_y': False}
        attrs = {
            "x_num_col_dims": src_op.desc.attr("x_num_col_dims"),
            "y_num_col_dims": src_op.desc.attr("y_num_col_dims"),
<<<<<<< HEAD
            OP_ROLE_KEY: src_op.attr('op_role'),
=======
            OP_ROLE_KEY: src_op.attr('op_role')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        inputs = {'X': X_var, 'Y': Weight_var}

        # infer out var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None
<<<<<<< HEAD
        ref_shape = infer_shape(
            main_block, Out_var, out_tensor_dist_attr, out_var_dist_attr
        )

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(["c_allreduce_sum", 'tmp'])
            ),
=======
        ref_shape = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_allreduce_sum", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            shape=Out_var.shape,
            dtype=Out_var.dtype,
            type=Out_var.type,
            lod_level=Out_var.lod_level,
            persistable=False,
            is_data=False,
<<<<<<< HEAD
            need_check_feed=Out_var.desc.need_check_feed(),
        )
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(
            intermediate_var_0, out_var_dist_attr
        )
=======
            need_check_feed=Out_var.desc.need_check_feed())
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             out_var_dist_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        inputs_ref_shape = {}
        inputs_original_shape = {}
        for var_name in inputs:
            var = inputs[var_name]
            inputs_original_shape[var_name] = var.shape
            input_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(var)
            input_var_dist_attr = op_dist_attr.get_input_dist_attr(var.name)
<<<<<<< HEAD
            input_ref_shape = infer_shape(
                main_block, var, input_tensor_dist_attr, input_var_dist_attr
            )
            inputs_ref_shape[var_name] = input_ref_shape
            var.desc.set_shape(input_ref_shape)

        mul_op = main_block.append_op(
            type='mul',
            inputs=inputs,
            outputs={'Out': intermediate_var_0},
            attrs=attrs,
        )
=======
            input_ref_shape = infer_shape(main_block, var,
                                          input_tensor_dist_attr,
                                          input_var_dist_attr)
            inputs_ref_shape[var_name] = input_ref_shape
            var.desc.set_shape(input_ref_shape)

        mul_op = main_block.append_op(type='mul',
                                      inputs=inputs,
                                      outputs={'Out': intermediate_var_0},
                                      attrs=attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if intermediate_var_0.shape != ref_shape:
            intermediate_var_0.desc.set_shape(ref_shape)

        for var_name in inputs:
            var = inputs[var_name]
            original_shape = inputs_original_shape[var_name]
            var.desc.set_shape(original_shape)

        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': intermediate_var_0},
            outputs={'Out': Out_var},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
<<<<<<< HEAD
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )
=======
                OP_ROLE_KEY: src_op.attr('op_role')
            })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if Out_var.shape != ref_shape:
            Out_var.desc.set_shape(ref_shape)

        # set dist op's dist_attr with serial op's dist_attr
        # matmulv2
<<<<<<< HEAD
        matmulv2_op_dist_attr = OperatorDistAttr()
=======
        matmulv2_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        matmulv2_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        matmulv2_op_dist_attr.impl_type = op_dist_attr.impl_type
        matmulv2_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in mul_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
                op_dist_attr
            )
            matmulv2_op_dist_attr.set_input_dist_attr(
                input_varname, input_dist_attr
            )
        output_varname = mul_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr
        )
        matmulv2_op_dist_attr.set_output_dist_attr(
            output_varname, output_dist_attr
        )
        ctx.set_op_dist_attr_for_program(mul_op, matmulv2_op_dist_attr)

        # allreduce
        allreduce_op_dist_attr = OperatorDistAttr()
=======
                op_dist_attr)
            matmulv2_op_dist_attr.set_input_dist_attr(input_varname,
                                                      input_dist_attr)
        output_varname = mul_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr)
        matmulv2_op_dist_attr.set_output_dist_attr(output_varname,
                                                   output_dist_attr)
        ctx.set_op_dist_attr_for_program(mul_op, matmulv2_op_dist_attr)

        # allreduce
        allreduce_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        allreduce_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        allreduce_op_dist_attr.impl_type = op_dist_attr.impl_type
        allreduce_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in c_allreduce_sum_op.desc.input_arg_names():
<<<<<<< HEAD
            input_var = main_block._var_recursive(input_varname)
            tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(input_var)
            assert tensor_dist_attr is not None
            allreduce_op_dist_attr.set_input_dist_attr(
                input_varname, tensor_dist_attr
            )
        for output_varname in c_allreduce_sum_op.desc.output_arg_names():
            output_dist_attr = op_dist_attr.get_output_dist_attr(output_varname)
            assert output_dist_attr is not None, "dist_attr is {}".format(
                op_dist_attr
            )
            allreduce_op_dist_attr.set_output_dist_attr(
                output_varname, output_dist_attr
            )
        ctx.set_op_dist_attr_for_program(
            c_allreduce_sum_op, allreduce_op_dist_attr
        )

        # init param sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            _init_param_sync(
                Weight_var, dist_op_context, startup_block, ctx, rank_id
            )
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @staticmethod
    def backward(ctx, *args, **kwargs):
        _right_operand_parameter_matmul_backward(ctx, *args, **kwargs)


# ReplicateParallel
class DistributedMulImpl2(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedMulImpl2, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        res = []
        backward_op = dist_op.serial_op
        dist_attr = dist_op.dist_attr
        main_block = backward_op.block
<<<<<<< HEAD

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MulGradOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        vars = main_block.vars

        # calc comp op cost
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        process_mesh = dist_attr.process_mesh
        processes = process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MulGradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("X")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0]
        if (
            batch_size_axis > -1
            and mesh_shape[batch_size_axis] > 1
            and is_parameter_related(backward_op.input("Y")[0], main_block)
        ):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            backward_op.input("X")[0])
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[
                batch_size_axis] > 1 and is_parameter_related(
                    backward_op.input("Y")[0], main_block):
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('Y@GRAD')[0]]
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return res

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        cost_mapping = build_comp_costs_from_descs(
            MulOpCost, ctx, processes, desc_mapping, cluster
        )
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        cost_mapping = build_comp_costs_from_descs(MulOpCost, ctx, processes,
                                                   desc_mapping, cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        res_cost = [cost_mapping]
        return res_cost

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        y_name = op_desc.input('Y')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            x_dims_mapping[-2]
        ):
=======
                x_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False
        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            y_dims_mapping[-2]
        ):
=======
                y_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        if is_valid_list_index(out_dims_mapping, -2) and is_dim_shard(
<<<<<<< HEAD
            out_dims_mapping[-2]
        ):
=======
                out_dims_mapping[-2]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        return True

    def is_auto_compatible(self, dist_op):
<<<<<<< HEAD
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
=======
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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


<<<<<<< HEAD
register_distributed_operator_impl(
    "mul", DistributedMulImpl0("column_parallel")
)
register_distributed_operator_impl("mul", DistributedMulImpl1("row_parallel"))
register_distributed_operator_impl(
    "mul", DistributedMulImpl2("replicate_parallel")
)
=======
register_distributed_operator_impl("mul",
                                   DistributedMulImpl0("column_parallel"))
register_distributed_operator_impl("mul", DistributedMulImpl1("row_parallel"))
register_distributed_operator_impl("mul",
                                   DistributedMulImpl2("replicate_parallel"))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
