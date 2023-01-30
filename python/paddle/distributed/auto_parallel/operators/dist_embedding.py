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
    EmbeddingGradOpCost,
    EmbeddingOpCost,
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
    _get_idx_in_axis,
    compute_compatible_and_update_dim_mapping,
    is_dim_replicate,
    is_dim_shard,
    set_var_dist_attr,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    gradient_synchronization,
    infer_shape,
    naive_copy_op_dist_attr_for_program,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    set_comm_op_dist_attr_for_program,
)


class DistributedEmbedding(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(
    DistributedEmbedding("lookup_table_v2")
)
register_distributed_operator_impl_container(
    DistributedEmbedding("c_embedding")
)
register_distributed_operator_impl_container(
    DistributedEmbedding("lookup_table")
)
=======
from .common import infer_shape
from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import gradient_synchronization
from .common import register_distributed_operator_impl, set_comm_op_dist_attr_for_program, naive_copy_op_dist_attr_for_program, is_parameter_related
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from ..dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from paddle.fluid import core, unique_name
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import Program, Parameter, Variable
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..process_group import new_process_group
from ..utils import _get_comm_group, _get_idx_in_axis, _get_corresponding_rank, set_var_dist_attr
from ..cost import build_comp_desc_from_dist_op, build_comm_desc_from_dist_op
from ..cost import build_comm_costs_from_descs, build_comp_costs_from_descs, build_dp_costs
from ..cost import EmbeddingOpCost, EmbeddingGradOpCost
from paddle.distributed.auto_parallel.cost.comm_op_cost import AllreduceSumOpCost, IdentityOpCost


class DistributedEmbedding(DistributedOperatorImplContainer):

    def __init__(self, op_type):
        super(DistributedEmbedding, self).__init__(op_type)


register_distributed_operator_impl_container(
    DistributedEmbedding("lookup_table_v2"))
register_distributed_operator_impl_container(
    DistributedEmbedding("c_embedding"))
register_distributed_operator_impl_container(
    DistributedEmbedding("lookup_table"))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def adopt_lookup_table_v1(ctx, main_block, src_op, Ids_var):

<<<<<<< HEAD
    assert (
        len(Ids_var.shape) == 3
    ), "input Ids to lookup_table should have 3 dimensions but got [{}] with shape [{}]".format(
        Ids_var.name, Ids_var.shape
    )
=======
    assert len(
        Ids_var.shape
    ) == 3, "input Ids to lookup_table should have 3 dimensions but got [{}] with shape [{}]".format(
        Ids_var.name, Ids_var.shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if not Ids_var.stop_gradient:
        raise NotImplementedError(
            'Requiring the gradient of Ids of lookup_table(v1）dist op is not currently supported. Please open an issue with details on your use case so that we can prioritize adding this (for instance, adversarial training for language model).'
        )

    target_shape = list(Ids_var.shape[:-1])
    intermediate_var_0 = main_block.create_var(
<<<<<<< HEAD
        name=unique_name.generate_with_ignorable_key(
            ".".join(["dist_reshape", 'tmp'])
        ),
=======
        name=unique_name.generate_with_ignorable_key(".".join(
            ["dist_reshape", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        dtype=Ids_var.dtype,
        shape=target_shape,
        type=core.VarDesc.VarType.LOD_TENSOR,
        persistable=False,
<<<<<<< HEAD
        stop_gradient=True,
    )

    target_shape = [0] + list(Ids_var.shape[:-1])
    xshape_var = main_block.create_var(
        name=unique_name.generate_with_ignorable_key(
            ".".join(["dist_Xshape", 'tmp'])
        ),
=======
        stop_gradient=True)

    target_shape = [0] + list(Ids_var.shape[:-1])
    xshape_var = main_block.create_var(
        name=unique_name.generate_with_ignorable_key(".".join(
            ["dist_Xshape", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        dtype=Ids_var.dtype,
        shape=target_shape,
        type=core.VarDesc.VarType.LOD_TENSOR,
        persistable=False,
<<<<<<< HEAD
        stop_gradient=True,
    )

    # TODO use inplace reshape for memory saving
    reshape_op = main_block.append_op(
        type='reshape2',
        inputs={'X': [Ids_var]},
        outputs={'Out': [intermediate_var_0], 'XShape': [xshape_var]},
        attrs={
            "shape": [0, -1],
        },
    )
=======
        stop_gradient=True)

    # TODO use inplace reshape for memory saving
    reshape_op = main_block.append_op(type='reshape2',
                                      inputs={'X': [Ids_var]},
                                      outputs={
                                          'Out': [intermediate_var_0],
                                          'XShape': [xshape_var]
                                      },
                                      attrs={
                                          "shape": [0, -1],
                                      })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # set dist attr
    op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
    Ids_var_dist_attr = op_dist_attr.get_input_dist_attr(Ids_var.name)
    assert Ids_var_dist_attr is not None
    intermediate_var_0_dist_attr = set_var_dist_attr(
<<<<<<< HEAD
        ctx,
        intermediate_var_0,
        Ids_var_dist_attr.dims_mapping,
        Ids_var_dist_attr.process_mesh,
    )
    set_var_dist_attr(
        ctx,
        xshape_var,
        [-1] + list(Ids_var_dist_attr.dims_mapping),
        Ids_var_dist_attr.process_mesh,
    )
    op_dist_attr.del_input_dist_attr(Ids_var.name)
    op_dist_attr.set_input_dist_attr(
        intermediate_var_0.name, intermediate_var_0_dist_attr
    )

    new_op_dist_attr = OperatorDistAttr()
    new_op_dist_attr.process_mesh = Ids_var_dist_attr.process_mesh
    new_op_dist_attr.impl_type = "default"
    new_op_dist_attr.impl_idx = 0
    new_op_dist_attr.set_input_dims_mapping(
        Ids_var.name, Ids_var_dist_attr.dims_mapping
    )
    new_op_dist_attr.set_output_dims_mapping(
        intermediate_var_0.name, Ids_var_dist_attr.dims_mapping
    )
    new_op_dist_attr.set_output_dims_mapping(
        xshape_var.name, [-1] + list(Ids_var_dist_attr.dims_mapping)
    )
=======
        ctx, intermediate_var_0, Ids_var_dist_attr.dims_mapping,
        Ids_var_dist_attr.process_mesh)
    set_var_dist_attr(ctx, xshape_var,
                      [-1] + list(Ids_var_dist_attr.dims_mapping),
                      Ids_var_dist_attr.process_mesh)
    op_dist_attr.del_input_dist_attr(Ids_var.name)
    op_dist_attr.set_input_dist_attr(intermediate_var_0.name,
                                     intermediate_var_0_dist_attr)

    new_op_dist_attr = OperatorDistributedAttribute()
    new_op_dist_attr.process_mesh = Ids_var_dist_attr.process_mesh
    new_op_dist_attr.impl_type = "default"
    new_op_dist_attr.impl_idx = 0
    new_op_dist_attr.set_input_dims_mapping(Ids_var.name,
                                            Ids_var_dist_attr.dims_mapping)
    new_op_dist_attr.set_output_dims_mapping(intermediate_var_0.name,
                                             Ids_var_dist_attr.dims_mapping)
    new_op_dist_attr.set_output_dims_mapping(
        xshape_var.name, [-1] + list(Ids_var_dist_attr.dims_mapping))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    ctx.set_op_dist_attr_for_program(reshape_op, new_op_dist_attr)

    return intermediate_var_0


# RowParallel
class DistributedEmbeddingImpl(DistributedOperatorImpl):
<<<<<<< HEAD
    def __init__(self, name):
        super().__init__(name)
=======

    def __init__(self, name):
        super(DistributedEmbeddingImpl, self).__init__(name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._forward_implemented = True
        self._backward_implemented = True

    def calc_cost(self, op_role, dist_op, ctx, cluster):
        """Calculate the cost by the op role."""
        cost = None
        if int(op_role) == int(OpRole.Forward):
            cost = self.calc_fwd_cost(dist_op, ctx, cluster)
        elif int(op_role) == int(OpRole.Backward):
            cost = self.calc_bwd_cost(dist_op, ctx, cluster)
        assert cost is not None
        return cost

    def calc_fwd_cost(self, dist_op, ctx, cluster):
        # calc comp op cost
<<<<<<< HEAD
        desc_mapping = build_comp_desc_from_dist_op(
            dist_op=dist_op, dist_context=ctx
        )
        processes = dist_op.dist_attr.process_mesh.process_ids
        # embedding need start_index
        cost_mapping = build_comp_costs_from_descs(
            EmbeddingOpCost, ctx, processes, desc_mapping, cluster
        )

        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("W")[0]
        )[0]
=======
        desc_mapping = build_comp_desc_from_dist_op(dist_op=dist_op,
                                                    dist_context=ctx)
        processes = dist_op.dist_attr.process_mesh.processes
        # embedding need start_index
        cost_mapping = build_comp_costs_from_descs(EmbeddingOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)

        serial_op = dist_op.serial_op
        parallel_axis = dist_op.dist_attr.get_input_dims_mapping(
            serial_op.input("W")[0])[0]
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

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        main_block = backward_op.block
        dist_attr = dist_op.dist_attr

        embedding_row_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("W")[0]
        )[0]
=======
            backward_op.input("W")[0])[0]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        parallel_axis = embedding_row_dim_mapping
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        var_names = [backward_op.input("Out@GRAD")[0]]
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
            EmbeddingGradOpCost, ctx, processes, desc_mapping, cluster
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
        cost_mapping = build_comp_costs_from_descs(EmbeddingGradOpCost, ctx,
                                                   processes, desc_mapping,
                                                   cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        res.append(cost_mapping)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            backward_op.input("Ids")[0]
        )
        mesh_shape = process_mesh.shape
=======
            backward_op.input("Ids")[0])
        mesh_shape = process_mesh.topology
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('W@GRAD')[0]]
<<<<<<< HEAD
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )
=======
            build_dp_costs(res, dist_op, ctx, var_names, attrs, parallel_axis,
                           cluster)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return res

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        if is_dim_replicate(w_dims_mapping[-2]) or is_dim_shard(
<<<<<<< HEAD
            w_dims_mapping[-1]
        ):
=======
                w_dims_mapping[-1]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in ids_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:]:
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

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)

<<<<<<< HEAD
        if ids_dims_mapping != out_dims_mapping[: len(ids_dims_mapping)]:
=======
        if ids_dims_mapping != out_dims_mapping[:len(ids_dims_mapping)]:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        out_name = op_desc.output('Out')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        for i in range(len(ids_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
<<<<<<< HEAD
                [ids_dims_mapping, out_dims_mapping], [i, i]
            )
=======
                [ids_dims_mapping, out_dims_mapping], [i, i])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            if dim_changed:
                changed = True

        dim_changed = compute_compatible_and_update_dim_mapping(
<<<<<<< HEAD
            [w_dims_mapping, out_dims_mapping], [-1, -1]
        )
        if dim_changed:
            changed = True

        if changed:
            op_dist_attr.set_input_dims_mapping(ids_name, ids_dims_mapping)
            op_dist_attr.set_input_dims_mapping(w_name, w_dims_mapping)
            op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)

=======
            [w_dims_mapping, out_dims_mapping], [-1, -1])
        if dim_changed:
            changed = True

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
=======
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # check validation of inputs / outputs
        assert 'Ids' in kwargs, "input [{}] is not given".format('Ids')
        assert 'W' in kwargs, "input [{}] is not given".format('W')
        assert 'Out' in kwargs, "output [{}] is not given".format('Out')

<<<<<<< HEAD
        assert (
            len(kwargs['Ids']) == 1
        ), "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['Ids']
        )
        assert (
            len(kwargs['W']) == 1
        ), "row_parallel_embedding input W take 1 variable but got {}".format(
            kwargs['W']
        )
        assert (
            len(kwargs['Out']) == 1
        ), "row_parallel_embedding output Out take 1 variable but got {}".format(
            kwargs['Out']
        )

        Ids_var = main_block._var_recursive(kwargs['Ids'][0])
        Weight_var = main_block._var_recursive(kwargs['W'][0])
        Out_var = main_block._var_recursive(kwargs['Out'][0])
=======
        assert len(
            kwargs['Ids']
        ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['Ids'])
        assert len(
            kwargs['W']
        ) == 1, "row_parallel_embedding input W take 1 variable but got {}".format(
            kwargs['W'])
        assert len(
            kwargs['Out']
        ) == 1, "row_parallel_embedding output Out take 1 variable but got {}".format(
            kwargs['Out'])

        Ids_var = main_block.var(kwargs['Ids'][0])
        Weight_var = main_block._var_recursive(kwargs['W'][0])
        Out_var = main_block.var(kwargs['Out'][0])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # support lookup_table_v1
        if src_op.type == 'lookup_table':
            Ids_var = adopt_lookup_table_v1(ctx, main_block, src_op, Ids_var)

        # got dist attribute info
        embedding_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
<<<<<<< HEAD
            Weight_var.name
        )[0]
        assert (
            embedding_row_dim_mapping >= 0
        ), "row_parallel_embedding's row should be divided by a specific mesh axis, but got [{}]".format(
            embedding_row_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in process_mesh_group:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        # A generalized method to caculate embedding offset using cartisian product
        relative_idx = _get_idx_in_axis(
            process_mesh_group,
            process_mesh_shape,
            embedding_row_dim_mapping,
            rank_id,
        )
=======
            Weight_var.name)[0]
        assert embedding_row_dim_mapping >= 0, "row_parallel_embedding's row should be divided by a specific mesh axis, but got [{}]".format(
            embedding_row_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in process_mesh_group:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        # A generalized method to caculate embedding offset using cartisian product
        relative_idx = _get_idx_in_axis(process_mesh_group, process_mesh_shape,
                                        embedding_row_dim_mapping, rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        per_part_size = Weight_var.shape[0]
        relative_idx = relative_idx * per_part_size

        # TODO caculate ring id
        parallel_axis = embedding_row_dim_mapping
<<<<<<< HEAD
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        # append op
        check_variable_and_dtype(
            Ids_var, 'input', ['int32', 'int64'], 'c_embedding'
        )
=======
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        # append op
        check_variable_and_dtype(Ids_var, 'input', ['int32', 'int64'],
                                 'c_embedding')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # infer new var shape with op dist attr
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
                ".".join(["c_embedding", 'tmp'])
            ),
=======
        ref_shape = infer_shape(main_block, Out_var, out_tensor_dist_attr,
                                out_var_dist_attr)

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_embedding", 'tmp'])),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dtype=Weight_var.dtype,
            shape=Out_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
<<<<<<< HEAD
            stop_gradient=Out_var.stop_gradient,
        )
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(
            intermediate_var_0, out_var_dist_attr
        )

        check_variable_and_dtype(
            Out_var,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'c_allreduce_sum',
        )

        c_embedding_op = main_block.append_op(
            type='c_embedding',
            inputs={'Ids': [Ids_var], 'W': [Weight_var]},
            outputs={'Out': [intermediate_var_0]},
            attrs={
                "start_index": relative_idx,
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )
=======
            stop_gradient=Out_var.stop_gradient)
        # set intermediate_var_0's dist_attr with Out_var's dist_attr
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             out_var_dist_attr)

        check_variable_and_dtype(
            Out_var, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'c_allreduce_sum')

        c_embedding_op = main_block.append_op(
            type='c_embedding',
            inputs={
                'Ids': [Ids_var],
                'W': [Weight_var]
            },
            outputs={'Out': [intermediate_var_0]},
            attrs={
                "start_index": relative_idx,
                OP_ROLE_KEY: src_op.attr('op_role')
            })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if intermediate_var_0.shape != ref_shape:
            intermediate_var_0.desc.set_shape(ref_shape)

        # use_model_parallel
        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': [intermediate_var_0]},
            outputs={'Out': [Out_var]},
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
        embedding_op_dist_attr = OperatorDistAttr()
=======
        embedding_op_dist_attr = OperatorDistributedAttribute()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        embedding_op_dist_attr.process_mesh = op_dist_attr.process_mesh
        embedding_op_dist_attr.impl_type = op_dist_attr.impl_type
        embedding_op_dist_attr.impl_idx = op_dist_attr.impl_idx
        for input_varname in c_embedding_op.desc.input_arg_names():
            input_dist_attr = op_dist_attr.get_input_dist_attr(input_varname)
            assert input_dist_attr is not None, "dist_attr is {}".format(
<<<<<<< HEAD
                op_dist_attr
            )
            embedding_op_dist_attr.set_input_dist_attr(
                input_varname, input_dist_attr
            )
        output_varname = c_embedding_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr
        )
        embedding_op_dist_attr.set_output_dist_attr(
            output_varname, output_dist_attr
        )
        ctx.set_op_dist_attr_for_program(c_embedding_op, embedding_op_dist_attr)

        # allreduce
        allreduce_op_dist_attr = OperatorDistAttr()
=======
                op_dist_attr)
            embedding_op_dist_attr.set_input_dist_attr(input_varname,
                                                       input_dist_attr)
        output_varname = c_embedding_op.desc.output_arg_names()[0]
        output_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert output_dist_attr is not None, "dist_attr is {}".format(
            op_dist_attr)
        embedding_op_dist_attr.set_output_dist_attr(output_varname,
                                                    output_dist_attr)
        ctx.set_op_dist_attr_for_program(c_embedding_op, embedding_op_dist_attr)

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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # param initialization sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            if Weight_var.name in dist_op_context.already_init_sync_vars:
                return
            dist_op_context.already_init_sync_vars.add(Weight_var.name)
            param = startup_block.var(Weight_var.name)
            param_dist_attr = ctx.get_tensor_dist_attr_for_program(param)
            process_mesh = param_dist_attr.process_mesh
            dim_mapping = param_dist_attr.dims_mapping

            # NOTE all not splited axis should be presented in mesh
<<<<<<< HEAD
            for axis, size in enumerate(process_mesh.shape):
                if size <= 1 or axis in dim_mapping:
                    pass
                else:
                    group_ranks = _get_comm_group(
                        process_mesh.process_ids,
                        process_mesh.shape,
                        axis,
                        rank_id,
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
=======
            for axis, size in enumerate(process_mesh.topology):
                if size <= 1 or axis in dim_mapping:
                    pass
                else:
                    group_ranks = _get_comm_group(process_mesh.processes,
                                                  process_mesh.topology, axis,
                                                  rank_id)
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @staticmethod
    def backward(ctx, *args, **kwargs):

        # by now the backward function only insert the gradient allreduce for dist op itself
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        backward_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
<<<<<<< HEAD
        assert (
            dist_attr is not None
        ), "backward op [{}] don't have dist attribute !".format(
            str(backward_op)
        )

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, dist_attr.process_mesh, rank_id
            )
=======
        assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(backward_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, dist_attr.process_mesh,
                                              rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        assert 'Ids' in kwargs, "input [{}] is not given".format('Ids')
        assert 'W' in kwargs, "input [{}] is not given".format('W')
        assert 'Out@GRAD' in kwargs, "input [{}] is not given".format('Out')
        assert 'W@GRAD' in kwargs, "output [{}] is not given".format('W@GRAD')

<<<<<<< HEAD
        assert (
            len(kwargs['Ids']) == 1
        ), "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['Ids']
        )
        assert (
            len(kwargs['W']) == 1
        ), "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['W']
        )
        assert (
            len(kwargs['Out@GRAD']) == 1
        ), "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['Out']
        )
        assert (
            len(kwargs['W@GRAD']) == 1
        ), "row_parallel_embedding output Ids take 1 variable but got {}".format(
            kwargs['W@GRAD']
        )

        Ids_var = main_block._var_recursive(kwargs['Ids'][0])
        Weight_var = main_block._var_recursive(kwargs['W'][0])
        Out_grad = main_block._var_recursive(kwargs['Out@GRAD'][0])
        Weight_grad = main_block._var_recursive(kwargs['W@GRAD'][0])

        embedding_row_dim_mapping = dist_attr.get_input_dims_mapping(
            Weight_var.name
        )[0]
        assert (
            embedding_row_dim_mapping >= 0
        ), "row_parallel_embedding's row should be divided by a specific mesh axis, but got [{}]".format(
            embedding_row_dim_mapping
        )
        process_mesh_shape = dist_attr.process_mesh.shape
        process_mesh_group = dist_attr.process_mesh.process_ids

        # A generalized method to caculate embedding offset using cartisian product
        relative_idx = _get_idx_in_axis(
            process_mesh_group,
            process_mesh_shape,
            embedding_row_dim_mapping,
            rank_id,
        )
=======
        assert len(
            kwargs['Ids']
        ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['Ids'])
        assert len(
            kwargs['W']
        ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['W'])
        assert len(
            kwargs['Out@GRAD']
        ) == 1, "row_parallel_embedding input Ids take 1 variable but got {}".format(
            kwargs['Out'])
        assert len(
            kwargs['W@GRAD']
        ) == 1, "row_parallel_embedding output Ids take 1 variable but got {}".format(
            kwargs['W@GRAD'])

        Ids_var = main_block.var(kwargs['Ids'][0])
        Weight_var = main_block.var(kwargs['W'][0])
        Out_grad = main_block.var(kwargs['Out@GRAD'][0])
        Weight_grad = main_block.var(kwargs['W@GRAD'][0])

        embedding_row_dim_mapping = dist_attr.get_input_dims_mapping(
            Weight_var.name)[0]
        assert embedding_row_dim_mapping >= 0, "row_parallel_embedding's row should be divided by a specific mesh axis, but got [{}]".format(
            embedding_row_dim_mapping)
        process_mesh_shape = dist_attr.process_mesh.topology
        process_mesh_group = dist_attr.process_mesh.processes

        # A generalized method to caculate embedding offset using cartisian product
        relative_idx = _get_idx_in_axis(process_mesh_group, process_mesh_shape,
                                        embedding_row_dim_mapping, rank_id)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        per_part_size = Weight_var.shape[0]
        relative_idx = relative_idx * per_part_size

        check_variable_and_dtype(
<<<<<<< HEAD
            Out_grad,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            '_c_identity',
        )

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(
                ".".join(["c_embedding", '@tmp_0@GRAD'])
            ),
=======
            Out_grad, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], '_c_identity')

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_embedding", '@tmp_0@GRAD'])),
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
            process_mesh_group,
            process_mesh_shape,
            embedding_row_dim_mapping,
            rank_id,
        )
=======
        ctx.set_tensor_dist_attr_for_program(intermediate_var_0,
                                             out_grad_dist_attr)

        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      embedding_row_dim_mapping, rank_id)
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
            intermediate_var_0, 'x', ['float16', 'float32', 'float64'], 'linear'
        )
        check_dtype(
            intermediate_var_0.dtype,
            'dtype',
            ['float16', 'float32', 'float64'],
            'linear',
        )

        set_comm_op_dist_attr_for_program(
            c_identity_op, dist_attr.process_mesh, out_grad_dist_attr, ctx
        )
=======
            })
        check_variable_and_dtype(intermediate_var_0, 'x',
                                 ['float16', 'float32', 'float64'], 'linear')
        check_dtype(intermediate_var_0.dtype, 'dtype',
                    ['float16', 'float32', 'float64'], 'linear')

        set_comm_op_dist_attr_for_program(c_identity_op, dist_attr.process_mesh,
                                          out_grad_dist_attr, ctx)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        c_embedding_grad_op_desc = main_block.append_op(type='nop').desc
        c_embedding_grad_op_desc.set_type("c_embedding_grad")
        c_embedding_grad_op_desc.set_input('Ids', [Ids_var.name])
        c_embedding_grad_op_desc.set_input('W', [Weight_var.name])
<<<<<<< HEAD
        c_embedding_grad_op_desc.set_input(
            'Out@GRAD', [intermediate_var_0.name]
        )
=======
        c_embedding_grad_op_desc.set_input('Out@GRAD',
                                           [intermediate_var_0.name])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        c_embedding_grad_op_desc.set_output('W@GRAD', [Weight_grad.name])
        c_embedding_grad_op_desc._set_attr('start_index', relative_idx)
        c_embedding_grad_op_desc._set_attr(OP_ROLE_KEY, OpRole.Backward)

        c_embedding_grad_op = main_block.ops[-1]
        assert c_embedding_grad_op.type == "c_embedding_grad"
<<<<<<< HEAD
        naive_copy_op_dist_attr_for_program(
            c_embedding_grad_op, backward_op, ctx
        )
=======
        naive_copy_op_dist_attr_for_program(c_embedding_grad_op, backward_op,
                                            ctx)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # data parallel gradient synchronization
        act_grad_names = [Ids_var.name]
        out_grad_names = [kwargs['W@GRAD'][0]]

<<<<<<< HEAD
        gradient_synchronization(
            ctx, backward_op, act_grad_names, out_grad_names, rank_id
        )


register_distributed_operator_impl(
    "lookup_table_v2", DistributedEmbeddingImpl("row_parallel")
)
register_distributed_operator_impl(
    "c_embedding", DistributedEmbeddingImpl("row_parallel")
)
register_distributed_operator_impl(
    "lookup_table", DistributedEmbeddingImpl("row_parallel")
)
=======
        gradient_synchronization(ctx, backward_op, act_grad_names,
                                 out_grad_names, rank_id)


register_distributed_operator_impl("lookup_table_v2",
                                   DistributedEmbeddingImpl("row_parallel"))
register_distributed_operator_impl("c_embedding",
                                   DistributedEmbeddingImpl("row_parallel"))
register_distributed_operator_impl("lookup_table",
                                   DistributedEmbeddingImpl("row_parallel"))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
