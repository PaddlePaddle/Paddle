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

from paddle.common_ops_import import check_variable_and_dtype
from paddle.distributed.auto_parallel.static.cost.comm_op_cost import (
    AllreduceSumOpCost,
    IdentityOpCost,
)
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY, OpRole
from paddle.framework import core
from paddle.utils import unique_name

from ..completion import get_phi_spmd_rule
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
    get_dist_tensor_spec,
    is_dim_replicate,
    is_dim_shard,
    set_var_dist_attr,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    ParallelMode,
    get_default_distributed_operator_impl,
    gradient_synchronization,
    naive_copy_op_dist_attr_for_program,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
    set_comm_op_dist_attr_for_program,
    update_op_dims_mapping,
)


class DistributedEmbedding(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)

    @staticmethod
    def update_dims_mapping(dist_op):
        # step1: prepare inputs need for rule (order args as PHI definition and filter out unnecessary args)
        op_desc = dist_op.serial_op.desc
        assert (
            dist_op.serial_op.type == "lookup_table_v2"
        ), f"{dist_op.serial_op.type} is not supported by dist embedding yet."

        x_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        out_name = op_desc.output('Out')[0]
        padding_idx = op_desc.attr('padding_idx')
        is_sparse = op_desc.attr('is_sparse')

        x_spec = get_dist_tensor_spec(dist_op, x_name)
        w_spec = get_dist_tensor_spec(dist_op, w_name)
        output_spec = get_dist_tensor_spec(dist_op, out_name, False)

        # step2: infer spmd
        rule = get_phi_spmd_rule("embedding")
        # tensor order following order in PHI definition
        fw_results = rule.infer_forward(x_spec, w_spec, padding_idx, is_sparse)
        bw_results = rule.infer_backward(
            x_spec, w_spec, output_spec, padding_idx, is_sparse
        )

        # step3: update dist_attr
        # tensor order following order in PHI definition
        changed = update_op_dims_mapping(
            dist_op, [x_name, w_name], [out_name], fw_results, bw_results
        )

        return changed

    @staticmethod
    def mapping_to_dist_operator_impl(dist_op, original_op_dist_attr):
        reverted = False
        op_dist_attr = dist_op.dist_attr
        op_desc = dist_op.serial_op.desc
        out_name = op_desc.output('Out')[0]
        out_dist_attr = op_dist_attr.get_output_dist_attr(out_name)

        # vocab parallel embedding
        if out_dist_attr._is_partial():
            op_dist_attr.impl_type = op_desc.type()
            op_dist_attr.impl_idx = 0
        # data parallel or col parallel of weight
        else:
            default_impl = get_default_distributed_operator_impl()
            op_dist_attr.impl_type = default_impl.type
            op_dist_attr.impl_idx = default_impl.idx

        return reverted


register_distributed_operator_impl_container(
    DistributedEmbedding("lookup_table_v2")
)
register_distributed_operator_impl_container(
    DistributedEmbedding("c_embedding")
)
register_distributed_operator_impl_container(
    DistributedEmbedding("lookup_table")
)


def adopt_lookup_table_v1(ctx, main_block, src_op, Ids_var):
    assert (
        len(Ids_var.shape) == 3
    ), f"input Ids to lookup_table should have 3 dimensions but got [{Ids_var.name}] with shape [{Ids_var.shape}]"
    if not Ids_var.stop_gradient:
        raise NotImplementedError(
            'Requiring the gradient of Ids of lookup_table(v1) dist op is not currently supported. Please open an issue with details on your use case so that we can prioritize adding this (for instance, adversarial training for language model).'
        )

    target_shape = list(Ids_var.shape[:-1])
    intermediate_var_0 = main_block.create_var(
        name=unique_name.generate_with_ignorable_key(
            ".".join(["dist_reshape", 'tmp'])
        ),
        dtype=Ids_var.dtype,
        shape=target_shape,
        type=core.VarDesc.VarType.LOD_TENSOR,
        persistable=False,
        stop_gradient=True,
    )

    target_shape = [0] + list(Ids_var.shape[:-1])
    xshape_var = main_block.create_var(
        name=unique_name.generate_with_ignorable_key(
            ".".join(["dist_Xshape", 'tmp'])
        ),
        dtype=Ids_var.dtype,
        shape=target_shape,
        type=core.VarDesc.VarType.LOD_TENSOR,
        persistable=False,
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

    # set dist attr
    op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)
    Ids_var_dist_attr = op_dist_attr.get_input_dist_attr(Ids_var.name)
    assert Ids_var_dist_attr is not None
    intermediate_var_0_dist_attr = set_var_dist_attr(
        ctx,
        intermediate_var_0,
        Ids_var_dist_attr.dims_mapping,
        Ids_var_dist_attr.process_mesh,
        chunk_id=Ids_var_dist_attr.chunk_id,
    )
    set_var_dist_attr(
        ctx,
        xshape_var,
        [-1] + list(Ids_var_dist_attr.dims_mapping),
        Ids_var_dist_attr.process_mesh,
        chunk_id=Ids_var_dist_attr.chunk_id,
    )
    # rename src_op's input
    src_op._rename_input(Ids_var.name, intermediate_var_0.name)
    op_dist_attr.del_input_dist_attr(Ids_var.name)
    op_dist_attr.set_input_dist_attr(
        intermediate_var_0.name, intermediate_var_0_dist_attr
    )

    new_op_dist_attr = OperatorDistAttr()
    new_op_dist_attr.process_mesh = Ids_var_dist_attr.process_mesh
    new_op_dist_attr.impl_type = "default"
    new_op_dist_attr.impl_idx = 0
    new_op_dist_attr.chunk_id = Ids_var_dist_attr.chunk_id
    new_op_dist_attr.set_input_dims_mapping(
        Ids_var.name, Ids_var_dist_attr.dims_mapping
    )
    new_op_dist_attr.set_output_dims_mapping(
        intermediate_var_0.name, Ids_var_dist_attr.dims_mapping
    )
    new_op_dist_attr.set_output_dims_mapping(
        xshape_var.name, [-1] + list(Ids_var_dist_attr.dims_mapping)
    )
    ctx.set_op_dist_attr_for_program(reshape_op, new_op_dist_attr)

    return intermediate_var_0


# RowParallel
class DistributedEmbeddingImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
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
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        var_names = serial_op.output("Out")
        c_allreduce_sum_desc_mapping = build_comm_desc_from_dist_op(
            "c_allreduce_sum",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
            parallel_axis=parallel_axis,
        )

        comm_op_cost_list = build_comm_costs_from_descs(
            AllreduceSumOpCost,
            ctx,
            processes,
            c_allreduce_sum_desc_mapping,
            cluster,
        )

        res_cost = [cost_mapping, comm_op_cost_list]

        return res_cost

    def calc_bwd_cost(self, dist_op, ctx, cluster):
        # by now the backward function only insert the gradient allreduce for dist op itself
        res = []
        backward_op = dist_op.serial_op
        main_block = backward_op.block
        dist_attr = dist_op.dist_attr

        embedding_row_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("W")[0]
        )[0]
        parallel_axis = embedding_row_dim_mapping
        attrs = {"use_calc_stream": True, "use_model_parallel": True}
        var_names = [backward_op.input("Out@GRAD")[0]]
        c_identity_desc_mapping = build_comm_desc_from_dist_op(
            "c_identity",
            dist_op,
            ctx,
            var_names,
            attrs=attrs,
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
        res.append(cost_mapping)

        # need gradient allreduce
        var_dim_mapping = dist_attr.get_input_dims_mapping(
            backward_op.input("Ids")[0]
        )
        mesh_shape = process_mesh.shape
        batch_size_axis = var_dim_mapping[0] if len(var_dim_mapping) > 0 else -1
        if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
            parallel_axis = batch_size_axis
            attrs = {"use_calc_stream": True}
            var_names = [backward_op.output('W@GRAD')[0]]
            build_dp_costs(
                res, dist_op, ctx, var_names, attrs, parallel_axis, cluster
            )

        return res

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        if is_dim_replicate(w_dims_mapping[-2]) or is_dim_shard(
            w_dims_mapping[-1]
        ):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in ids_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False

        if is_dim_shard(ids_dims_mapping[0]) and is_dim_shard(
            w_dims_mapping[-2]
        ):
            if ids_dims_mapping[0] == w_dims_mapping[-2]:
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
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)

        if ids_dims_mapping != out_dims_mapping[: len(ids_dims_mapping)]:
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
                [ids_dims_mapping, out_dims_mapping], [i, i]
            )
            if dim_changed:
                changed = True

        dim_changed = compute_compatible_and_update_dim_mapping(
            [w_dims_mapping, out_dims_mapping], [-1, -1]
        )
        if dim_changed:
            changed = True

        if changed:
            op_dist_attr.set_input_dims_mapping(ids_name, ids_dims_mapping)
            op_dist_attr.set_input_dims_mapping(w_name, w_dims_mapping)
            op_dist_attr.set_output_dims_mapping(out_name, out_dims_mapping)

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
        assert (
            op_dist_attr is not None
        ), f"forward op [{str(src_op)}] don't have dist attribute !"

        # check validation of inputs / outputs
        assert 'Ids' in kwargs, "input [{}] is not given".format('Ids')
        assert 'W' in kwargs, "input [{}] is not given".format('W')
        assert 'Out' in kwargs, "output [{}] is not given".format('Out')

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

        # support lookup_table_v1
        if src_op.type == 'lookup_table':
            Ids_var = adopt_lookup_table_v1(ctx, main_block, src_op, Ids_var)

        # got dist attribute info
        embedding_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name
        )[0]
        assert (
            embedding_row_dim_mapping >= 0
        ), f"row_parallel_embedding's row should be divided by a specific mesh axis, but got [{embedding_row_dim_mapping}]"
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in process_mesh_group:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        # A generalized method to calculate embedding offset using cartesian product
        relative_idx = _get_idx_in_axis(
            process_mesh_group,
            process_mesh_shape,
            embedding_row_dim_mapping,
            rank_id,
        )

        per_part_size = Weight_var.shape[0]
        relative_idx = relative_idx * per_part_size

        # TODO calculate ring id
        parallel_axis = embedding_row_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        # append op
        check_variable_and_dtype(
            Ids_var, 'input', ['int32', 'int64'], 'c_embedding'
        )

        # infer new var shape with op dist attr
        out_tensor_dist_attr = ctx.get_tensor_dist_attr_for_program(Out_var)
        assert out_tensor_dist_attr is not None
        out_var_dist_attr = op_dist_attr.get_output_dist_attr(Out_var.name)
        assert out_var_dist_attr is not None

        c_embedding_op_desc = main_block.append_op(type='nop').desc
        c_embedding_op_desc.set_type("c_embedding")
        c_embedding_op_desc.set_input('Ids', [Ids_var.name])
        c_embedding_op_desc.set_input('W', [Weight_var.name])
        c_embedding_op_desc.set_output('Out', [Out_var.name])
        c_embedding_op_desc._set_attr('start_index', relative_idx)
        c_embedding_op_desc._set_attr(OP_ROLE_KEY, src_op.attr('op_role'))
        c_embedding_op = main_block.ops[-1]
        assert c_embedding_op.type == "c_embedding"
        naive_copy_op_dist_attr_for_program(c_embedding_op, src_op, ctx)

        # use_model_parallel
        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': [Out_var]},
            outputs={'Out': [Out_var]},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
                OP_ROLE_KEY: src_op.attr('op_role'),
            },
        )
        c_allreduce_sum_op._set_attr(
            'op_namescope', '/' + ParallelMode.TensorParallel
        )
        # allreduce
        set_comm_op_dist_attr_for_program(
            c_allreduce_sum_op,
            op_dist_attr.process_mesh,
            out_var_dist_attr,
            ctx,
            chunk_id=op_dist_attr.chunk_id,
        )

        # param initialization sync
        if Weight_var.is_parameter and not op_dist_attr.is_recompute:
            if Weight_var.name in dist_op_context.already_init_sync_vars:
                return
            dist_op_context.already_init_sync_vars.add(Weight_var.name)
            param = startup_block.var(Weight_var.name)
            param_dist_attr = ctx.get_tensor_dist_attr_for_program(param)
            process_mesh = param_dist_attr.process_mesh
            dim_mapping = param_dist_attr.dims_mapping

            # NOTE all not splitted axis should be presented in mesh
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

    @staticmethod
    def backward(ctx, *args, **kwargs):
        # by now the backward function only insert the gradient allreduce for dist op itself
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        backward_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        dist_attr = ctx.get_op_dist_attr_for_program(backward_op)
        assert (
            dist_attr is not None
        ), f"backward op [{str(backward_op)}] don't have dist attribute !"

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, dist_attr.process_mesh, rank_id
            )

        assert 'Ids' in kwargs, "input [{}] is not given".format('Ids')
        assert 'W' in kwargs, "input [{}] is not given".format('W')
        assert 'Out@GRAD' in kwargs, "input [{}] is not given".format('Out')
        assert 'W@GRAD' in kwargs, "output [{}] is not given".format('W@GRAD')

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
        ), f"row_parallel_embedding's row should be divided by a specific mesh axis, but got [{embedding_row_dim_mapping}]"
        process_mesh_shape = dist_attr.process_mesh.shape
        process_mesh_group = dist_attr.process_mesh.process_ids

        # A generalized method to calculate embedding offset using cartesian product
        relative_idx = _get_idx_in_axis(
            process_mesh_group,
            process_mesh_shape,
            embedding_row_dim_mapping,
            rank_id,
        )
        per_part_size = Weight_var.shape[0]
        relative_idx = relative_idx * per_part_size

        c_embedding_grad_op_desc = main_block.append_op(type='nop').desc
        c_embedding_grad_op_desc.set_type("c_embedding_grad")
        c_embedding_grad_op_desc.set_input('Ids', [Ids_var.name])
        c_embedding_grad_op_desc.set_input('W', [Weight_var.name])
        c_embedding_grad_op_desc.set_input('Out@GRAD', [Out_grad.name])
        c_embedding_grad_op_desc.set_output('W@GRAD', [Weight_grad.name])
        c_embedding_grad_op_desc._set_attr('start_index', relative_idx)
        c_embedding_grad_op_desc._set_attr(OP_ROLE_KEY, OpRole.Backward)

        c_embedding_grad_op = main_block.ops[-1]
        assert c_embedding_grad_op.type == "c_embedding_grad"
        naive_copy_op_dist_attr_for_program(
            c_embedding_grad_op, backward_op, ctx
        )

        # data parallel gradient synchronization
        act_grad_names = [Ids_var.name]
        out_grad_names = [kwargs['W@GRAD'][0]]

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
