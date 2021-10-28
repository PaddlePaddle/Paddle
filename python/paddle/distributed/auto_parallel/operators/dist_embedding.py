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

from .common import DistributedOperator
from .common import DistributedOperatorImpl
from .common import register_distributed_operator
from .common import register_distributed_operator_impl
from .common import copy_distributed_attr_for_var
from .common import copy_distributed_attr_for_dist_op
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from ..attribute import OperatorDistributedAttribute
from paddle.fluid import core, unique_name
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from ..process import new_process_group
from ..utils import _get_comm_group, _get_idx_in_axis, _get_corresponding_rank


class DistributedEmbedding(DistributedOperator):
    def __init__(self, name):
        super(DistributedEmbedding, self).__init__()
        self._name = name


register_distributed_operator("lookup_table_v2",
                              DistributedEmbedding("embedding"))
register_distributed_operator("c_embedding", DistributedEmbedding("embedding"))


# RowParallel
class DistributedEmbeddingImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedEmbeddingImpl, self).__init__()
        self._name = name
        self._forward_implemented = True
        self._backward_implemented = True

    def is_process_mesh_compatible(self, op_dist_attr):
        """ No restriction for now. """
        return True

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        if is_dim_replicate(w_dims_mapping[-2]) or is_dim_shard(w_dims_mapping[
                -1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in ids_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_owner_op().desc
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        op_desc = op_dist_attr.get_owner_op().desc
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        out_name = op_desc.output('Out')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        for i in range(len(ids_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [ids_dims_mapping, out_dims_mapping], [i, i])
            if dim_changed:
                changed = True

        dim_changed = compute_compatible_and_update_dim_mapping(
            [w_dims_mapping, out_dims_mapping], [-1, -1])
        if dim_changed:
            changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        kwargs: inputname_mapping & outputname_mapping
        """

        dist_op_helper = ctx.get_dist_op_helper()
        main_block = dist_op_helper.get_dst_main_program().global_block()
        startup_block = dist_op_helper.get_dst_startup_program().global_block()
        src_op = dist_op_helper.get_cur_src_op()
        rank_id = dist_op_helper.get_rank_id()
        op_dist_attr = ctx.get_op_distributed_attr_for_program(src_op)
        assert op_dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(src_op))

        # check validation of inputs / outputs 
        assert 'Ids' in kwargs, "input [{}] is not given".format('Ids')
        assert 'W' in kwargs, "input [{}] is not given".format('W')
        assert 'Out' in kwargs, "output [{}] is not given".format('Out')

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
        Weight_var = main_block.var(kwargs['W'][0])
        Out_var = main_block.var(kwargs['Out'][0])

        # got dist attribute info
        embedding_row_dim_mapping = op_dist_attr.get_input_dims_mapping(
            Weight_var.name)[0]
        assert embedding_row_dim_mapping >= 0, "row_parallel_embedding's row should be divided by a specific mesh axis, but got [{}]".format(
            embedding_row_dim_mapping)
        process_mesh_shape = op_dist_attr.get_process_mesh().topology
        process_mesh_group = op_dist_attr.get_process_mesh().process_group

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in process_mesh_group:
            rank_id = _get_corresponding_rank(op_dist_attr.get_process_mesh(),
                                              rank_id)

        # A generalized method to caculate embedding offset using cartisian product
        relative_idx = _get_idx_in_axis(process_mesh_group, process_mesh_shape,
                                        embedding_row_dim_mapping, rank_id)

        per_part_size = Weight_var.shape[0]
        relative_idx = relative_idx * per_part_size

        # TODO caculate ring id 
        parallel_axis = embedding_row_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        # append op
        check_variable_and_dtype(Ids_var, 'input', ['int32', 'int64'],
                                 'c_embedding')

        intermediate_var_0 = main_block.create_var(
            name=unique_name.generate_with_ignorable_key(".".join(
                ["c_embedding", 'tmp'])),
            dtype=Weight_var.dtype,
            shape=Out_var.shape,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=Out_var.stop_gradient)

        # copy Out_var's dist_attr to intermediate_var_0's dist_attr
        copy_distributed_attr_for_var(op_dist_attr, intermediate_var_0, Out_var)

        check_variable_and_dtype(
            Out_var, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'c_allreduce_sum')

        c_embedding_op = main_block.append_op(
            type='c_embedding',
            inputs={'Ids': [Ids_var],
                    'W': [Weight_var]},
            outputs={'Out': [intermediate_var_0]},
            attrs={"start_index": relative_idx})

        # use_model_parallel
        c_allreduce_sum_op = main_block.append_op(
            type='c_allreduce_sum',
            inputs={'X': [intermediate_var_0]},
            outputs={'Out': [Out_var]},
            attrs={
                'ring_id': group.id,
                'use_calc_stream': True,
                'use_model_parallel': True,
            })

        # copy serial op's dist_attr to dist op's dist_attr
        copy_distributed_attr_for_dist_op(c_embedding_op, main_block,
                                          op_dist_attr)
        copy_distributed_attr_for_dist_op(c_allreduce_sum_op, main_block,
                                          op_dist_attr)

        # param initialization sync
        assert Weight_var.name not in dist_op_helper.already_init_sync_vars
        dist_op_helper.already_init_sync_vars.add(Weight_var.name)
        param = startup_block.var(Weight_var.name)
        param_dist_attr = ctx.get_tensor_distributed_attr_for_program(param)
        process_mesh = param_dist_attr.get_process_mesh()
        dim_mapping = param_dist_attr.get_dims_mapping()

        # NOTE all not splited axis should be presented in mesh 
        for axis, size in enumerate(process_mesh.topology):
            if size <= 1 or axis in dim_mapping:
                pass
            else:
                group_ranks = _get_comm_group(process_mesh.process_group,
                                              process_mesh.topology, axis,
                                              rank_id)
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

    @staticmethod
    def backward(ctx, *args, **kwargs):

        # by now the backward function only insert the gradient allreduce for dist op itself
        dist_op_helper = ctx.get_dist_op_helper()
        main_block = dist_op_helper.get_dst_main_program().global_block()
        backward_op = dist_op_helper.get_cur_src_op()
        rank_id = dist_op_helper.get_rank_id()
        dist_attr = ctx.get_op_distributed_attr_for_program(backward_op)
        assert dist_attr is not None, "backward op [{}] don't have dist attribute !".format(
            str(backward_op))

        # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
        if rank_id not in dist_attr.get_process_mesh().process_group:
            rank_id = _get_corresponding_rank(dist_attr.get_process_mesh(),
                                              rank_id)

        # check if need gradient allreduce
        need_gradient_allreduce = False

        assert 'Ids' in kwargs, "input [{}] is not given".format('Ids')
        assert 'W' in kwargs, "input [{}] is not given".format('W')
        assert 'Out@GRAD' in kwargs, "input [{}] is not given".format('Out')
        assert 'W@GRAD' in kwargs, "output [{}] is not given".format('W@GRAD')

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
        process_mesh = dist_attr.get_process_mesh()
        var_dim_mapping = dist_attr.get_input_dims_mapping(Ids_var.name)
        mesh_shape = process_mesh.topology
        batch_size_axis = var_dim_mapping[0]
        if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
            need_gradient_allreduce = True

            group_ranks = _get_comm_group(process_mesh.process_group,
                                          process_mesh.topology,
                                          batch_size_axis, rank_id)
            dp_degree = len(group_ranks)
            dp_group = new_process_group(group_ranks)

        if need_gradient_allreduce:
            W_Grad_var = main_block.var(kwargs['W@GRAD'][0])
            allreduce_op = main_block.append_op(
                type='c_allreduce_sum',
                inputs={'X': [W_Grad_var]},
                outputs={'Out': [W_Grad_var]},
                attrs={
                    'ring_id': dp_group.id,
                    'use_calc_stream': True,
                    OP_ROLE_KEY: OpRole.Backward
                })
            scale_op = main_block.append_op(
                type='scale',
                inputs={'X': W_Grad_var},
                outputs={'Out': W_Grad_var},
                attrs={'scale': 1.0 / dp_degree,
                       OP_ROLE_KEY: OpRole.Backward})
            main_block._sync_with_cpp()

            dims_mapping = ctx.get_tensor_distributed_attr_for_program(
                W_Grad_var).get_dims_mapping()
            process_mesh = dist_attr.get_process_mesh()
            for op in [allreduce_op, scale_op]:
                op_attr = OperatorDistributedAttribute(op, ctx)
                op_attr.set_process_mesh(process_mesh)
                op_attr.set_output_dims_mapping(W_Grad_var.name, dims_mapping)
                op_attr.set_input_dims_mapping(W_Grad_var.name, dims_mapping)
                ctx.set_op_distributed_attr_for_program(op, op_attr)


register_distributed_operator_impl("lookup_table_v2",
                                   DistributedEmbeddingImpl("row_parallel"))
register_distributed_operator_impl("c_embedding",
                                   DistributedEmbeddingImpl("row_parallel"))
