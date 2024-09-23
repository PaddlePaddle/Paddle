# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import re
import warnings

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.converter import Converter
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.static import global_scope

from .pass_base import PassBase, register_pass


@register_pass("auto_parallel_c_embedding_pass")
class AutoParallelCEmbeddingPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        main_program = self.get_attr("dist_program")
        ops = main_program.global_block().ops
        for i, op in enumerate(ops):
            if op.name() == 'pd_op.embedding':
                placements = op.operand(1).source().placements
                dim_map, partial_status = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements, op.operand(1).source().ndim
                    )
                )
                if dim_map[1] == -1 or dim_map[1] in partial_status:
                    warnings.warn(
                        "The c_embedding pass is only applicable to column-wise parallel `embedding` kernel."
                    )
                    return False
                else:
                    return True
        warnings.warn(
            "The c_embedding pass is only applicable to column-wise parallel `embedding` kernel."
        )
        return False

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        concrete_program = self.get_attr("concrete_program")
        ops = main_program.global_block().ops
        for i, op in enumerate(ops):
            if op.name() == 'pd_op.embedding':
                # replace embedding with c_embedding
                paddle.pir.set_insertion_point(op)
                num_embeddings = op.operand(1).source().type().shape[0]
                world_size = paddle.distributed.get_world_size()
                rank = paddle.distributed.get_rank()
                per_part_size = num_embeddings // world_size
                vocab_start_index = rank * per_part_size
                t_op = paddle._C_ops.c_embedding(
                    op.operand(1).source(),
                    op.operand(0).source(),
                    vocab_start_index,
                    num_embeddings,
                )
                t_op.get_defining_op().op_role = int(OpRole.Forward)
                new_op = t_op.get_defining_op()
                op.result(0).replace_all_uses_with(t_op)
                op.erase()

                # # input0 weight
                placements_input0 = new_op.operand(0).source().placements
                dim_map_input0, partial_status_input0 = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements_input0, new_op.operand(0).source().ndim
                    )
                )
                mp_axis = dim_map_input0[1]
                dim_map_input0 = [mp_axis, -1]
                dist_attr_input0 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.operand(0).source().process_mesh,
                        dim_map_input0,
                        partial_status_input0,
                    )
                )
                dist_type_input0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    new_op.operand(0).source().type(), dist_attr_input0
                )
                new_op.operand(0).source().set_type(dist_type_input0)

                # # input1 x
                placements_input1 = new_op.operand(1).source().placements
                dim_map_input1, partial_status_input1 = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements_input1, new_op.operand(1).source().ndim
                    )
                )
                dist_attr_input1 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.operand(1).source().process_mesh,
                        dim_map_input1,
                        partial_status_input1,
                    )
                )

                # output
                placements_out0 = new_op.results()[0].placements
                dim_map_out0, partial_status_out0 = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements_out0, new_op.results()[0].ndim
                    )
                )
                partial_status_out0 = {
                    mp_axis: paddle.base.core.ReduceType.kRedSum
                }
                dist_attr_out0 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.results()[0].process_mesh,
                        dim_map_out0,
                        partial_status_out0,
                    )
                )

                new_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        new_op.operand(0).source().process_mesh,
                        [dist_attr_input0, dist_attr_input1],
                        [dist_attr_out0],
                    )
                )

                # update builtin.parameter dims_map
                param_op = new_op.operand(0).source().get_defining_op()
                placements = param_op.results()[0].placements
                dim_map, partial_status = (
                    dist.auto_parallel.placement_type.to_dim_map(
                        placements, param_op.results()[0].ndim
                    )
                )
                dim_map = [mp_axis, -1]
                param_dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        param_op.results()[0].process_mesh,
                        dim_map,
                        partial_status,
                    )
                )
                param_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        param_op.results()[0].process_mesh,
                        [],
                        [param_dist_attr],
                    )
                )

                # update reshard dims_map
                for op in new_op.results()[0].all_used_ops():
                    placements_in = op.operand(0).source().placements
                    dim_map_in, partial_status_in = (
                        dist.auto_parallel.placement_type.to_dim_map(
                            placements_in, op.operand(0).source().ndim
                        )
                    )
                    partial_status_in = {
                        mp_axis: paddle.base.core.ReduceType.kRedSum
                    }
                    dist_attr_in = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            op.operand(0).source().process_mesh,
                            dim_map_in,
                            partial_status_in,
                        )
                    )

                    placements_out = op.results()[0].placements
                    dim_map_out, partial_status_out = (
                        dist.auto_parallel.placement_type.to_dim_map(
                            placements_out, op.results()[0].ndim
                        )
                    )
                    dist_attr_out = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            op.results()[0].process_mesh,
                            dim_map_out,
                            partial_status_out,
                        )
                    )

                    op.dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            op.operand(0).source().process_mesh,
                            [dist_attr_in],
                            [dist_attr_out],
                        )
                    )

                # update param
                dy_params = concrete_program.parameters[0]
                pattern = re.compile(r'embedding_.*\.w_0\.dist')
                for index, param in enumerate(dy_params):
                    if pattern.match(param.name):
                        dist_attr = {
                            "dims_mapping": param_dist_attr.dims_mapping,
                            "process_shape": param_dist_attr.process_mesh.shape,
                            "process_group": param_dist_attr.process_mesh.process_ids,
                        }
                        place = paddle.framework.CUDAPlace(
                            paddle.distributed.ParallelEnv().dev_id
                        )
                        scope_param = (
                            global_scope().var(param.name).get_tensor()
                        )
                        scope_param._share_data_with(
                            param.get_tensor().get_tensor()
                        )
                        sliced_param = Converter.slice_with_dist_attr(
                            param.numpy(), dist_attr
                        )
                        scope_param.set(sliced_param, place)
                        param.get_tensor()._clear()
                        concrete_program.parameters[0][index] = None
